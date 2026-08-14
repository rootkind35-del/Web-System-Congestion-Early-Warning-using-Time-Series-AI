import os
import sys
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import classification_report, mean_absolute_error, r2_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.tcn_multitask import TCNDualAttBiLSTM_MultiTask
from src.models.multitask_baselines import StandardLSTM_MultiTask, SGTCNLSTM_MultiTask, BiLSTMAttention_MultiTask, Transformer_MultiTask
from src.train_all_massive_azure import AzureStreamingDataset

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure_parquet_3GB")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "lab")

def measure_inference_time(model, dummy_input, device, num_runs=50):
    model.eval()
    dummy_input = dummy_input.to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    start = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy_input)
            
    end = time.perf_counter()
    avg_ms = ((end - start) / num_runs) * 1000.0
    return avg_ms

def evaluate_model(model_name, model, test_loader, device):
    model_path = os.path.join(MODEL_SAVE_DIR, f"best_{model_name}_120E.pth")
    if not os.path.exists(model_path):
        log.warning(f"Model weights not found for {model_name} at {model_path}. Skipping.")
        return None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Measure footprint
    torch.cuda.reset_peak_memory_stats(device)
    dummy_batch = torch.randn(1, 30, 7) # Batch size 1 for pure inference time
    inf_time = measure_inference_time(model, dummy_batch, device)
    max_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
    
    all_b_preds, all_b_true = [], []
    all_t_preds, all_t_true = [], []
    all_r_preds, all_r_true = [], []
    
    log.info(f"Evaluating {model_name}...")
    with torch.no_grad():
        for i, (X, y_b, y_t, y_r) in enumerate(test_loader):
            X, y_b, y_t, y_r = X.to(device), y_b.to(device), y_t.to(device), y_r.to(device)
            p_b, p_t, p_r = model(X)
            
            b_pred = (torch.sigmoid(p_b[:, 0]) > 0.5).cpu().numpy()
            b_true = y_b[:, 0].cpu().numpy()
            all_b_preds.extend(b_pred)
            all_b_true.extend(b_true)
            
            t_pred = torch.argmax(p_t[:, 0, :], dim=1).cpu().numpy()
            t_true = y_t[:, 0].cpu().numpy()
            all_t_preds.extend(t_pred)
            all_t_true.extend(t_true)
            
            all_r_preds.extend(p_r[:, 0].cpu().numpy())
            all_r_true.extend(y_r[:, 0].cpu().numpy())
            
            if i >= 100: # Limit eval samples to ~400k for speed
                break
                
    # Calculate Metrics
    # Binary FPR
    tn, fp, fn, tp = confusion_matrix(all_b_true, all_b_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    report = classification_report(all_b_true, all_b_preds, output_dict=True, zero_division=0)
    b_f1 = report['1.0']['f1-score'] if '1.0' in report else 0
    
    # R2 Score for Regression
    r2 = r2_score(all_r_true, all_r_preds)
    mae = mean_absolute_error(all_r_true, all_r_preds)
    
    return {
        'Model': model_name,
        'F1_Binary': b_f1,
        'FPR_False_Alarms': fpr,
        'R2_Score': r2,
        'MAE': mae,
        'Inference_ms': inf_time,
        'VRAM_MB': max_vram
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    test_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "test_*.parquet")))
    test_ds = AzureStreamingDataset(test_files, is_train=False, max_files=1)
    test_loader = DataLoader(test_ds, batch_size=4096, num_workers=2)
    
    models = {
        'StandardLSTM_MultiTask': StandardLSTM_MultiTask(input_dim=7),
        'SGTCNLSTM_MultiTask': SGTCNLSTM_MultiTask(input_dim=7),
        'BiLSTMAttention_MultiTask': BiLSTMAttention_MultiTask(input_dim=7),
        'Transformer_MultiTask': Transformer_MultiTask(input_dim=7),
        'TCNDualAttBiLSTM_MultiTask': TCNDualAttBiLSTM_MultiTask(input_dim=7)
    }
    
    results = []
    for name, model in models.items():
        res = evaluate_model(name, model, test_loader, device)
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("FULL BENCHMARK REPORT")
        print("="*80)
        print(df.to_string(index=False))
        
        # Save to CSV
        out_csv = os.path.join(PROJECT_ROOT, "data", "reports", "full_benchmark_results.csv")
        df.to_csv(out_csv, index=False)
        print(f"\nResults saved to {out_csv}")
    else:
        print("No models were evaluated.")

if __name__ == "__main__":
    main()
