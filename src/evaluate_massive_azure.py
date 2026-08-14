import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score, confusion_matrix
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.multitask_baselines import StandardLSTM_MultiTask
from src.train_all_massive_azure import AzureStreamingDataset

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure_parquet")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lab", "best_StandardLSTM_MultiTask_120E.pth")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    test_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "test_*.parquet")))
    # Limit to 1 file for quick testing report
    test_ds = AzureStreamingDataset(test_files, is_train=False, max_files=1)
    test_loader = DataLoader(test_ds, batch_size=4096, num_workers=2)
    
    model = StandardLSTM_MultiTask(input_dim=7).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        log.info(f"Loaded model from {MODEL_PATH}")
    else:
        log.error("Model file not found!")
        return
        
    model.eval()
    
    all_b_preds, all_b_true = [], []
    all_t_preds, all_t_true = [], []
    all_r_preds, all_r_true = [], []
    
    log.info("Evaluating on Test Set...")
    with torch.no_grad():
        for i, (X, y_b, y_t, y_r) in enumerate(test_loader):
            X, y_b, y_t, y_r = X.to(device), y_b.to(device), y_t.to(device), y_r.to(device)
            p_b, p_t, p_r = model(X)
            
            # Horizon T+5 (index 0)
            b_pred = (torch.sigmoid(p_b[:, 0]) > 0.5).cpu().numpy()
            b_true = y_b[:, 0].cpu().numpy()
            all_b_preds.extend(b_pred)
            all_b_true.extend(b_true)
            
            t_pred = torch.argmax(p_t[:, 0, :], dim=1).cpu().numpy() # p_t shape: (batch, 3_horizons, 3_classes)
            t_true = y_t[:, 0].cpu().numpy()
            all_t_preds.extend(t_pred)
            all_t_true.extend(t_true)
            
            all_r_preds.extend(p_r[:, 0].cpu().numpy())
            all_r_true.extend(y_r[:, 0].cpu().numpy())
            
            if i >= 200: # process ~800,000 samples for quick report
                break

    print("\n" + "="*50)
    print("TEST REPORT: StandardLSTM_MultiTask (Epoch 14 Snapshot)")
    print("="*50)
    
    print("\n[1] Binary Classification (Congestion Yes/No)")
    print(classification_report(all_b_true, all_b_preds, target_names=['Normal', 'Congestion'], zero_division=0))
    
    print("\n[2] Tri-level Classification (Normal/Warning/Critical)")
    print(classification_report(all_t_true, all_t_preds, target_names=['Normal', 'Warning', 'Critical'], zero_division=0))
    
    print("\n[3] Risk Score Regression")
    mae = mean_absolute_error(all_r_true, all_r_preds)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
