import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Add project root to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.sg_tcn_lstm import MultiHorizonTCNLSTM
from src.models.bilstm_attention import BiLSTMAttention
from src.models.baselines import StandardLSTM
from src.models.tcn_bilstm_attention import TCNBiLSTMAttention
from src.models.tcn_gru_attention import TCNGRUAttention
from src.models.tcn_dualatt_bilstm import TCNDualAttBiLSTM
from src.models.tcn_dualatt_transformer import TCNDualAttTransformer
from src.models.cnn_patch_bilstm_attention import CNNPatchBiLSTMAttention
from src.models.tsmixer_dualatt_bilstm import TSMixerDualAttBiLSTM
from src.models.memory_tcn_dualatt_bilstm import MemoryTCNDualAttBiLSTM
from src.models.dlinear import MultivariateDLinear
from src.models.itransformer import iTransformer

def load_data(data_path: str, batch_size: int = 256):
    print(f"Loading preprocessed dataset from: {data_path}")
    data = np.load(data_path)
    
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.float32)
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, X_train.shape[-1]

def get_model(model_name: str, input_dim: int, device: torch.device):
    if model_name == "sg_tcn_lstm":
        return MultiHorizonTCNLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "bilstm_attention":
        return BiLSTMAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "standard_lstm":
        return StandardLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "tcn_bilstm_attention":
        return TCNBiLSTMAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "tcn_gru_attention":
        return TCNGRUAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "tcn_dualatt_bilstm":
        return TCNDualAttBiLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "tcn_dualatt_transformer":
        return TCNDualAttTransformer(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "cnn_patch_bilstm_attention":
        return CNNPatchBiLSTMAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3, patch_len=5).to(device)
    elif model_name == "tsmixer_dualatt_bilstm":
        return TSMixerDualAttBiLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3, seq_len=30).to(device)
    elif model_name == "memory_tcn_dualatt_bilstm":
        return MemoryTCNDualAttBiLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3, num_memory_slots=10).to(device)
    elif model_name == "dlinear":
        return MultivariateDLinear(seq_len=30, input_dim=input_dim, output_dim=3).to(device)
    elif model_name == "itransformer":
        return iTransformer(seq_len=30, input_dim=input_dim, output_dim=3, d_model=64, n_heads=4, e_layers=2).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def evaluate_model(model, loader, device, scaler):
    model.eval()
    y_true_list = []
    y_pred_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            # FP16 inference
            with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):
                outputs = model(X_batch)
            y_pred_list.append(outputs.cpu().numpy())
            y_true_list.append(y_batch.numpy())
            
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    # We must inverse transform predictions back to actual CPU load percentages
    # to evaluate operational errors correctly.
    # Scaler has 4 columns: cpu_usage, ram_usage, req_rate, latency_ms.
    # Scaler was fit on df_smoothed[numeric_cols].
    # CPU is index 0.
    y_true_actual = np.zeros_like(y_true)
    y_pred_actual = np.zeros_like(y_pred)
    
    # Inverse scaling column-wise
    for h in range(3):
        dummy_true = np.zeros((len(y_true), 4))
        dummy_pred = np.zeros((len(y_pred), 4))
        dummy_true[:, 0] = y_true[:, h]
        dummy_pred[:, 0] = y_pred[:, h]
        y_true_actual[:, h] = scaler.inverse_transform(dummy_true)[:, 0]
        y_pred_actual[:, h] = scaler.inverse_transform(dummy_pred)[:, 0]
        
    metrics = {}
    horizons = [5, 10, 15]
    for i, h in enumerate(horizons):
        mse = mean_squared_error(y_true_actual[:, i], y_pred_actual[:, i])
        mae = mean_absolute_error(y_true_actual[:, i], y_pred_actual[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_actual[:, i], y_pred_actual[:, i])
        
        metrics[f"T+{h}_MSE"] = mse
        metrics[f"T+{h}_MAE"] = mae
        metrics[f"T+{h}_RMSE"] = rmse
        metrics[f"T+{h}_R2"] = r2
        
    return metrics

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Environment: {device}")
    
    train_loader, val_loader, test_loader, input_dim = load_data(args.data, args.batch_size)
    
    scaler_path = os.path.join(project_root, "data", "processed", "minmax_scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    model_dir = args.output_dir
    os.makedirs(model_dir, exist_ok=True)
    
    runs_stats = []
    
    for run in range(args.runs):
        print(f"\n--- Model: {args.model} | Run {run+1}/{args.runs} ---")
        
        # Set random seeds for reproducibility and run statistics
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)
            
        model = get_model(args.model, input_dim, device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        best_val_loss = float('inf')
        run_model_path = os.path.join(model_dir, f"best_{args.model}_run{run}.pth")
        
        # Early stopping helper
        patience = 15
        patience_counter = 0
        
        for epoch in range(args.epochs):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                
                # FP16 Mixed Precision
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                else:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            
            time_taken = time.time() - start_time
            print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time_taken:.1f}s")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), run_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        # Load best weights of this run for testing
        model.load_state_dict(torch.load(run_model_path))
        test_metrics = evaluate_model(model, test_loader, device, scaler)
        print(f"Run {run+1} Test Metrics:")
        for k, v in test_metrics.items():
            print(f" - {k}: {v:.6f}")
        runs_stats.append(test_metrics)
        
        # Save the very first run model as the default production model for demo
        if run == 0:
            default_model_path = os.path.join(model_dir, f"best_{args.model}.pth")
            torch.save(model.state_dict(), default_model_path)
            
    # Calculate statistics across multiple runs
    df_stats = pd.DataFrame(runs_stats)
    mean_stats = df_stats.mean()
    std_stats = df_stats.std()
    
    summary = {}
    for col in df_stats.columns:
        summary[f"{col}_mean"] = mean_stats[col]
        summary[f"{col}_std"] = std_stats[col]
        
    summary_path = os.path.join(model_dir, f"summary_stats_{args.model}.json")
    pd.Series(summary).to_json(summary_path, indent=4)
    
    print(f"\n================ SUMMARY FOR {args.model} ================")
    for col in df_stats.columns:
        print(f"{col}: {mean_stats[col]:.6f} ± {std_stats[col]:.6f}")
    print(f"Saved summary to: {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-run training with strict evaluation.")
    
    default_data = os.path.join(project_root, "data", "processed", "processed_dataset.npz")
    default_model_dir = os.path.join(project_root, "models")
    
    parser.add_argument('--model', type=str, required=True, choices=[
        'standard_lstm', 'sg_tcn_lstm', 'bilstm_attention', 
        'tcn_bilstm_attention', 'tcn_gru_attention', 'tcn_dualatt_bilstm',
        'tcn_dualatt_transformer', 'cnn_patch_bilstm_attention',
        'tsmixer_dualatt_bilstm', 'memory_tcn_dualatt_bilstm',
        'dlinear', 'itransformer'
    ])
    parser.add_argument('--data', type=str, default=default_data)
    parser.add_argument('--output_dir', type=str, default=default_model_dir)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512) # Larger batch size to speed up training
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    run_experiment(args)
