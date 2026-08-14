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
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add project root to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.sg_tcn_lstm import MultiHorizonTCNLSTM
from src.models.bilstm_attention import BiLSTMAttention
from src.models.baselines import StandardLSTM
from src.models.tcn_bilstm_attention import TCNBiLSTMAttention
from src.models.tcn_dualatt_bilstm import TCNDualAttBiLSTM
from src.models.cnn_patch_bilstm_attention import CNNPatchBiLSTMAttention
from src.models.tsmixer_dualatt_bilstm import TSMixerDualAttBiLSTM

def load_pt_data(name, batch_size=256, shuffle=False):
    path = os.path.join(project_root, "data", "processed", f"{name}.pt")
    X, y = torch.load(path)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, X.shape[-1]

def get_model(model_name: str, input_dim: int, device: torch.device):
    if model_name == "standard_lstm":
        return StandardLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "tcn_dualatt_bilstm":
        return TCNDualAttBiLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    elif model_name == "cnn_patch_bilstm_attention":
        return CNNPatchBiLSTMAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3, patch_len=5).to(device)
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
    
    # Univariate Inverse transform (CPU pct is the only column)
    y_true_actual = np.zeros_like(y_true)
    y_pred_actual = np.zeros_like(y_pred)
    
    for h in range(3):
        y_true_actual[:, h] = scaler.inverse_transform(y_true[:, h].reshape(-1, 1)).flatten()
        y_pred_actual[:, h] = scaler.inverse_transform(y_pred[:, h].reshape(-1, 1)).flatten()
        
    metrics = {}
    horizons = [5, 10, 15]
    for i, h in enumerate(horizons):
        mse = mean_squared_error(y_true_actual[:, i], y_pred_actual[:, i])
        mae = mean_absolute_error(y_true_actual[:, i], y_pred_actual[:, i])
        rmse = np.sqrt(mse)
        
        metrics[f"T+{h}_MSE"] = mse
        metrics[f"T+{h}_MAE"] = mae
        metrics[f"T+{h}_RMSE"] = rmse
        
    return metrics

def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Environment: {device}")
    
    train_loader, input_dim = load_pt_data("azure_train", args.batch_size, shuffle=True)
    val_loader, _ = load_pt_data("azure_val", args.batch_size, shuffle=False)
    test_azure_loader, _ = load_pt_data("azure_test", args.batch_size, shuffle=False)
    test_google_loader, _ = load_pt_data("google_test", args.batch_size, shuffle=False)
    
    scaler_path = os.path.join(project_root, "data", "processed", "azure_scaler.pkl")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    model_dir = args.output_dir
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\n================ MODEL: {args.model} ================")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        
    model = get_model(args.model, input_dim, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    run_model_path = os.path.join(model_dir, f"best_{args.model}.pth")
    
    patience = 5
    patience_counter = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
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
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time_taken:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), run_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    print("\n--- Evaluating Best Model ---")
    model.load_state_dict(torch.load(run_model_path))
    
    metrics_azure = evaluate_model(model, test_azure_loader, device, scaler)
    print("IN-DOMAIN TEST (AZURE):")
    for k, v in metrics_azure.items():
        print(f"  {k}: {v:.4f}")
        
    metrics_google = evaluate_model(model, test_google_loader, device, scaler)
    print("CROSS-DOMAIN TEST (GOOGLE ZERO-SHOT):")
    for k, v in metrics_google.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=[
        'standard_lstm', 'tcn_dualatt_bilstm', 'cnn_patch_bilstm_attention'
    ])
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, "models"))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    run_experiment(args)
