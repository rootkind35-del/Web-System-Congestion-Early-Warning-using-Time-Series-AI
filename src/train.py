import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys
import argparse
import time

# Sửa lỗi ModuleNotFoundError: Thêm thư mục gốc vào PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.sg_tcn_lstm import MultiHorizonTCNLSTM
from src.models.bilstm_attention import BiLSTMAttention

def load_data(data_path: str, batch_size: int = 128):
    print(f"Đang tải dữ liệu từ: {data_path}")
    data = np.load(data_path)
    
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train.shape[-1]

def get_model(model_name: str, input_dim: int, device: torch.device):
    if model_name == "sg_tcn_lstm":
        print("Khởi tạo mô hình: SG-TCN-LSTM")
        model = MultiHorizonTCNLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3)
    elif model_name == "bilstm_attention":
        print("Khởi tạo mô hình: BiLSTM-Attention")
        model = BiLSTMAttention(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=3)
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_name}. Chọn 'sg_tcn_lstm' hoặc 'bilstm_attention'")
    
    return model.to(device)

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Môi trường huấn luyện: {device}")
    
    train_loader, val_loader, input_dim = load_data(args.data, args.batch_size)
    
    model = get_model(args.model, input_dim, device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    import pandas as pd
    
    print(f"Bắt đầu huấn luyện mô hình [{args.model}] trong {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    model_path = os.path.join(args.output_dir, f"best_{args.model}.pth")
    history_path = os.path.join(args.output_dir, f"history_{args.model}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    
    history_data = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Huấn luyện
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Đánh giá (Validation)
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
        print(f"Epoch {epoch+1:03d}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time_taken:.1f}s")
        
        history_data.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time_sec': time_taken
        })
        
        # Lưu checkpoint tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            
    # Lưu lịch sử huấn luyện ra file CSV
    pd.DataFrame(history_data).to_csv(history_path, index=False)
    print(f"Huấn luyện hoàn tất! Mô hình tốt nhất được lưu tại: {model_path}")
    print(f"Lịch sử huấn luyện được lưu tại: {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình chuẩn cho báo cáo khoa học.")
    
    default_data = os.path.join(project_root, "data", "processed", "processed_dataset.npz")
    default_model_dir = os.path.join(project_root, "models")
    
    parser.add_argument('--model', type=str, required=True, choices=['sg_tcn_lstm', 'bilstm_attention'], 
                        help='Chọn kiến trúc mạng để huấn luyện.')
    parser.add_argument('--data', type=str, default=default_data)
    parser.add_argument('--output_dir', type=str, default=default_model_dir)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    train_model(args)
