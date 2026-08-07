import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNDualAttBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super(TCNDualAttBiLSTM, self).__init__()
        
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Feature Attention (Channel Attention)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Temporal Attention (applied before BiLSTM to weight timesteps)
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
        
        self.bilstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        
        # 1. TCN
        x = x.permute(0, 2, 1) # (batch, input_dim, seq_len)
        x = self.tcn(x)
        x = x.permute(0, 2, 1) # (batch, seq_len, hidden_dim)
        
        # 2. Feature Attention
        # Tính trọng số cho từng feature channel (dựa trên trung bình toàn chuỗi)
        # x_mean.shape: (batch_size, hidden_dim)
        x_mean = torch.mean(x, dim=1) 
        f_weights = self.feature_attention(x_mean).unsqueeze(1) # (batch, 1, hidden_dim)
        x = x * f_weights # Broadcast to (batch, seq_len, hidden_dim)
        
        # 3. Temporal Attention
        # Tính trọng số cho từng timestep
        t_weights = F.softmax(self.temporal_attention(x), dim=1) # (batch, seq_len, 1)
        # Khác với context vector (sum), ở đây ta chỉ scale các timestep theo trọng số
        # để BiLSTM học được mốc thời gian nào quan trọng.
        x = x * t_weights 
        
        # 4. BiLSTM
        lstm_out, _ = self.bilstm(x) # lstm_out: (batch, seq_len, hidden_dim * 2)
        
        # Lấy hidden state cuối cùng của 2 chiều
        # Hoặc dùng mean pooling để lấy thông tin toàn chuỗi
        # Ở đây ta dùng output tại timestep cuối cùng như chuẩn LSTM
        last_out = lstm_out[:, -1, :] 
        
        # 5. Dense (Multi-step Output)
        predictions = self.fc(last_out)
        
        return predictions
