import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNDualAttBiLSTM_MultiTask(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(TCNDualAttBiLSTM_MultiTask, self).__init__()
        
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
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
        
        # Multi-task Output heads (3 horizons: T+5, T+10, T+15)
        self.head_binary = nn.Linear(hidden_dim * 2, 3) # BCEWithLogits
        self.head_tri = nn.Linear(hidden_dim * 2, 9)    # CrossEntropy (3 horizons * 3 classes)
        self.head_risk = nn.Linear(hidden_dim * 2, 3)   # MSE, Sigmoid is applied here to bound to [0,1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        
        x_mean = torch.mean(x, dim=1) 
        f_weights = self.feature_attention(x_mean).unsqueeze(1)
        x = x * f_weights
        
        t_weights = F.softmax(self.temporal_attention(x), dim=1)
        x = x * t_weights 
        
        lstm_out, _ = self.bilstm(x)
        last_out = lstm_out[:, -1, :] 
        
        out_binary = self.head_binary(last_out)
        out_tri = self.head_tri(last_out)
        out_tri = out_tri.view(-1, 3, 3) # (batch, 3 horizons, 3 classes)
        out_risk = self.sigmoid(self.head_risk(last_out)) # Bound to [0,1]
        
        return out_binary, out_tri, out_risk
