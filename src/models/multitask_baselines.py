import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Standard LSTM Multi-task
class StandardLSTM_MultiTask(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(StandardLSTM_MultiTask, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.head_binary = nn.Linear(hidden_dim, 3)
        self.head_tri = nn.Linear(hidden_dim, 9)
        self.head_risk = nn.Linear(hidden_dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out_binary = self.head_binary(last_out)
        out_tri = self.head_tri(last_out).view(-1, 3, 3)
        out_risk = self.sigmoid(self.head_risk(last_out))
        return out_binary, out_tri, out_risk


# 2. SG-TCN-LSTM Multi-task
class SGTCNLSTM_MultiTask(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(SGTCNLSTM_MultiTask, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.head_binary = nn.Linear(hidden_dim, 3)
        self.head_tri = nn.Linear(hidden_dim, 9)
        self.head_risk = nn.Linear(hidden_dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out_binary = self.head_binary(last_out)
        out_tri = self.head_tri(last_out).view(-1, 3, 3)
        out_risk = self.sigmoid(self.head_risk(last_out))
        return out_binary, out_tri, out_risk


# 3. BiLSTM-Attention Multi-task
class BiLSTMAttention_MultiTask(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(BiLSTMAttention_MultiTask, self).__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1)
        )
        self.head_binary = nn.Linear(hidden_dim * 2, 3)
        self.head_tri = nn.Linear(hidden_dim * 2, 9)
        self.head_risk = nn.Linear(hidden_dim * 2, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        
        out_binary = self.head_binary(context)
        out_tri = self.head_tri(context).view(-1, 3, 3)
        out_risk = self.sigmoid(self.head_risk(context))
        return out_binary, out_tri, out_risk


# 4. Transformer Multi-task (Heavyweight SOTA)
class Transformer_MultiTask(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, nhead: int = 4):
        super(Transformer_MultiTask, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.head_binary = nn.Linear(hidden_dim, 3)
        self.head_tri = nn.Linear(hidden_dim, 9)
        self.head_risk = nn.Linear(hidden_dim, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        
        # Pass through Transformer Encoder
        out = self.transformer_encoder(x)
        
        # Take the last time step representation
        last_out = out[:, -1, :]
        
        out_binary = self.head_binary(last_out)
        out_tri = self.head_tri(last_out).view(-1, 3, 3)
        out_risk = self.sigmoid(self.head_risk(last_out))
        return out_binary, out_tri, out_risk

