import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNGRUAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super(TCNGRUAttention, self).__init__()
        
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.bigru = nn.GRU(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Temporal Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        
        gru_out, _ = self.bigru(x) 
        
        attention_weights = F.softmax(self.attention(gru_out), dim=1)
        context_vector = torch.sum(attention_weights * gru_out, dim=1)
        
        predictions = self.fc(context_vector)
        return predictions
