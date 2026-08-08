import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TCNDualAttTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super(TCNDualAttTransformer, self).__init__()
        
        # 1. TCN for local feature extraction
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 2. Feature Attention
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. Transformer Encoder (replaces BiLSTM + Temporal Attention)
        # Increased dropout to 0.2 to prevent overfitting on short sequences
        self.pos_encoder = PositionalEncoding(d_model=hidden_dim, dropout=0.2, max_len=200)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=4, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.2,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        
        # 1. TCN
        x = x.permute(0, 2, 1) # (batch, input_dim, seq_len)
        x = self.tcn(x)
        x = x.permute(0, 2, 1) # (batch, seq_len, hidden_dim)
        
        # 2. Feature Attention
        x_mean = torch.mean(x, dim=1) 
        f_weights = self.feature_attention(x_mean).unsqueeze(1) 
        x = x * f_weights 
        
        # 3. Transformer Encoder (batch_first=True)
        # However, our pos_encoder expects (seq_len, batch, dim)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        
        transformer_out = self.transformer_encoder(x) # (batch, seq_len, hidden_dim)
        
        # Aggregate temporal dimension using Mean Pooling instead of just the last token
        # This prevents over-reliance on a single step and improves stability
        pooled_out = torch.mean(transformer_out, dim=1)
        
        # 4. Dense (Multi-step Output)
        predictions = self.fc(pooled_out)
        
        return predictions
