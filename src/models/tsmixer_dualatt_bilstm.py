import torch
import torch.nn as nn
import torch.nn.functional as F

class TSMixerBlock(nn.Module):
    def __init__(self, seq_len: int, input_dim: int, dropout: float = 0.1):
        super(TSMixerBlock, self).__init__()
        # Time-mixing
        self.norm1 = nn.BatchNorm1d(seq_len)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature-mixing
        self.norm2 = nn.BatchNorm1d(seq_len)
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        
        # Time-mixing (operates on seq_len dimension)
        res = x
        x_norm = self.norm1(x)
        x_time = x_norm.permute(0, 2, 1) # (batch, input_dim, seq_len)
        x_time = self.time_mlp(x_time)
        x_time = x_time.permute(0, 2, 1) # (batch, seq_len, input_dim)
        x = res + x_time
        
        # Feature-mixing (operates on input_dim dimension)
        res = x
        x_norm = self.norm2(x)
        x_feat = self.feature_mlp(x_norm)
        x = res + x_feat
        
        return x

class TSMixerDualAttBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3, seq_len: int = 30):
        super(TSMixerDualAttBiLSTM, self).__init__()
        
        # 1. TS-Mixer for Spatiotemporal extraction
        self.ts_mixer = TSMixerBlock(seq_len=seq_len, input_dim=input_dim)
        
        # Projection to hidden_dim for BiLSTM
        self.projection = nn.Linear(input_dim, hidden_dim)
        
        # 2. Feature Attention
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 3. BiLSTM for sequential dynamics
        self.bilstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 4. Temporal Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 5. Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        
        # 1. TS-Mixer
        x = self.ts_mixer(x) # (batch, seq_len, input_dim)
        x = self.projection(x) # (batch, seq_len, hidden_dim)
        
        # 2. Feature Attention
        x_mean = torch.mean(x, dim=1) 
        f_weights = self.feature_attention(x_mean).unsqueeze(1) 
        x = x * f_weights 
        
        # 3. BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # 4. Temporal Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 5. Output
        predictions = self.fc(context_vector)
        
        return predictions
