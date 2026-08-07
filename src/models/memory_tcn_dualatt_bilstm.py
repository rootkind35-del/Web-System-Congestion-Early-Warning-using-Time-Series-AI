import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ParametricMemory(nn.Module):
    def __init__(self, num_slots: int = 10, dim: int = 64):
        super(ParametricMemory, self).__init__()
        self.num_slots = num_slots
        self.dim = dim
        # Learnable memory matrix
        self.memory = nn.Parameter(torch.Tensor(num_slots, dim))
        nn.init.kaiming_uniform_(self.memory, a=math.sqrt(5))
        
    def forward(self, queries):
        # queries: (batch_size, seq_len, dim)
        # memory: (num_slots, dim)
        
        batch_size, seq_len, dim = queries.size()
        
        # Compute dot product attention between queries and memory slots
        # Reshape queries to (batch * seq_len, dim)
        q = queries.reshape(-1, dim)
        
        # Attention scores: (batch * seq_len, num_slots)
        attn_scores = F.softmax(torch.matmul(q, self.memory.T) / math.sqrt(dim), dim=-1)
        
        # Readout: (batch * seq_len, dim)
        readout = torch.matmul(attn_scores, self.memory)
        
        # Reshape back to (batch, seq_len, dim)
        readout = readout.view(batch_size, seq_len, dim)
        
        return readout

class MemoryTCNDualAttBiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3, num_memory_slots: int = 10):
        super(MemoryTCNDualAttBiLSTM, self).__init__()
        
        # 1. TCN for local feature extraction
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 2. Parametric Memory Module (Augmenting TCN output)
        self.memory = ParametricMemory(num_slots=num_memory_slots, dim=hidden_dim)
        
        # 3. Feature Attention
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 4. BiLSTM for sequential dynamics
        self.bilstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 5. Temporal Attention
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 6. Output layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        
        # 1. TCN
        x = x.permute(0, 2, 1) # (batch, input_dim, seq_len)
        x = self.tcn(x)
        x = x.permute(0, 2, 1) # (batch, seq_len, hidden_dim)
        
        # 2. Memory Augmentation
        memory_readout = self.memory(x)
        x = x + memory_readout # Residual connection with memory
        
        # 3. Feature Attention
        x_mean = torch.mean(x, dim=1) 
        f_weights = self.feature_attention(x_mean).unsqueeze(1) 
        x = x * f_weights 
        
        # 4. BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # 5. Temporal Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 6. Output
        predictions = self.fc(context_vector)
        
        return predictions
