import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPatchBiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3, patch_len: int = 5):
        super(CNNPatchBiLSTMAttention, self).__init__()
        
        self.patch_len = patch_len
        # Assuming seq_len = 30, patch_len = 5 -> num_patches = 6
        
        # 1. 1D-CNN (Patch Embedding)
        # Instead of sliding over all timesteps, we treat each patch as a flat vector 
        # or use Conv1D with stride = patch_len
        self.patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=patch_len, stride=patch_len),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # 2. BiLSTM for inter-patch dynamics
        self.bilstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 3. Temporal Attention over Patches
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # 4. Dense (Multi-step Output)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len=30, input_dim=4)
        
        # 1. Patching & CNN Embedding
        x = x.permute(0, 2, 1) # (batch, input_dim, seq_len)
        # Conv1d with kernel=5, stride=5 produces (batch, hidden_dim, num_patches=6)
        x = self.patch_embedding(x) 
        
        x = x.permute(0, 2, 1) # (batch, num_patches, hidden_dim)
        
        # 2. BiLSTM over Patches
        lstm_out, _ = self.bilstm(x) # (batch, num_patches, hidden_dim * 2)
        
        # 3. Temporal Attention
        attention_weights = F.softmax(self.attention(lstm_out), dim=1) # (batch, num_patches, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1) # (batch, hidden_dim * 2)
        
        # 4. Dense Output
        predictions = self.fc(context_vector)
        
        return predictions
