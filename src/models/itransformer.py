import torch
import torch.nn as nn
import math

class iTransformer(nn.Module):
    """
    iTransformer (2024): Applies self-attention on the variate (feature) dimension 
    instead of the time dimension.
    """
    def __init__(self, seq_len: int = 30, input_dim: int = 4, output_dim: int = 3, d_model: int = 64, n_heads: int = 4, e_layers: int = 2):
        super(iTransformer, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # In iTransformer, the time series of each variate is embedded into a token.
        # So we project seq_len to d_model.
        self.project_in = nn.Linear(seq_len, d_model)
        
        # Transformer Encoder applied across the feature dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        # We need to project the d_model back to our multi-horizon output.
        # Since we are predicting CPU (feature 0), we can extract the CPU token and project it.
        self.project_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        
        # Invert: Treat features as tokens
        x_inv = x.permute(0, 2, 1) # (batch_size, input_dim, seq_len)
        
        # Embed each feature's full time series into a d_model vector
        enc_in = self.project_in(x_inv) # (batch_size, input_dim, d_model)
        
        # Apply Transformer Encoder (Attention across features)
        enc_out = self.encoder(enc_in) # (batch_size, input_dim, d_model)
        
        # We are only forecasting CPU load, which is the 0-th feature.
        # Extract the contextualized representation of the CPU feature.
        cpu_repr = enc_out[:, 0, :] # (batch_size, d_model)
        
        # Project to target horizons
        out = self.project_out(cpu_repr) # (batch_size, output_dim)
        
        return out
