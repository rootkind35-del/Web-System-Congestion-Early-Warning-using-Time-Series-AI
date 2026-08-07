import torch
import torch.nn as nn

class StandardLSTM(nn.Module):
    """
    Standard LSTM architecture without Bidirectional mapping or Attention mechanism.
    Used as a baseline model for comparative analysis.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        super(StandardLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False, # Standard unidirectional LSTM
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Maps the final hidden state directly to the multi-horizon predictions
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out.shape: (batch_size, seq_len, hidden_dim)
        
        # Take the output at the last timestep
        last_out = lstm_out[:, -1, :]
        
        predictions = self.fc(last_out)
        return predictions
