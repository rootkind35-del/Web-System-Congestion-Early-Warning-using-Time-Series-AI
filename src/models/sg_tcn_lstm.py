import torch
import torch.nn as nn

class MultiHorizonTCNLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        """
        Kiến trúc mạng lai TCN-LSTM dự báo đa chân trời (Multi-Horizon).
        
        :param input_dim: Số lượng đặc trưng đầu vào (VD: 4 - CPU, RAM, Req, Latency).
        :param hidden_dim: Kích thước state ẩn của LSTM và số filters của TCN.
        :param num_layers: Số tầng của mạng LSTM.
        :param output_dim: Số lượng chân trời dự báo (VD: 3 - 5, 10, 15 phút).
        """
        super(MultiHorizonTCNLSTM, self).__init__()
        
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
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
        
        # Đầu ra dự báo tương lai
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        # TCN yêu cầu channel ở chiều thứ 2: (batch_size, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        
        x = self.tcn(x)
        
        # Đưa về lại cho LSTM: (batch_size, seq_len, hidden_dim)
        x = x.permute(0, 2, 1)
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Chỉ lấy output tại bước thời gian cuối cùng của chuỗi (last timestep)
        last_out = lstm_out[:, -1, :]
        
        # Đưa qua tầng tuyến tính (Fully Connected) để ra dự báo đa mốc
        predictions = self.fc(last_out)
        return predictions
