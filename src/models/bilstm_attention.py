import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 3):
        """
        Kiến trúc mạng BiLSTM-Attention dự báo đa chân trời (Multi-Horizon).
        Nhạy bén với các mốc thời gian bùng nổ lưu lượng (Flash Crowds).
        
        :param input_dim: Số lượng đặc trưng đầu vào (VD: 4 - CPU, RAM, Req, Latency).
        :param hidden_dim: Kích thước state ẩn của BiLSTM (thực tế sẽ là hidden_dim * 2).
        :param num_layers: Số tầng của mạng LSTM.
        :param output_dim: Số lượng chân trời dự báo (VD: 3 - 5, 10, 15 phút).
        """
        super(BiLSTMAttention, self).__init__()
        
        # Mạng LSTM 2 chiều (Bidirectional = True)
        self.bilstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Self-Attention Layer
        # Kích thước đầu ra của BiLSTM là hidden_dim * 2
        self.attention = nn.Linear(hidden_dim * 2, 1)
        
        # Đầu ra dự báo tương lai
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x.shape: (batch_size, seq_len, input_dim)
        
        # Đi qua BiLSTM
        lstm_out, _ = self.bilstm(x) 
        # lstm_out.shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Cơ chế Self-Attention: Gán trọng số cho từng mốc thời gian (timestep)
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        # attention_weights.shape: (batch_size, seq_len, 1)
        
        # Nhân trọng số attention với output của LSTM
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        # context_vector.shape: (batch_size, hidden_dim * 2)
        
        # Đưa qua tầng tuyến tính (Fully Connected) để dự báo
        predictions = self.fc(context_vector)
        # predictions.shape: (batch_size, output_dim)
        
        return predictions
