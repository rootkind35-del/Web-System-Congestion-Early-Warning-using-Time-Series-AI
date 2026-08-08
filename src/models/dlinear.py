import torch
import torch.nn as nn

class MovingAverage(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear (2023): Simple linear model following decomposition.
    """
    def __init__(self, seq_len: int = 30, output_dim: int = 3, input_dim: int = 4):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # Decomp
        self.decompsition = SeriesDecomp(kernel_size=5)
        
        # Linear layers for Trend and Remainder
        # We predict output_dim timesteps for the CPU channel (index 0)
        # But DLinear typically maps seq_len to pred_len directly per channel.
        # Here we have 4 input channels, but we only want to predict 3 future steps for CPU load.
        # We can use a linear layer that takes the entire flattened sequence or independent channels.
        # Standard DLinear processes each channel independently (Channel Independence).
        # We will process CPU channel (channel 0) to predict future CPU.
        self.Linear_Trend = nn.Linear(seq_len, output_dim)
        self.Linear_Seasonal = nn.Linear(seq_len, output_dim)
        
    def forward(self, x):
        # x: [Batch, seq_len, input_dim]
        # Decompose
        seasonal_init, trend_init = self.decompsition(x)
        
        # We only care about predicting CPU load (assumed to be feature index 0)
        # If we want to utilize other features, DLinear can be expanded to cross-channel, 
        # but pure DLinear is channel-independent.
        # For fairness in multivariate forecasting where other features help, 
        # we will use a flattened approach or project all features.
        
        # Let's use all features flattened to predict the 3 horizons to be fair to multivariate setups.
        # Actually, let's stick to DLinear's channel independence for the CPU channel as the pure DLinear does.
        # Wait, if we only use CPU channel, we ignore RAM, Req Rate, Latency.
        # To make it a Multivariate DLinear that predicts 1 target, we map [seq_len * input_dim] -> output_dim
        
        pass

# Redefining Multivariate DLinear for 1 Target Variable (CPU Load)
class MultivariateDLinear(nn.Module):
    def __init__(self, seq_len: int = 30, input_dim: int = 4, output_dim: int = 3):
        super(MultivariateDLinear, self).__init__()
        self.decomp = SeriesDecomp(kernel_size=5)
        self.linear_trend = nn.Linear(seq_len * input_dim, output_dim)
        self.linear_seasonal = nn.Linear(seq_len * input_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        seasonal, trend = self.decomp(x)
        
        seasonal = seasonal.reshape(seasonal.size(0), -1) # (batch, seq_len * input_dim)
        trend = trend.reshape(trend.size(0), -1) # (batch, seq_len * input_dim)
        
        seasonal_output = self.linear_seasonal(seasonal)
        trend_output = self.linear_trend(trend)
        
        return seasonal_output + trend_output
