import numpy as np

class DynamicThresholdEMA:
    """
    Thuật toán Ngưỡng Nghẽn Động dựa trên Trung bình trượt hàm mũ (EMA).
    """
    def __init__(self, slo_latency: float = 2000.0, alpha: float = 0.1, k_sigma: float = 3.0):
        self.slo_latency = slo_latency
        self.alpha = alpha
        self.k_sigma = k_sigma
        
        self.ema_cpu = None
        self.var_cpu = 0.0

    def update_and_detect(self, current_cpu: float, current_latency: float) -> dict:
        """
        Cập nhật EMA và kiểm tra điểm dữ liệu có phải là nghẽn không.
        """
        if self.ema_cpu is None:
            self.ema_cpu = current_cpu
            self.var_cpu = 0.0
            return {"congested": False, "dynamic_cpu_threshold": current_cpu}
            
        delta = current_cpu - self.ema_cpu
        self.ema_cpu = self.ema_cpu + self.alpha * delta
        self.var_cpu = (1 - self.alpha) * (self.var_cpu + self.alpha * delta**2)
        std_cpu = np.sqrt(self.var_cpu)
        
        threshold = self.ema_cpu + self.k_sigma * std_cpu
        
        # Điều kiện khắt khe: CPU vượt ngưỡng VÀ hệ thống phản hồi chậm
        is_cpu_high = current_cpu > threshold
        is_latency_high = current_latency > self.slo_latency
        
        return {
            "congested": is_cpu_high and is_latency_high,
            "is_cpu_high": is_cpu_high,
            "is_latency_high": is_latency_high,
            "dynamic_cpu_threshold": threshold,
            "ema_cpu": self.ema_cpu,
            "std_cpu": std_cpu
        }
