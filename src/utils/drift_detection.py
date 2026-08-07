class PageHinkleyDriftDetector:
    """
    Thuật toán Page-Hinkley để phát hiện Concept Drift (Trôi dạt phân phối)
    dành cho luồng dữ liệu (Data Stream) thời gian thực.
    """
    def __init__(self, min_instances=30, delta=0.005, threshold=50.0, alpha=0.9999):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        
        self.x_mean = 0.0
        self.sum_T = 0.0
        self.count = 0

    def update(self, x: float) -> bool:
        """
        Cập nhật thuật toán với điểm dữ liệu mới (thường là Sai số - Loss).
        Trả về True nếu phát hiện Concept Drift.
        """
        self.count += 1
        self.x_mean = self.x_mean + (x - self.x_mean) / self.count
        self.sum_T = self.alpha * self.sum_T + (x - self.x_mean - self.delta)
        
        if self.sum_T < 0:
            self.sum_T = 0.0
            
        if self.count > self.min_instances and self.sum_T > self.threshold:
            self.reset()
            return True
            
        return False

    def reset(self):
        self.x_mean = 0.0
        self.sum_T = 0.0
        self.count = 0
