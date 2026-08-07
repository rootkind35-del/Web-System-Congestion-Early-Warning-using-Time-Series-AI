import os
import sys
import time
import numpy as np
import torch
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.bilstm_attention import BiLSTMAttention
from src.utils.thresholding import DynamicThresholdEMA
from src.utils.drift_detection import PageHinkleyDriftDetector
import pandas as pd
from scipy.signal import savgol_filter

class RealtimeSimulator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMAttention(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3).to(self.device)
        model_path = os.path.join(project_root, "models", "best_bilstm_attention.pth")
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        scaler_path = os.path.join(project_root, "data", "processed", "minmax_scaler.pkl")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        self.threshold = DynamicThresholdEMA(slo_latency=2000.0, alpha=0.1, k_sigma=3.0)
        self.drift_detector = PageHinkleyDriftDetector(min_instances=30, delta=0.05, threshold=50, alpha=0.9999)
        
        # Load EXACTLY the 11.11 spike from the real dataset (Row 451920 is Nov 10, 20:00)
        data_path = os.path.join(project_root, "data", "raw", "multi_year_web_metrics.csv")
        # Đọc header và nối với đoạn data cần thiết
        df = pd.read_csv(data_path, skiprows=range(1, 451920), nrows=2000)
        self.timestamps = df['timestamp'].values
        
        features = ['cpu_usage', 'memory_usage', 'req_rate', 'latency']
        
        # 1. Áp dụng chuẩn SG Filter y hệt như huấn luyện
        for col in features:
            df[col] = savgol_filter(df[col], window_length=15, polyorder=3)
            
        # 2. Lưu lại bản gốc (đã làm mượt) để hiển thị Actual CPU trên Dashboard
        self.actual_cpu_series = df['cpu_usage'].values.copy()
        
        # 3. Scale dữ liệu
        if self.scaler:
            self.scaled_data = self.scaler.transform(df[features])
        else:
            self.scaled_data = df[features].values
            
        self.total_steps = len(self.scaled_data)
        self.current_step = 0
        
    def inverse_scale_cpu(self, scaled_val):
        """Hàm biến đổi ngược từ scaled (0-1) về giá trị CPU thật (0-100%)"""
        if not self.scaler: return scaled_val
        dummy = np.zeros((1, 4))
        dummy[0, 0] = scaled_val
        return self.scaler.inverse_transform(dummy)[0, 0]

    def get_features(self, step):
        # Window of 30 past minutes (T-29 to T)
        window = self.scaled_data[step : step+30]
        X = np.zeros((1, 30, 4))
        X[0, :, :] = window
        return X

    def generate_stream(self, interval_sec=1.0):
        self.current_step = 0
        
        # Chạy từ bước 0 đến (total - 30 - 15) để đủ cửa sổ 30 quá khứ và 15 tương lai
        while self.current_step < self.total_steps - 45:
            current_time = self.timestamps[self.current_step + 29] # Thời điểm hiện tại T
            
            # Lấy features (T-29 đến T)
            X_np = self.get_features(self.current_step)
            X_tensor = torch.tensor(X_np, dtype=torch.float32).to(self.device)
            
            # Suy luận
            start_infer = time.time()
            with torch.no_grad():
                if self.device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        y_pred = self.model(X_tensor).cpu().numpy()[0]
                else:
                    y_pred = self.model(X_tensor).cpu().numpy()[0]
            infer_time_ms = (time.time() - start_infer) * 1000
            
            # Inverse transform the predictions back to CPU %
            pred_t5 = self.inverse_scale_cpu(y_pred[0])
            pred_t10 = self.inverse_scale_cpu(y_pred[1])
            pred_t15 = self.inverse_scale_cpu(y_pred[2])
            
            actual_cpu = self.actual_cpu_series[self.current_step + 29]
            
            # Tính toán Dynamic Threshold và Alert
            thresh_info = self.threshold.update_and_detect(actual_cpu, current_latency=45.0)
            ema_thresh = thresh_info['dynamic_cpu_threshold']
            is_congested_t5 = pred_t5 > ema_thresh
            is_congested_t10 = pred_t10 > ema_thresh
            
            # Tính sai số để đưa vào Concept Drift (Dùng T+5 để test)
            actual_t5 = self.actual_cpu_series[self.current_step + 29 + 5]
            error = abs(pred_t5 - actual_t5)
            drift_detected = self.drift_detector.update(error)
            
            yield {
                "time": str(current_time).split(' ')[1] if ' ' in str(current_time) else str(current_time),
                "step": self.current_step,
                "actual_cpu": float(actual_cpu),
                "pred_t5": float(pred_t5),
                "pred_t10": float(pred_t10),
                "pred_t15": float(pred_t15),
                "ema_threshold": float(ema_thresh),
                "alert_t5": bool(is_congested_t5),
                "alert_t10": bool(is_congested_t10),
                "drift": bool(drift_detected),
                "infer_time": float(infer_time_ms)
            }
            
            self.current_step += 1
            time.sleep(interval_sec)
