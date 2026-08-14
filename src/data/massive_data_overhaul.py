import os
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ─── HYPERPARAMS ────────────────────────────────────────────────────────
SEQ_LEN = 30
HORIZONS = [5, 10, 15]
TRAIN_RATIO = 0.8
NOISE_RATIO = 0.20 # 20% noise injection

NUM_SAMPLES_PER_DOMAIN = 5000

def forward_mm1(req_rate):
    """Domain 2 (Web): NASA -> Simulate CPU/RAM"""
    cpu = (req_rate / 6000.0) * 100.0 + np.random.normal(0, 2, len(req_rate))
    cpu = np.clip(cpu, 0, 100)
    ram = 30 + (cpu * 0.45) + np.random.normal(0, 1.5, len(req_rate))
    ram = np.clip(ram, 0, 100)
    net = req_rate * np.random.uniform(1.2, 2.5, len(req_rate))
    disk = 10 + req_rate * 0.05 + np.exp(np.clip((ram - 90)/2.0, 0, 5))
    
    latency = 45.0 + 6 * np.exp(np.clip((cpu - 80)/4.0, -5, 10))
    error_rate = np.clip((latency - 3000)/5000.0, 0, 1)*100.0 + np.exp(np.clip((cpu-98)/0.5, 0, 5))
    error_rate = np.clip(error_rate, 0, 100)
    return cpu, ram, disk, net, latency, error_rate

def inverse_mm1(cpu):
    """Domain 1 (Cloud): Azure -> Simulate Request/Error"""
    # Reverse engineering request rate from CPU
    req_rate = (cpu / 100.0) * 6000.0 + np.random.normal(0, 50, len(cpu))
    req_rate = np.clip(req_rate, 10, 10000)
    ram = 30 + (cpu * 0.45) + np.random.normal(0, 1.5, len(cpu))
    ram = np.clip(ram, 0, 100)
    net = req_rate * np.random.uniform(1.2, 2.5, len(cpu))
    disk = 10 + req_rate * 0.05 + np.exp(np.clip((ram - 90)/2.0, 0, 5))
    
    latency = 45.0 + 6 * np.exp(np.clip((cpu - 80)/4.0, -5, 10))
    error_rate = np.clip((latency - 3000)/5000.0, 0, 1)*100.0 + np.exp(np.clip((cpu-98)/0.5, 0, 5))
    error_rate = np.clip(error_rate, 0, 100)
    return req_rate, ram, disk, net, latency, error_rate

def generate_domain_data(domain_name):
    N = NUM_SAMPLES_PER_DOMAIN
    log.info(f"Generating massive dataset for {domain_name}...")
    try:
        timestamps = np.arange(N)
        
        if domain_name == "cloud_azure":
            base_cpu = np.clip(np.random.normal(40, 15, N) + np.sin(np.linspace(0, 100, N))*20, 0, 100)
            req_rate, ram, disk, net, lat, err = inverse_mm1(base_cpu)
            cpu = base_cpu
        elif domain_name == "web_nasa":
            base_req = np.clip(np.random.normal(2000, 800, N) + np.sin(np.linspace(0, 50, N))*1000, 10, None)
            cpu, ram, disk, net, lat, err = forward_mm1(base_req)
            req_rate = base_req
        elif domain_name == "micro_rs_anomic":
            req_rate = np.random.lognormal(mean=7, sigma=1, size=N)
            cpu, ram, disk, net, lat, err = forward_mm1(req_rate)
            lat += np.random.exponential(100, N) * (np.random.rand(N) > 0.95)
        elif domain_name == "micro_train_ticket":
            req_rate = np.random.normal(3000, 1500, N)
            req_rate = np.clip(req_rate, 10, None)
            cpu, ram, disk, net, lat, err = forward_mm1(req_rate)
            err += np.random.exponential(5, N) * (np.random.rand(N) > 0.98)
            err = np.clip(err, 0, 100)

        df = pd.DataFrame({
            'timestamp': timestamps,
            'Request_rate': req_rate.astype(np.float32),
            'CPU_usage': cpu.astype(np.float32),
            'Memory_usage': ram.astype(np.float32),
            'Disk_IO': disk.astype(np.float32),
            'Network_IO': net.astype(np.float32),
            'Response_time': lat.astype(np.float32),
            'Error_Rate_5xx': err.astype(np.float32)
        })
        return df
    except Exception as e:
        import traceback
        log.error(f"Error in generation: {e}")
        log.error(traceback.format_exc())
        sys.exit(1)

def inject_noise(values, noise_ratio=0.20):
    """Trộn 20% nhiễu vào dữ liệu train như yêu cầu của user."""
    log.info(f"Injecting {noise_ratio*100}% noise into training data...")
    noisy_values = values.copy()
    N, D = noisy_values.shape
    
    # Số lượng điểm ảnh hưởng
    num_noisy_points = int(N * D * noise_ratio)
    
    # 1. Random Gaussian Noise
    idx_row = np.random.randint(0, N, num_noisy_points // 2)
    idx_col = np.random.randint(0, D, num_noisy_points // 2)
    noisy_values[idx_row, idx_col] += np.random.normal(0, np.std(values, axis=0)[idx_col], len(idx_row))
    
    # 2. Extreme Spikes (Anomaly injection)
    idx_row = np.random.randint(0, N, num_noisy_points // 4)
    idx_col = np.random.randint(0, D, num_noisy_points // 4)
    noisy_values[idx_row, idx_col] *= np.random.uniform(2.0, 5.0, len(idx_row))
    
    # 3. Missing values (Dropped out to 0)
    idx_row = np.random.randint(0, N, num_noisy_points // 4)
    idx_col = np.random.randint(0, D, num_noisy_points // 4)
    noisy_values[idx_row, idx_col] = 0.0
    
    return noisy_values

def build_windows(values, seq_len, horizons):
    max_h = max(horizons)
    total_len = seq_len + max_h
    N = len(values) - total_len + 1
    
    X = np.zeros((N, seq_len, 7), dtype=np.float32)
    y = np.zeros((N, len(horizons)), dtype=np.float32)
    
    for i in range(N):
        X[i] = values[i : i + seq_len]
        # Target: Predict CPU (col 2)
        y[i] = [values[i + seq_len - 1 + h, 2] for h in horizons]
        
    return X, y

def process_domain(domain_name):
    df = generate_domain_data(domain_name)
    N = len(df)
    n_train = int(N * TRAIN_RATIO)
    
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    
    # Scaling
    scaler = MinMaxScaler()
    # Fit only on clean train
    train_clean = train_df.drop(columns=['timestamp']).values
    scaler.fit(train_clean)
    
    # Transform test
    test_vals = scaler.transform(test_df.drop(columns=['timestamp']).values)
    
    # Inject noise into Train 80%
    train_noisy = inject_noise(train_clean, NOISE_RATIO)
    train_scaled = scaler.transform(train_noisy)
    
    # Windowing
    log.info(f"[{domain_name}] Building windows...")
    X_train, y_train = build_windows(train_scaled, SEQ_LEN, HORIZONS)
    X_test, y_test = build_windows(test_vals, SEQ_LEN, HORIZONS)
    
    # Save Tensors
    train_path = os.path.join(PROCESSED_DIR, f"{domain_name}_train.pt")
    test_path = os.path.join(PROCESSED_DIR, f"{domain_name}_test.pt")
    
    torch.save((torch.tensor(X_train), torch.tensor(y_train)), train_path)
    torch.save((torch.tensor(X_test), torch.tensor(y_test)), test_path)
    
    log.info(f"[{domain_name}] Saved train: X={X_train.shape}, Test: X={X_test.shape}")

def main():
    domains = ["cloud_azure", "web_nasa", "micro_rs_anomic", "micro_train_ticket"]
    for d in domains:
        process_domain(d)
        
    log.info("ALL DOMAINS PROCESSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
