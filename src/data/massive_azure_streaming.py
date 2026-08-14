"""
massive_azure_streaming.py
─────────────────────────
Xử lý song song 55 files Azure .csv.gz (~37GB gốc, ~550 triệu dòng).
1. Đọc từng file.
2. Áp dụng Inverse M/M/1 tính 7 cột đặc trưng.
3. Đánh nhãn 3 chế độ (Binary, Tri-level, Risk).
4. Sort theo vm_id và timestamp để Dataset dễ dàng sinh window.
5. Chia Train/Test theo thời gian (80% file đầu là Train, 20% file cuối là Test).
6. Lưu định dạng Parquet (Snappy nén rất mạnh, đọc siêu tốc).
"""

import os
import glob
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_AZURE_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "azure")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure_parquet")
os.makedirs(PROCESSED_DIR, exist_ok=True)

def inverse_mm1(cpu):
    cpu_clip = np.clip(cpu, 1.0, 100.0)
    req_rate = (cpu_clip / 100.0) * 6000.0 + np.random.normal(0, 50, len(cpu))
    req_rate = np.clip(req_rate, 10, 10000)
    ram = 30 + (cpu_clip * 0.45) + np.random.normal(0, 1.5, len(cpu))
    ram = np.clip(ram, 0, 100)
    net = req_rate * np.random.uniform(1.2, 2.5, len(cpu))
    disk = 10 + req_rate * 0.05 + np.exp(np.clip((ram - 90)/2.0, 0, 5))
    
    latency = 45.0 + 6 * np.exp(np.clip((cpu_clip - 80)/4.0, -5, 10))
    error_rate = np.clip((latency - 3000)/5000.0, 0, 1)*100.0 + np.exp(np.clip((cpu_clip-98)/0.5, 0, 5))
    error_rate = np.clip(error_rate, 0, 100)
    
    return req_rate.astype(np.float32), ram.astype(np.float32), disk.astype(np.float32), net.astype(np.float32), latency.astype(np.float32), error_rate.astype(np.float32)

def process_file(file_info):
    file_path, file_idx, split = file_info
    out_file = os.path.join(PROCESSED_DIR, f"{split}_file_{file_idx:03d}.parquet")
    
    if os.path.exists(out_file):
        log.info(f"Skipping {out_file} - already exists.")
        return file_idx
        
    try:
        log.info(f"[{file_idx}] Reading {os.path.basename(file_path)}")
        df = pd.read_csv(file_path, header=None, names=['timestamp', 'vm_id', 'min_cpu', 'max_cpu', 'avg_cpu'])
        
        cpu = df['avg_cpu'].values
        req, ram, disk, net, lat, err = inverse_mm1(cpu)
        
        df['Request_rate'] = req
        df['Memory_usage'] = ram
        df['Disk_IO'] = disk
        df['Network_IO'] = net
        df['Response_time'] = lat
        df['Error_Rate_5xx'] = err
        
        # Binary Label
        df['label_binary'] = ((cpu > 90) | (err > 5) | (lat > 1500) | (ram > 95)).astype(np.float32)
        
        # Tri-level Label
        # 2 = CRITICAL, 1 = WARNING, 0 = NORMAL
        cond_crit = (cpu > 90) | (err > 5) | ((cpu > 80) & (ram > 90))
        cond_warn = (cpu > 80) | (err > 2) | (lat > 1000)
        df['label_tri'] = np.where(cond_crit, 2, np.where(cond_warn, 1, 0)).astype(np.float32)
        
        # Risk Score (0 -> 1)
        risk = (
            0.30 * (cpu / 100.0) +
            0.20 * (ram / 100.0) +
            0.25 * (np.clip(err, 0, 100) / 100.0) +
            0.15 * (np.clip(lat, 0, 3000) / 3000.0) +
            0.10 * (np.clip(disk, 0, 500) / 500.0)
        )
        df['label_risk'] = np.clip(risk, 0.0, 1.0).astype(np.float32)
        
        # Drop raw cpu columns except avg_cpu which is now CPU_usage
        df = df.rename(columns={'avg_cpu': 'CPU_usage'})
        df = df.drop(columns=['min_cpu', 'max_cpu'])
        
        # Sort by vm_id then timestamp for easy windowing in DataLoader
        log.info(f"[{file_idx}] Sorting by vm_id and timestamp...")
        df = df.sort_values(['vm_id', 'timestamp'])
        
        log.info(f"[{file_idx}] Saving to Parquet...")
        df.to_parquet(out_file, engine='pyarrow', compression='snappy')
        log.info(f"[{file_idx}] Done! Rows: {len(df):,}")
        return file_idx
        
    except Exception as e:
        log.error(f"[{file_idx}] Error processing file: {e}")
        return None

def main():
    files = sorted(glob.glob(os.path.join(RAW_AZURE_DIR, "*.csv.gz")))
    if not files:
        log.error("No raw Azure files found!")
        return
        
    log.info(f"Found {len(files)} files to process.")
    
    # 80/20 Time-based split
    split_idx = int(len(files) * 0.8)
    file_infos = []
    for i, f in enumerate(files):
        split = "train" if i < split_idx else "test"
        file_infos.append((f, i, split))
        
    log.info(f"Split: {split_idx} Train files, {len(files)-split_idx} Test files.")
    
    # Use max 6 workers to avoid choking RAM (each df takes ~4GB RAM)
    workers = min(os.cpu_count() or 4, 6)
    log.info(f"Starting ProcessPoolExecutor with {workers} workers...")
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for res in executor.map(process_file, file_infos):
            pass
            
    log.info("ALL FILES PROCESSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
