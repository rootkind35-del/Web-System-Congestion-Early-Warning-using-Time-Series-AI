import numpy as np
import pandas as pd
import os
import urllib.request
import gzip
import re
import collections
from datetime import datetime, timedelta

# Import ECommerceEventInjector
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)
from src.data.event_injector import ECommerceEventInjector

def download_logs(target_path: str, url: str):
    if not os.path.exists(target_path):
        print(f"Downloading logs from {url}...")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        urllib.request.urlretrieve(url, target_path)
        print("Download complete.")
    else:
        print(f"Log archive at {target_path} already exists locally.")

def parse_logs(gzip_path: str):
    print(f"Parsing access logs from {gzip_path}...")
    pattern = re.compile(r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})')
    counts = collections.Counter()
    
    with gzip.open(gzip_path, 'rt', encoding='latin1') as f:
        for i, line in enumerate(f):
            match = pattern.search(line)
            if match:
                ts_str = match.group(1)
                minute_str = ts_str[:-3]
                counts[minute_str] += 1
            if i > 0 and i % 500000 == 0:
                print(f"Parsed {i} log entries...")
                
    return counts

def compute_8d_multivariate(pd_timestamps, req_rate, max_req_capacity=6000.0):
    num_samples = len(req_rate)
    
    # 1. Request_rate (Given)
    
    # 2. CPU_usage
    # Linear initially, but bottlenecks at 100%
    cpu_usage = (req_rate / max_req_capacity) * 100.0
    cpu_usage += np.random.normal(0, 2, num_samples) # system noise
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    # 3. Memory_usage
    # Scales with CPU, but has a base usage and memory leak (accumulates slowly)
    ram_usage = 30 + (cpu_usage * 0.45) + np.random.normal(0, 1.5, num_samples)
    ram_usage = np.clip(ram_usage, 0, 100)
    
    # 4. Network_IO (Mbps)
    # Proportional to requests + background noise
    network_io = req_rate * np.random.uniform(1.2, 2.5, num_samples) + np.random.normal(5, 1, num_samples)
    network_io = np.clip(network_io, 0, None)
    
    # 5. Disk_IO (MB/s)
    # Spikes hard when Memory > 90% due to OS swapping
    disk_io = 10 + req_rate * 0.05 + np.random.normal(0, 5, num_samples)
    swap_factor = np.exp(np.clip((ram_usage - 90) / 2.0, 0, 5)) - 1
    disk_io += swap_factor * 50.0  # Massive spike if swapping
    disk_io = np.clip(disk_io, 0, None)
    
    # 6. Response_time (Latency in ms)
    # M/M/1 queueing theory. Exponential growth when CPU is near 100.
    base_latency = 45.0
    queue_factor = np.exp(np.clip((cpu_usage - 80) / 4.0, -5, 10))
    latency = base_latency + 6 * queue_factor + np.random.lognormal(mean=1.2, sigma=0.4, size=num_samples)
    latency = np.clip(latency, 20, 30000)
    
    # 7. Error_Rate_5xx (%)
    # Non-linear probability based on Response_time timeout (e.g., > 3000ms triggers errors)
    # And if CPU == 100%, errors spike.
    error_prob = np.clip((latency - 3000) / 5000.0, 0, 1) * 100.0 # Up to 100% error rate if latency > 8000ms
    cpu_err_factor = np.exp(np.clip((cpu_usage - 98) / 0.5, 0, 5)) - 1
    error_rate = error_prob + cpu_err_factor * 5.0 + np.random.normal(0, 0.1, num_samples)
    error_rate = np.clip(error_rate, 0, 100)
    
    # Assemble DataFrame exactly as requested
    df = pd.DataFrame({
        'timestamp': pd_timestamps,
        'Request_rate': np.round(req_rate, 2),
        'CPU_usage': np.round(cpu_usage, 2),
        'Memory_usage': np.round(ram_usage, 2),
        'Disk_IO': np.round(disk_io, 2),
        'Network_IO': np.round(network_io, 2),
        'Response_time': np.round(latency, 2),
        'Error_Rate_5xx': np.round(error_rate, 2)
    })
    
    return df

def generate_calgary_dataset(output_path: str):
    url = "http://ita.ee.lbl.gov/traces/calgary_access_log.gz"
    gzip_path = os.path.join(project_root, "data", "raw", "calgary_access_log.gz")
    
    download_logs(gzip_path, url)
    counts = parse_logs(gzip_path)
    
    start_time = datetime(1994, 10, 24, 0, 0)
    end_time = datetime(1995, 10, 11, 23, 59)
    delta = timedelta(minutes=1)
    
    timestamps = []
    curr = start_time
    while curr <= end_time:
        timestamps.append(curr)
        curr += delta
        
    print(f"Generated {len(timestamps)} minutes for Calgary.")
    
    base_req_rate = []
    for ts in timestamps:
        key = ts.strftime("%d/%b/%Y:%H:%M")
        base_req_rate.append(counts.get(key, 0))
        
    base_req_rate = np.array(base_req_rate, dtype=float) * 350.0
    pd_timestamps = pd.DatetimeIndex(timestamps)
    
    injector = ECommerceEventInjector(pd_timestamps, base_req_rate)
    req_rate = injector.inject_mega_sales(target_months=[11, 12], target_days=[11, 12], multiplier=8.0)
    req_rate = injector.inject_payday_sales(multiplier=3.0)
    req_rate = np.clip(req_rate, 10, None)
    
    df = compute_8d_multivariate(pd_timestamps, req_rate)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Done! Saved Calgary Train dataset at:", output_path)

def generate_nasa_test_dataset(output_path: str):
    url = "http://ita.ee.lbl.gov/traces/NASA_access_log_Aug95.gz"
    gzip_path = os.path.join(project_root, "data", "raw", "NASA_access_log_Aug95.gz")
    
    download_logs(gzip_path, url)
    counts = parse_logs(gzip_path)
    
    start_time = datetime(1995, 8, 1, 0, 0)
    end_time = datetime(1995, 8, 31, 23, 59)
    delta = timedelta(minutes=1)
    
    timestamps = []
    curr = start_time
    while curr <= end_time:
        timestamps.append(curr)
        curr += delta
        
    print(f"Generated {len(timestamps)} minutes for NASA Test.")
    
    base_req_rate = []
    for ts in timestamps:
        key = ts.strftime("%d/%b/%Y:%H:%M")
        base_req_rate.append(counts.get(key, 0))
        
    base_req_rate = np.array(base_req_rate, dtype=float) * 12.0
    req_rate = np.clip(base_req_rate, 10, None)
    
    pd_timestamps = pd.DatetimeIndex(timestamps)
    
    df = compute_8d_multivariate(pd_timestamps, req_rate)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Done! Saved NASA Test dataset at:", output_path)

if __name__ == "__main__":
    train_path = os.path.join(project_root, "data", "raw", "web_system_multivariate_train.csv")
    test_path = os.path.join(project_root, "data", "raw", "web_system_multivariate_test.csv")
    
    print("=== Generating Train/Val dataset (Calgary 8-Dimensional) ===")
    generate_calgary_dataset(train_path)
    
    print("=== Generating Test dataset (NASA 8-Dimensional) ===")
    generate_nasa_test_dataset(test_path)
