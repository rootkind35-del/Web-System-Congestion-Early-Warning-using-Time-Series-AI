import numpy as np
import pandas as pd
import os
import argparse
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

def download_nasa_logs(target_path: str, url: str):
    if not os.path.exists(target_path):
        print(f"Downloading NASA access log from {url}...")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        urllib.request.urlretrieve(url, target_path)
        print("Download complete.")
    else:
        print(f"NASA access log at {target_path} already exists locally.")

def parse_nasa_logs(gzip_path: str):
    print(f"Parsing NASA access logs from {gzip_path}...")
    pattern = re.compile(r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})')
    counts = collections.Counter()
    
    with gzip.open(gzip_path, 'rt', encoding='latin1') as f:
        for i, line in enumerate(f):
            match = pattern.search(line)
            if match:
                ts_str = match.group(1)
                # Drop the seconds to group by minute (e.g. 01/Jul/1995:00:00)
                minute_str = ts_str[:-3]
                counts[minute_str] += 1
            if i > 0 and i % 500000 == 0:
                print(f"Parsed {i} log entries...")
                
    return counts

def generate_dataset_for_month(month_name: str, output_path: str, inject_events: bool):
    url = f"http://ita.ee.lbl.gov/traces/NASA_access_log_{month_name}.gz"
    gzip_path = os.path.join(project_root, "data", "raw", f"NASA_access_log_{month_name}.gz")
    
    # Download logs
    download_nasa_logs(gzip_path, url)
    
    # Parse logs to get raw request rate per minute
    counts = parse_nasa_logs(gzip_path)
    
    # Generate continuous time range
    if month_name == "Jul95":
        start_time = datetime(1995, 7, 1, 0, 0)
        end_time = datetime(1995, 7, 31, 23, 59)
    else:
        start_time = datetime(1995, 8, 1, 0, 0)
        end_time = datetime(1995, 8, 31, 23, 59)
        
    delta = timedelta(minutes=1)
    timestamps = []
    curr = start_time
    while curr <= end_time:
        timestamps.append(curr)
        curr += delta
        
    print(f"Generated {len(timestamps)} continuous minutes for {month_name}.")
    
    # Map raw requests per minute
    base_req_rate = []
    for ts in timestamps:
        # Format matching log: 01/Jul/1995:00:00
        key = ts.strftime("%d/%b/%Y:%H:%M")
        base_req_rate.append(counts.get(key, 0))
        
    base_req_rate = np.array(base_req_rate, dtype=float)
    pd_timestamps = pd.DatetimeIndex(timestamps)
    
    # Scale up to modern enterprise load (peak ~1000 req/min)
    base_req_rate *= 12.0
    
    if inject_events:
        # Inject Sales anomalies
        injector = ECommerceEventInjector(pd_timestamps, base_req_rate)
        req_rate = injector.inject_mega_sales(target_months=[7], target_days=[11, 12], multiplier=8.0)
        req_rate = injector.inject_payday_sales(multiplier=3.0)
        req_rate = np.clip(req_rate, 10, None)
    else:
        # Pure real traffic, scaled up, no sales injection
        req_rate = np.clip(base_req_rate, 10, None)
        
    # Compute system telemetry based on queuing dynamics
    num_samples = len(req_rate)
    max_req_capacity = 6000.0  # Server capacity limit
    
    # CPU usage dynamically scales with request rate
    cpu_usage = (req_rate / max_req_capacity) * 100.0
    cpu_usage += np.random.normal(0, 2, num_samples) # system noise
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    # RAM usage scales with CPU load (dynamic memory allocation)
    ram_usage = 30 + (cpu_usage * 0.45) + np.random.normal(0, 1.5, num_samples)
    ram_usage = np.clip(ram_usage, 0, 100)
    
    # Response Latency modeled using M/M/1 queuing behavior (exponential growth near capacity)
    base_latency = 45.0
    queue_factor = np.exp(np.clip((cpu_usage - 80) / 4, -5, 10))
    latency = base_latency + 6 * queue_factor + np.random.lognormal(mean=1.2, sigma=0.4, size=num_samples)
    latency = np.clip(latency, 20, 15000)
    
    print("Saving processed telemetry dataset...")
    df = pd.DataFrame({
        'timestamp': pd_timestamps,
        'cpu_usage': np.round(cpu_usage, 2),
        'ram_usage': np.round(ram_usage, 2),
        'req_rate': np.round(req_rate, 2),
        'latency_ms': np.round(latency, 2)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Done! Saved dataset at:", output_path)

if __name__ == "__main__":
    train_path = os.path.join(project_root, "data", "raw", "multi_year_web_metrics.csv")
    test_path = os.path.join(project_root, "data", "raw", "test_web_metrics.csv")
    
    print("=== Generating Train/Val dataset (July 1995 with 20% mixed events) ===")
    generate_dataset_for_month("Jul95", train_path, inject_events=True)
    
    print("=== Generating Test dataset (August 1995, 100% real NASA traffic) ===")
    generate_dataset_for_month("Aug95", test_path, inject_events=False)
