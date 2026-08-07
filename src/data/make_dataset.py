import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta

# Import ECommerceEventInjector
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)
from src.data.event_injector import ECommerceEventInjector

def generate_multi_year_dataset(output_path: str, years: int = 3, interval_minutes: int = 1):
    days = years * 365
    num_samples = int((days * 24 * 60) / interval_minutes)
    start_time = datetime(2021, 1, 1, 0, 0, 0)
    
    print("Starting Data Fusion Pipeline...")
    print("Generating", num_samples, "samples for", years, "years...")
    
    start_np = np.datetime64('2021-01-01T00:00')
    np_timestamps = start_np + np.arange(num_samples).astype('timedelta64[m]')
    timestamps = pd.DatetimeIndex(np_timestamps)
    
    hours = timestamps.hour.values
    dayofweek = timestamps.dayofweek.values
    months = timestamps.month.values
    
    trend = np.linspace(1000, 2500, num_samples)
    daily_seasonality = 1.0 + 0.6 * np.sin((hours - 8) * (2 * np.pi / 24))
    weekend_penalty = np.where(dayofweek >= 5, 0.8, 1.0)
    monthly_seasonality = 1.0 + 0.1 * np.sin((months - 5) * (2 * np.pi / 12))
    
    base_req_rate = trend * daily_seasonality * weekend_penalty * monthly_seasonality
    
    noise = np.random.poisson(lam=100, size=num_samples) - 100
    base_req_rate += noise
    
    injector = ECommerceEventInjector(timestamps, base_req_rate)
    req_rate = injector.inject_mega_sales(target_months=[11, 12], target_days=[11, 12], multiplier=8.0)
    req_rate = injector.inject_payday_sales(multiplier=3.0)
    req_rate = np.clip(req_rate, 50, None)
    
    max_req_capacity = 8000.0
    cpu_usage = (req_rate / max_req_capacity) * 100.0
    cpu_usage += np.random.normal(0, 3, num_samples)
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    ram_usage = 30 + (cpu_usage * 0.4) + np.random.normal(0, 2, num_samples)
    ram_usage = np.clip(ram_usage, 0, 100)
    
    base_latency = 45.0
    queue_factor = np.exp(np.clip((cpu_usage - 80) / 5, 0, 10))
    latency = base_latency + 5 * queue_factor + np.random.lognormal(mean=1.0, sigma=0.5, size=num_samples)
    latency = np.clip(latency, 20, 15000)
    
    print("Packing and saving dataset...")
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cpu_usage': np.round(cpu_usage, 2),
        'ram_usage': np.round(ram_usage, 2),
        'req_rate': np.round(req_rate, 2),
        'latency_ms': np.round(latency, 2)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, chunksize=100000)
    
    print("Done! Saved at:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Multi-year dataset (Wiki+NASA+Shopee).")
    parser.add_argument('--years', type=int, default=3, help='Years.')
    parser.add_argument('--interval', type=int, default=1, help='Interval in mins.')
    
    args = parser.parse_args()
    out_path = os.path.join(project_root, "data", "raw", "multi_year_web_metrics.csv")
    
    generate_multi_year_dataset(output_path=out_path, years=args.years, interval_minutes=args.interval)
