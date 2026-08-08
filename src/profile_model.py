import os
import sys
import time
import torch
import psutil
import numpy as np
import joblib

import numpy as np
import joblib
import pandas as pd

# Add project root to PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.tcn_dualatt_bilstm import TCNDualAttBiLSTM
from src.features.sg_filter import apply_sg_filter
from src.utils.thresholding import DynamicThresholdEMA
from src.utils.drift_detection import PageHinkleyDriftDetector

def profile_model():
    print("=================================================================")
    print("              RIGOROUS RESOURCE AND LATENCY PROFILE              ")
    print("=================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Platform: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"Execution Device: {device}")
    
    # Load scaling parameters for benchmarking
    scaler_path = os.path.join(project_root, "data", "processed", "minmax_scaler.pkl")
    scaler = joblib.load(scaler_path)
    
    # 1. Initialize proposed TCN-DualAtt-BiLSTM Model
    model = TCNDualAttBiLSTM(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    model.eval()
    
    # Load trained weights to measure actual model size on disk
    model_weight_path = os.path.join(project_root, "models", "best_tcn_dualatt_bilstm.pth")
    if os.path.exists(model_weight_path):
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        file_size_kb = os.path.getsize(model_weight_path) / 1024.0
    else:
        file_size_kb = 0.0
        print("[Warning] Trained model weights not found, using uninitialized weights.")
    
    # 2. Parameter count and model size calculation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Memory size calculation
    # FP32 model: total_params * 4 bytes
    fp32_size_kb = (total_params * 4) / 1024.0
    # FP16 model: total_params * 2 bytes
    fp16_size_kb = (total_params * 2) / 1024.0
    
    print("\n--- Model Complexity & Disk Footprint ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Calculated FP32 Size: {fp32_size_kb:.2f} KB")
    print(f"Calculated FP16 Size: {fp16_size_kb:.2f} KB")
    print(f"Serialized File Size (.pth): {file_size_kb:.2f} KB")
    
    # 3. Micro-benchmark input simulation
    # Simulate a single input window W = 30 timesteps, F = 4 features
    raw_telemetry_window = np.random.randn(30, 4)
    
    # Warm-up phase
    # PyTorch CUDA uses lazy initialization; warm up to obtain clean benchmark statistics
    print("\nWarming up CUDA/CPU kernels (100 iterations)...")
    dummy_input_tensor = torch.randn(1, 30, 4).to(device)
    for _ in range(100):
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(dummy_input_tensor)
            else:
                _ = model(dummy_input_tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # 4. Latency isolation benchmark (1000 iterations)
    iterations = 1000
    pre_latencies = []
    inf_latencies = []
    post_latencies = []
    
    # Dynamic threshold and PH drift modules
    threshold_detector = DynamicThresholdEMA(slo_latency=100.0)
    drift_detector = PageHinkleyDriftDetector()
    
    print(f"Running isolated benchmarks over {iterations} iterations...")
    
    for _ in range(iterations):
        # --- Preprocessing Step (Causal Filtering + MinMax Scaling) ---
        t_start = time.perf_counter()
        # Scale and smooth
        df_temp = pd.DataFrame(raw_telemetry_window, columns=['cpu_usage', 'ram_usage', 'req_rate', 'latency_ms'])
        scaled_window = scaler.transform(df_temp)
        # Apply causal filter on the window
        filtered_window = apply_sg_filter(scaled_window, window_length=15, polyorder=3)
        t_pre = (time.perf_counter() - t_start) * 1000 # ms
        pre_latencies.append(t_pre)
        
        # Prepare tensor
        input_tensor = torch.tensor(filtered_window, dtype=torch.float32).unsqueeze(0).to(device)
        
        # --- Inference Step (Model Prediction using FP16 Autocast) ---
        t_start = time.perf_counter()
        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    predictions = model(input_tensor)
                torch.cuda.synchronize()
            else:
                predictions = model(input_tensor)
        t_inf = (time.perf_counter() - t_start) * 1000 # ms
        inf_latencies.append(t_inf)
        
        pred_cpu_scaled = predictions.cpu().numpy()[0]
        # Re-scale target CPU load
        dummy_arr = np.zeros((3, 4))
        dummy_arr[:, 0] = pred_cpu_scaled
        pred_cpu_actual = scaler.inverse_transform(dummy_arr)[:, 0]
        
        # --- Post-processing Step (EMA Thresholding + Page-Hinkley Drift) ---
        t_start = time.perf_counter()
        current_latency = raw_telemetry_window[-1, 3]
        for p in pred_cpu_actual:
            # Update dynamic thresholding
            _ = threshold_detector.update_and_detect(p, current_latency)
            # Concept drift monitor
            _ = drift_detector.update(abs(p - 30.0)) # mock error residual
        t_post = (time.perf_counter() - t_start) * 1000 # ms
        post_latencies.append(t_post)
        
    # Analyze latencies
    def analyze_lat(lat_list):
        return {
            "mean": np.mean(lat_list),
            "median": np.median(lat_list),
            "p95": np.percentile(lat_list, 95),
            "p99": np.percentile(lat_list, 99)
        }
        
    pre_stats = analyze_lat(pre_latencies)
    inf_stats = analyze_lat(inf_latencies)
    post_stats = analyze_lat(post_latencies)
    
    total_latencies = [pre_latencies[i] + inf_latencies[i] + post_latencies[i] for i in range(iterations)]
    total_stats = analyze_lat(total_latencies)
    
    print("\n--- Latency Performance (Milliseconds) ---")
    print(f"Preprocessing (Scaling + Filtering):")
    print(f" - Mean: {pre_stats['mean']:.4f} ms | Median: {pre_stats['median']:.4f} ms | P95: {pre_stats['p95']:.4f} ms | P99: {pre_stats['p99']:.4f} ms")
    print(f"Model Inference (TCN-DualAtt-BiLSTM FP16):")
    print(f" - Mean: {inf_stats['mean']:.4f} ms | Median: {inf_stats['median']:.4f} ms | P95: {inf_stats['p95']:.4f} ms | P99: {inf_stats['p99']:.4f} ms")
    print(f"Post-processing (Alerting + Concept Drift):")
    print(f" - Mean: {post_stats['mean']:.4f} ms | Median: {post_stats['median']:.4f} ms | P95: {post_stats['p95']:.4f} ms | P99: {post_stats['p99']:.4f} ms")
    print(f"End-to-End System Processing (Total):")
    print(f" - Mean: {total_stats['mean']:.4f} ms | Median: {total_stats['median']:.4f} ms | P95: {total_stats['p95']:.4f} ms | P99: {total_stats['p99']:.4f} ms")
    print(f"Throughput: {1000.0 / total_stats['mean']:.1f} pipelines/second")
    
    # 5. Process-level Memory Profile
    process = psutil.Process(os.getpid())
    rss_memory_mb = process.memory_info().rss / (1024 * 1024)
    vms_memory_mb = process.memory_info().vms / (1024 * 1024)
    
    print("\n--- Runtime Memory Consumption ---")
    print(f"System RAM consumped by process (RSS): {rss_memory_mb:.2f} MB")
    print(f"Virtual RAM allocated by process (VMS): {vms_memory_mb:.2f} MB")
    
    if device.type == 'cuda':
        vram_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        vram_reserved_mb = torch.cuda.memory_reserved(device) / (1024 * 1024)
        vram_max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
        # Estimate parameter VRAM
        param_memory_mb = (total_params * 4) / (1024 * 1024) if not next(model.parameters()).is_cuda else 0.0 # FP32 is 4 bytes
        # Under CUDA, it's already included in allocated memory
        
        print(f"GPU VRAM Allocated (Active Model + Tensors): {vram_allocated_mb:.4f} MB")
        print(f"GPU VRAM Reserved (PyTorch Cache Manager): {vram_reserved_mb:.4f} MB")
        print(f"GPU VRAM Peak Allocation during execution: {vram_max_allocated_mb:.4f} MB")
        print(f"Estimated Pure Model Parameter VRAM: {total_params * 4 / (1024*1024):.4f} MB")
        
    print("=================================================================")
    
    # Save statistics for manuscript tables
    summary_data = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "inf_mean": inf_stats['mean'],
        "inf_median": inf_stats['median'],
        "inf_p95": inf_stats['p95'],
        "total_mean": total_stats['mean'],
        "total_median": total_stats['median'],
        "total_p95": total_stats['p95'],
        "total_params": total_params,
        "pth_size_kb": file_size_kb,
        "rss_memory_mb": rss_memory_mb,
        "vram_allocated_mb": vram_allocated_mb if device.type == 'cuda' else 0.0,
        "vram_reserved_mb": vram_reserved_mb if device.type == 'cuda' else 0.0
    }
    pd.Series(summary_data).to_json(os.path.join(project_root, "models", "profile_metrics.json"), indent=4)
    print("Saved profiling stats to models/profile_metrics.json")

if __name__ == "__main__":
    profile_model()
