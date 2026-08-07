import os
import sys
import time
import torch
import psutil

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.bilstm_attention import BiLSTMAttention

def profile_model():
    print("=== MODEL RESOURCE USAGE REPORT ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Khởi tạo mô hình
    model = BiLSTMAttention(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    model.eval()
    
    # 2. Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n1. Model Complexity:")
    print(f" - Total Parameters: {total_params:,}")
    print(f" - Trainable Params: {trainable_params:,}")
    print(f" - Estimated Model Size (FP16): {total_params * 2 / 1024:.2f} KB")
    
    # 3. Đo tốc độ suy luận (Inference Latency)
    # Giả lập 1 cửa sổ dữ liệu (Batch=1, SeqLen=30, Features=4)
    dummy_input = torch.randn(1, 30, 4).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
        
    # Đo đạc 100 lần
    latencies = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    _ = model(dummy_input)
            else:
                _ = model(dummy_input)
            latencies.append((time.time() - start) * 1000) # ms
            
    avg_latency = sum(latencies) / len(latencies)
    p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
    
    print(f"\n2. Inference Speed (Batch=1, Seq=30):")
    print(f" - Average Latency: {avg_latency:.2f} ms")
    print(f" - P99 Latency (Max spike): {p99_latency:.2f} ms")
    print(f" - Throughput: {1000/avg_latency:.1f} predictions/second")
    
    # 4. RAM / VRAM Usage
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / (1024 * 1024)
    print(f"\n3. Memory Usage:")
    print(f" - System RAM consumed by Python Process: {ram_usage:.2f} MB")
    
    if device.type == 'cuda':
        vram_allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
        vram_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
        print(f" - GPU VRAM Allocated (Model + Data): {vram_allocated:.2f} MB")
        print(f" - GPU VRAM Reserved (PyTorch Cache): {vram_reserved:.2f} MB")

if __name__ == "__main__":
    profile_model()
