import torch
import numpy as np
import joblib
import os
import sys
import argparse
import time

# Sửa lỗi ModuleNotFoundError
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.models.sg_tcn_lstm import MultiHorizonTCNLSTM
from src.models.bilstm_attention import BiLSTMAttention
from src.utils.thresholding import DynamicThresholdEMA

def load_inference_env(model_name: str, model_path: str, scaler_path: str):
    """
    Tải môi trường suy luận (Model, Scaler).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Khởi động suy luận trên {device} với mô hình {model_name}...")
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    
    # Khởi tạo mô hình & Load Weights
    if model_name == "sg_tcn_lstm":
        model = MultiHorizonTCNLSTM(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3)
    elif model_name == "bilstm_attention":
        model = BiLSTMAttention(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3)
    else:
        raise ValueError("Tên mô hình không hợp lệ.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, scaler, device

def predict_realtime(model, scaler, device, recent_data_window: np.ndarray):
    """
    Thực hiện dự báo (Inference) thời gian thực bằng FP16 AMP.
    """
    scaled_data = scaler.transform(recent_data_window)
    input_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    start_time = time.perf_counter()
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            predictions = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    inference_time = (time.perf_counter() - start_time) * 1000 # tính bằng ms
    
    pred_cpu_scaled = predictions.cpu().numpy()[0]
    dummy_arr = np.zeros((len(pred_cpu_scaled), 4))
    dummy_arr[:, 0] = pred_cpu_scaled # CPU ở cột index 0
    pred_cpu_actual = scaler.inverse_transform(dummy_arr)[:, 0]
    
    return pred_cpu_actual, inference_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy suy luận thời gian thực với mô hình FP16.")
    
    parser.add_argument('--model', type=str, required=True, choices=['sg_tcn_lstm', 'bilstm_attention'])
    args = parser.parse_args()
    
    model_file = os.path.join(project_root, "models", f"best_{args.model}.pth")
    scaler_file = os.path.join(project_root, "data", "processed", "minmax_scaler.pkl")
    
    # Mock data
    np.random.seed(99)
    mock_realtime_data = np.array([
        [50 + np.random.normal(0, 5), 40 + np.random.normal(0, 2), 1000 + np.random.normal(0, 100), 60 + np.random.normal(0, 10)]
        for _ in range(30)
    ])
    
    try:
        model, scaler, device = load_inference_env(args.model, model_file, scaler_file)
        
        print("\n[Predicting]...")
        predictions, inf_time = predict_realtime(model, scaler, device, mock_realtime_data)
        
        print(f"\n✅ Dự báo tải CPU trong 3 mốc (5, 10, 15 phút) tới:")
        print(f" -> T+5 phút : {predictions[0]:.2f} %")
        print(f" -> T+10 phút: {predictions[1]:.2f} %")
        print(f" -> T+15 phút: {predictions[2]:.2f} %")
        
        print(f"⚡ Thời gian Inference (FP16): {inf_time:.3f} ms")
        
        print("\n[Threshold Check]")
        detector = DynamicThresholdEMA(slo_latency=2000.0)
        for row in mock_realtime_data:
             detector.update_and_detect(current_cpu=row[0], current_latency=row[3])
             
        current_latency = mock_realtime_data[-1, 3]
        for horizon, pred in zip([5, 10, 15], predictions):
             res = detector.update_and_detect(pred, current_latency)
             print(f"Báo động Mốc T+{horizon}m: {'CÓ NGHẼN (WARNING)' if res['is_cpu_high'] else 'BÌNH THƯỜNG'} (Dự báo CPU: {pred:.1f}% vs Ngưỡng động: {res['dynamic_cpu_threshold']:.1f}%)")
             
    except Exception as e:
        print(f"Lỗi: {e}. Vui lòng đảm bảo đã chạy src/train.py --model {args.model} để sinh file trọng số trước.")
