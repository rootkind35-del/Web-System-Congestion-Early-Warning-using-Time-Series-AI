import pandas as pd
import numpy as np
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
import joblib

# Import Savitzky-Golay filter
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)
from src.features.sg_filter import apply_sg_filter

def create_sliding_windows(features: np.ndarray, target: np.ndarray, window_size: int = 30, horizons: list = [5, 10, 15]):
    max_horizon = max(horizons)
    num_samples = len(features) - window_size - max_horizon + 1
    
    if num_samples <= 0:
        return np.empty((0, window_size, features.shape[1])), np.empty((0, len(horizons)))
        
    X = np.zeros((num_samples, window_size, features.shape[1]))
    y = np.zeros((num_samples, len(horizons)))
    
    for i in range(num_samples):
        X[i] = features[i : i + window_size]
        current_time_idx = i + window_size - 1
        for j, h in enumerate(horizons):
            y[i, j] = target[current_time_idx + h]
            
    return X, y

def build_features(input_path: str, output_dir: str):
    print(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)
    
    numeric_cols = ['cpu_usage', 'ram_usage', 'req_rate', 'latency_ms']
    
    print("Applying Causal Savitzky-Golay Filter to remove noise without temporal leakage...")
    df_smoothed = df.copy()
    df_smoothed[numeric_cols] = apply_sg_filter(df[numeric_cols].values, window_length=15, polyorder=3)
    
    # 1. Xác định chỉ số phân tách trước khi scale (Chronological Split)
    total_len = len(df_smoothed)
    train_end = int(total_len * 0.7)
    val_end = train_end + int(total_len * 0.15)
    
    print(f"Dataset split indices: Train = [0, {train_end}], Val = [{train_end}, {val_end}], Test = [{val_end}, {total_len}]")
    
    # 2. Fit MinMaxScaler CHỈ trên tập huấn luyện (Train Set) để chống rò rỉ thông tin phân phối
    print("Fitting MinMaxScaler only on Train set...")
    scaler = MinMaxScaler()
    scaler.fit(df_smoothed.iloc[:train_end][numeric_cols])
    
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "minmax_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # 3. Transform toàn bộ tập dữ liệu bằng Scaler đã học trên Train set
    df_scaled_values = scaler.transform(df_smoothed[numeric_cols])
    
    # 4. Trích xuất mảng để tạo sliding window cho từng tập độc lập
    window_size = 30
    horizons = [5, 10, 15]
    max_h = max(horizons)
    
    # Tập Train
    train_features = df_scaled_values[:train_end]
    train_target = train_features[:, 0] # CPU is at column 0 (scaled)
    X_train, y_train = create_sliding_windows(train_features, train_target, window_size, horizons)
    
    # Tập Val (bao gồm phần gối đầu window_size từ cuối tập Train để không mất dữ liệu biên)
    val_start_idx = train_end - window_size
    val_features = df_scaled_values[val_start_idx:val_end]
    val_target = val_features[:, 0]
    X_val, y_val = create_sliding_windows(val_features, val_target, window_size, horizons)
    
    # Tập Test (bao gồm phần gối đầu window_size từ cuối tập Val)
    test_start_idx = val_end - window_size
    test_features = df_scaled_values[test_start_idx:]
    test_target = test_features[:, 0]
    X_test, y_test = create_sliding_windows(test_features, test_target, window_size, horizons)
    
    out_file = os.path.join(output_dir, "processed_dataset")
    print(f"Compressing and saving Tensor to disk...")
    np.savez_compressed(
        out_file, 
        X_train=X_train.astype(np.float32), y_train=y_train.astype(np.float32),
        X_val=X_val.astype(np.float32), y_val=y_val.astype(np.float32),
        X_test=X_test.astype(np.float32), y_test=y_test.astype(np.float32)
    )
    
    print(f"Done processing! Shapes:")
    print(f" - Train: {X_train.shape}, {y_train.shape}")
    print(f" - Val  : {X_val.shape}, {y_val.shape}")
    print(f" - Test : {X_test.shape}, {y_test.shape}")
    print(f"Processed dataset saved to: {out_file}.npz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and create features.")
    
    default_input = os.path.join(project_root, "data", "raw", "multi_year_web_metrics.csv")
    default_output = os.path.join(project_root, "data", "processed")
    
    parser.add_argument('--input', type=str, default=default_input)
    parser.add_argument('--output_dir', type=str, default=default_output)
    
    args = parser.parse_args()
    build_features(args.input, args.output_dir)
