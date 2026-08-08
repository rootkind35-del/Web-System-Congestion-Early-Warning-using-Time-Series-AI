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

def build_features(train_input_path: str, test_input_path: str, output_dir: str):
    print(f"Loading raw Train/Val data from: {train_input_path}")
    df_train_raw = pd.read_csv(train_input_path)
    
    print(f"Loading raw Test data from: {test_input_path}")
    df_test_raw = pd.read_csv(test_input_path)
    
    numeric_cols = ['cpu_usage', 'ram_usage', 'req_rate', 'latency_ms']
    
    print("Applying Causal Savitzky-Golay Filter separately on Train/Val and Test sets...")
    df_train_smoothed = df_train_raw.copy()
    df_train_smoothed[numeric_cols] = apply_sg_filter(df_train_raw[numeric_cols].values, window_length=15, polyorder=3)
    
    df_test_smoothed = df_test_raw.copy()
    df_test_smoothed[numeric_cols] = apply_sg_filter(df_test_raw[numeric_cols].values, window_length=15, polyorder=3)
    
    # 1. Split July (Train/Val) dataset: 80% Train, 20% Val
    total_train_len = len(df_train_smoothed)
    train_end = int(total_train_len * 0.8)
    
    print(f"Train/Val split: Train indices = [0, {train_end}], Val indices = [{train_end}, {total_train_len}]")
    print(f"Test split: 100% August 1995 dataset = [0, {len(df_test_smoothed)}]")
    
    # 2. Fit MinMaxScaler strictly on the Train split (July 80%) to avoid any data leakage
    print("Fitting MinMaxScaler strictly on July Train set...")
    scaler = MinMaxScaler()
    scaler.fit(df_train_smoothed.iloc[:train_end][numeric_cols])
    
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "minmax_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # 3. Transform datasets using the trained Train-set scaler
    train_val_scaled = scaler.transform(df_train_smoothed[numeric_cols])
    test_scaled = scaler.transform(df_test_smoothed[numeric_cols])
    
    # 4. Build sliding windows
    window_size = 30
    horizons = [5, 10, 15]
    
    # Train set (July first 80%)
    train_features = train_val_scaled[:train_end]
    train_target = train_features[:, 0] # CPU is column 0
    X_train, y_train = create_sliding_windows(train_features, train_target, window_size, horizons)
    
    # Validation set (July last 20%, overlapping window_size to avoid boundary loss)
    val_start_idx = train_end - window_size
    val_features = train_val_scaled[val_start_idx:]
    val_target = val_features[:, 0]
    X_val, y_val = create_sliding_windows(val_features, val_target, window_size, horizons)
    
    # Test set (August 100%, fresh sliding windows starting at 0)
    test_features = test_scaled
    test_target = test_features[:, 0]
    X_test, y_test = create_sliding_windows(test_features, test_target, window_size, horizons)
    
    out_file = os.path.join(output_dir, "processed_dataset")
    print(f"Compressing and saving Tensors to disk...")
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
    
    default_train_input = os.path.join(project_root, "data", "raw", "multi_year_web_metrics.csv")
    default_test_input = os.path.join(project_root, "data", "raw", "test_web_metrics.csv")
    default_output = os.path.join(project_root, "data", "processed")
    
    parser.add_argument('--train_input', type=str, default=default_train_input)
    parser.add_argument('--test_input', type=str, default=default_test_input)
    parser.add_argument('--output_dir', type=str, default=default_output)
    
    args = parser.parse_args()
    build_features(args.train_input, args.test_input, args.output_dir)
