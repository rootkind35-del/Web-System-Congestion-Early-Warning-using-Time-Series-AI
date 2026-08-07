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

def create_sliding_windows(data_df: pd.DataFrame, window_size: int = 30, horizons: list = [5, 10, 15]):
    print(f"Creating sliding windows: Window size={window_size}, Horizons={horizons}")
    
    features = data_df[['cpu_usage', 'ram_usage', 'req_rate', 'latency_ms']].values
    target = data_df['cpu_usage'].values
    
    max_horizon = max(horizons)
    num_samples = len(data_df) - window_size - max_horizon + 1
    
    X = np.zeros((num_samples, window_size, 4))
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
    
    print("Applying Savitzky-Golay (SG Filter) to remove noise...")
    df_smoothed = df.copy()
    df_smoothed[numeric_cols] = apply_sg_filter(df[numeric_cols].values, window_length=15, polyorder=3)
    
    print("Scaling data...")
    scaler = MinMaxScaler()
    df_scaled = df_smoothed.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_smoothed[numeric_cols])
    
    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "minmax_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    X, y = create_sliding_windows(df_scaled, window_size=30, horizons=[5, 10, 15])
    
    total_samples = len(X)
    train_size = int(total_samples * 0.7)
    val_size = int(total_samples * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    out_file = os.path.join(output_dir, "processed_dataset.npz")
    print(f"Compressing and saving Tensor to disk (this might take a while)...")
    np.savez_compressed(
        out_file, 
        X_train=X_train.astype(np.float32), y_train=y_train.astype(np.float32),
        X_val=X_val.astype(np.float32), y_val=y_val.astype(np.float32),
        X_test=X_test.astype(np.float32), y_test=y_test.astype(np.float32)
    )
    
    print(f"Done processing! X_train shape: {X_train.shape}")
    print(f"Processed dataset saved to: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and create features.")
    
    default_input = os.path.join(project_root, "data", "raw", "multi_year_web_metrics.csv")
    default_output = os.path.join(project_root, "data", "processed")
    
    parser.add_argument('--input', type=str, default=default_input)
    parser.add_argument('--output_dir', type=str, default=default_output)
    
    args = parser.parse_args()
    build_features(args.input, args.output_dir)
