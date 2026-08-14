"""
pipeline_azure.py
==================
Phase 5: Multivariate Spatial Aggregation.
Trích xuất 5 đặc trưng từ Azure (mean, std, max, min, overload_count).
Dữ liệu sẽ được gói thành Tensor [Samples, 30, 5].
Target Y vẫn là 3 Horizons của 'mean_cpu' (CPU trung bình toàn hệ thống).
"""

import os
import sys
import gzip
import json
import time
import pickle
import logging
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "azure"
MINUTE_DIR = PROJECT_ROOT / "data" / "minute"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

# ── Hyperparams ────────────────────────────────────────────────────────────────
BATCH_LINES = 100_000   
SEQ_LEN = 30            
HORIZONS = [5, 10, 15]  
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
OVERLOAD_THRESHOLD = 85.0 # Ngưỡng để đếm số lượng máy tính bị quá tải cục bộ

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(REPORTS_DIR / "pipeline_azure_multivariate.log"), mode="w", encoding="utf-8"),
    ]
)
log = logging.getLogger(__name__)


def read_gz_chunks(fpath: Path, batch_size: int):
    with gzip.open(str(fpath), "rt", encoding="utf-8", errors="replace") as gz:
        buf = []
        for line in gz:
            buf.append(line)
            if len(buf) >= batch_size:
                yield "".join(buf)
                buf = []
        if buf:
            yield "".join(buf)


def stage_1_raw_to_minute_multivariate():
    """Đọc dữ liệu thô Azure, trích xuất 5 dimensions cho mỗi phút."""
    log.info("=== STAGE 1: Raw to Multivariate Minute (Azure) ===")
    files = sorted(RAW_DIR.glob("*.csv.gz"))
    log.info(f"Tìm thấy {len(files)} files CSV.gz")

    if not files:
        log.error("Không tìm thấy dữ liệu Azure!")
        return False

    all_minute_data = []
    total_rows = 0

    for i, fpath in enumerate(files):
        log.info(f"Đang xử lý [{i+1}/{len(files)}]: {fpath.name}")
        for text_batch in read_gz_chunks(fpath, BATCH_LINES):
            try:
                # col[0]=ts, col[2]=min_cpu, col[3]=max_cpu, col[4]=avg_cpu
                chunk = pd.read_csv(
                    StringIO(text_batch), header=None,
                    usecols=[0, 2, 3, 4], on_bad_lines="skip"
                )
                chunk.columns = ["ts_sec", "min_cpu", "max_cpu", "avg_cpu"]
                chunk = chunk.apply(pd.to_numeric, errors="coerce")
                chunk.dropna(inplace=True)
                
                # Filter valid ranges
                valid = (chunk["avg_cpu"] >= 0) & (chunk["avg_cpu"] <= 100.0)
                chunk = chunk[valid]

                if chunk.empty:
                    continue

                # Elapsed minute
                chunk["elapsed_minute"] = (chunk["ts_sec"] // 60).astype("int64")
                
                # Flag overloaded VMs
                chunk["is_overload"] = (chunk["avg_cpu"] > OVERLOAD_THRESHOLD).astype(int)

                # Groupby minute to extract 5 features
                grouped = chunk.groupby("elapsed_minute").agg(
                    mean_cpu=("avg_cpu", "mean"),
                    std_cpu=("avg_cpu", "std"),
                    max_spike=("max_cpu", "max"),
                    min_cpu=("min_cpu", "min"),
                    overload_count=("is_overload", "sum")
                ).reset_index()
                
                all_minute_data.append(grouped)
                total_rows += len(chunk)
            except Exception as e:
                log.warning(f"Lỗi khi xử lý batch trong file {fpath.name}: {e}")

    log.info(f"Đã đọc tổng cộng {total_rows:,} rows từ raw data Azure.")
    
    if not all_minute_data:
        log.error("Không có dữ liệu hợp lệ sau khi parse!")
        return False

    log.info("Gộp và thống kê toàn bộ dataset theo phút (Multivariate)...")
    combined = pd.concat(all_minute_data, ignore_index=True)
    
    # Do có thể 1 phút bị chia thành nhiều batch, ta group lại lần nữa
    final_df = combined.groupby("elapsed_minute").agg(
        mean_cpu=("mean_cpu", "mean"),       # Xấp xỉ (chính xác nhất là weighted mean nhưng mean cũng đủ tốt)
        std_cpu=("std_cpu", "mean"),         # Xấp xỉ std
        max_spike=("max_spike", "max"),      # Chính xác tuyệt đối
        min_cpu=("min_cpu", "min"),          # Chính xác tuyệt đối
        overload_count=("overload_count", "sum") # Chính xác tuyệt đối
    ).sort_index()

    # Fill NaN std_cpu (nếu phút đó chỉ có 1 VM) bằng 0
    final_df["std_cpu"] = final_df["std_cpu"].fillna(0.0)

    # Reindex and Interpolate
    log.info("Reindex và Linear Interpolation...")
    min_idx = final_df.index.min()
    max_idx = final_df.index.max()
    final_df = final_df.reindex(range(min_idx, max_idx + 1))
    final_df = final_df.interpolate(method='linear', limit=4)
    final_df.dropna(inplace=True)

    out_file = MINUTE_DIR / "azure_multivariate.parquet"
    final_df.to_parquet(out_file)
    log.info(f"Đã lưu ma trận đa biến ({len(final_df):,} dòng x 5 cột) vào {out_file.name}")
    return True


def build_windows_gap_aware_mv(minutes_array: np.ndarray, values: np.ndarray, seq_len: int, horizons: list):
    """
    Tạo sliding windows cho Multivariate.
    values shape: [N, 5]. 
    X_list shape: [Samples, seq_len, 5]
    y_list shape: [Samples, len(horizons)] (chỉ dự báo mean_cpu tức là cột 0).
    """
    max_h = max(horizons)
    total_len = seq_len + max_h
    N = len(values) - total_len + 1
    
    if N <= 0:
        raise ValueError("Dữ liệu quá ngắn để tạo window.")

    X_list, y_list = [], []
    
    for i in range(N):
        time_slice = minutes_array[i : i + total_len]
        if time_slice[-1] - time_slice[0] == total_len - 1:
            x_seq = values[i : i + seq_len, :]  # Toàn bộ 5 features
            # Chỉ lấy cột 0 (mean_cpu) làm Target Y
            y_seq = [values[i + seq_len - 1 + h, 0] for h in horizons]
            X_list.append(x_seq)
            y_list.append(y_seq)
            
    X = np.array(X_list, dtype="float32")
    y = np.array(y_list, dtype="float32")
    return X, y


def stage_3_4_5_split_scale_window():
    """Thực hiện chia split (Azure), fit/transform scaler đa biến và tạo window."""
    log.info("=== STAGE 3, 4, 5: Split, Scale & Windowing (Multivariate Azure) ===")
    in_file = MINUTE_DIR / "azure_multivariate.parquet"
    df = pd.read_parquet(in_file)
    
    N = len(df)
    n_train = int(N * TRAIN_RATIO)
    n_val = int(N * VAL_RATIO)
    
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val :]
    
    log.info(f"Split size: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    
    # ── Stage 4: Scaling (5 Features) ──
    log.info("Fitting MinMaxScaler (5D) trên tập Train...")
    scaler = MinMaxScaler()
    
    train_vals = train_df.values  # Shape: [N_train, 5]
    train_scaled = scaler.fit_transform(train_vals)
    train_mins = train_df.index.values
    
    val_vals = val_df.values
    val_scaled = scaler.transform(val_vals)
    val_mins = val_df.index.values
    
    test_vals = test_df.values.copy()
    # Clip test values
    for i in range(5):
        test_vals[:, i] = np.clip(test_vals[:, i], scaler.data_min_[i], scaler.data_max_[i])
    test_scaled = scaler.transform(test_vals)
    test_mins = test_df.index.values
    
    scaler_path = PROCESSED_DIR / "azure_mv_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Đã lưu MV scaler: {scaler_path.name}")
    
    # ── Stage 5: Windowing ──
    log.info("Đang tạo Sliding Windows Multivariate...")
    def save_split(mins, vals, name):
        try:
            X, y = build_windows_gap_aware_mv(mins, vals, SEQ_LEN, HORIZONS)
            log.info(f"  azure_mv_{name}: X={X.shape}, y={y.shape}")
            out_path = PROCESSED_DIR / f"azure_mv_{name}.pt"
            torch.save((torch.tensor(X), torch.tensor(y)), str(out_path))
            log.info(f"  Saved -> {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
        except ValueError as e:
            log.warning(f"  azure_mv_{name}: Bỏ qua - {e}")
            
    save_split(train_mins, train_scaled, "train")
    save_split(val_mins, val_scaled, "val")
    save_split(test_mins, test_scaled, "test")
    
    return scaler


def main():
    t0 = time.time()
    log.info("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ ĐA BIẾN (MULTIVARIATE) AZURE")
    
    # Stage 1 finished successfully and saved azure_multivariate.parquet
    # We only need to run the split, scale, windowing stage now.
    stage_3_4_5_split_scale_window()
            
    elapsed = time.time() - t0
    log.info(f"HOÀN THÀNH PIPELINE TỔNG THỜI GIAN: {elapsed/60:.2f} phút")

if __name__ == "__main__":
    main()
