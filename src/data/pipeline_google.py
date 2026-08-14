"""
pipeline_google.py
==================
Phase 1: Xử lý dữ liệu Google Cluster 2011 theo 5 stages chuẩn khoa học.
- Bỏ qua datetime giả mạo (dùng elapsed_minute)
- Không dùng ffill() qua các khoảng trống
- Split theo thời gian cho riêng domain Google
- Univariate CPU forecasting
- Target horizons: [5, 10, 15] phút
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

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "google"
MINUTE_DIR = PROJECT_ROOT / "data" / "minute"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

for d in [MINUTE_DIR, PROCESSED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparams ────────────────────────────────────────────────────────────────
BATCH_LINES = 100_000   # An toàn cho bộ nhớ
SEQ_LEN = 30            # Lịch sử 30 phút
HORIZONS = [5, 10, 15]  # Dự báo tương lai t+5, t+10, t+15
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(REPORTS_DIR / "pipeline_google.log"), mode="w"),
    ]
)
log = logging.getLogger(__name__)


def read_gz_chunks(fpath: Path, batch_size: int):
    """Đọc gzip theo batch dòng để tránh lỗi bộ nhớ của pd.read_csv với CPython 3.14"""
    with gzip.open(str(fpath), "rt", encoding="utf-8", errors="replace") as gz:
        buf = []
        for line in gz:
            buf.append(line)
            if len(buf) >= batch_size:
                yield "".join(buf)
                buf = []
        if buf:
            yield "".join(buf)


def stage_1_raw_to_minute():
    """Đọc dữ liệu thô, chuyển thành mức phút (elapsed minute), lưu parquet."""
    log.info("=== STAGE 1: Raw to Minute (Google) ===")
    files = sorted(RAW_DIR.glob("part-*.csv.gz"))
    log.info(f"Tìm thấy {len(files)} files CSV.gz")

    if not files:
        log.error("Không tìm thấy dữ liệu Google!")
        return False

    all_minute_data = []
    total_rows = 0

    for fpath in files:
        log.info(f"Đang xử lý: {fpath.name}")
        for text_batch in read_gz_chunks(fpath, BATCH_LINES):
            try:
                # Chỉ lấy col[4]=ts_us, col[5]=cpu_frac
                chunk = pd.read_csv(
                    StringIO(text_batch), header=None,
                    usecols=[4, 5], on_bad_lines="skip"
                )
                chunk.columns = ["ts_us", "cpu_frac"]
                chunk["ts_us"] = pd.to_numeric(chunk["ts_us"], errors="coerce")
                chunk["cpu_frac"] = pd.to_numeric(chunk["cpu_frac"], errors="coerce")
                chunk.dropna(inplace=True)
                chunk = chunk[(chunk["cpu_frac"] >= 0) & (chunk["cpu_frac"] <= 1.0)]

                if chunk.empty:
                    continue

                # Tính elapsed minute từ microseconds (1 min = 60,000,000 us)
                # Dùng floor division
                chunk["elapsed_minute"] = (chunk["ts_us"] // 60_000_000).astype("int64")
                chunk["cpu_pct"] = chunk["cpu_frac"] * 100.0

                # Nhóm theo elapsed_minute và tính trung bình
                grouped = chunk.groupby("elapsed_minute")["cpu_pct"].mean().reset_index()
                all_minute_data.append(grouped)
                total_rows += len(chunk)
            except Exception as e:
                log.warning(f"Lỗi khi xử lý batch trong file {fpath.name}: {e}")

    log.info(f"Đã đọc tổng cộng {total_rows:,} rows từ raw data.")
    
    if not all_minute_data:
        log.error("Không có dữ liệu hợp lệ sau khi parse!")
        return False

    # Gom toàn bộ lại và nhóm theo elapsed_minute một lần nữa
    log.info("Gộp và thống kê toàn bộ dataset theo phút...")
    combined = pd.concat(all_minute_data, ignore_index=True)
    final_series = combined.groupby("elapsed_minute")["cpu_pct"].mean().sort_index()

    out_file = MINUTE_DIR / "google.parquet"
    final_series.to_frame().to_parquet(out_file)
    log.info(f"Đã lưu {len(final_series):,} phút vào {out_file.name}")
    return True


def stage_2_quality_control():
    """Kiểm tra chất lượng dữ liệu: missing, gap, range."""
    log.info("=== STAGE 2: Quality Control (Google) ===")
    in_file = MINUTE_DIR / "google.parquet"
    if not in_file.exists():
        log.error(f"Không tìm thấy file {in_file}")
        return False

    df = pd.read_parquet(in_file)
    
    # Calculate metrics
    total_minutes = len(df)
    min_minute = int(df.index.min())
    max_minute = int(df.index.min()) # oops, wait, int(df.index.max()) is correct
    max_minute = int(df.index.max())
    duration = max_minute - min_minute + 1
    
    # Detect gaps
    minutes_array = df.index.values
    diffs = np.diff(minutes_array)
    gaps = diffs[diffs > 1]
    total_missing_minutes = (diffs - 1).sum() if len(diffs) > 0 else 0
    missing_rate = float(total_missing_minutes / duration) * 100 if duration > 0 else 0

    cpu = df["cpu_pct"]
    invalid_range = len(cpu[(cpu < 0) | (cpu > 100)])

    report = {
        "dataset": "Google Cluster 2011",
        "rows": total_minutes,
        "min_elapsed_minute": min_minute,
        "max_elapsed_minute": max_minute,
        "total_duration_minutes": duration,
        "total_missing_minutes": int(total_missing_minutes),
        "missing_rate_pct": round(missing_rate, 2),
        "gap_count": len(gaps),
        "max_gap_size": int(gaps.max() - 1) if len(gaps) > 0 else 0,
        "invalid_range_count": invalid_range,
        "cpu_min": round(float(cpu.min()), 2),
        "cpu_max": round(float(cpu.max()), 2),
        "cpu_mean": round(float(cpu.mean()), 2)
    }

    report_file = REPORTS_DIR / "google_data_quality.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    
    log.info(f"Quality Report:\n{json.dumps(report, indent=4)}")
    
    # We do NOT use ffill() to fill large gaps. We will split on the available continuous indices.
    # To keep things simple and avoid synthetic gaps, we will just treat the dataframe as is,
    # but the sliding window generator needs to be gap-aware.
    
    return True


def build_windows_gap_aware(minutes_array: np.ndarray, values: np.ndarray, seq_len: int, horizons: list):
    """
    Tạo sliding windows nhưng KIỂM TRA GAP.
    Chỉ tạo window nếu khoảng thời gian (minutes_array) thực sự liên tục.
    Nghĩa là window cuối cùng (seq_len + max_horizon) phải có hiệu số phút = đúng độ dài.
    """
    max_h = max(horizons)
    total_len = seq_len + max_h
    N = len(values) - total_len + 1
    
    if N <= 0:
        raise ValueError("Dữ liệu quá ngắn để tạo window.")

    X_list, y_list = [], []
    
    for i in range(N):
        # Lấy slice thời gian
        time_slice = minutes_array[i : i + total_len]
        # Nếu thực sự liên tục, time_slice[-1] - time_slice[0] == total_len - 1
        if time_slice[-1] - time_slice[0] == total_len - 1:
            # Tạo feature X
            x_seq = values[i : i + seq_len]
            # Tạo target Y
            y_seq = [values[i + seq_len - 1 + h] for h in horizons]
            
            X_list.append(x_seq)
            y_list.append(y_seq)
            
    X = np.array(X_list, dtype="float32")
    # Expand dims cho X: (N, seq_len) -> (N, seq_len, 1) vì univariate
    X = np.expand_dims(X, axis=-1)
    
    y = np.array(y_list, dtype="float32")
    return X, y


def stage_3_4_5_split_scale_window():
    """Thực hiện chia split, fit/transform scaler và tạo window (Windowing first for short datasets)."""
    log.info("=== STAGE 3, 4, 5: Split, Scale & Windowing (Google) ===")
    in_file = MINUTE_DIR / "google.parquet"
    df = pd.read_parquet(in_file)
    
    N = len(df)
    n_train_raw = int(N * TRAIN_RATIO)
    
    # ── Stage 4: Scaling ──
    log.info("Fitting MinMaxScaler trên 80% đầu của raw data...")
    scaler = MinMaxScaler()
    
    # Fit on the first 80% of the data
    train_vals_for_fit = df["cpu_pct"].values[:n_train_raw].reshape(-1, 1)
    scaler.fit(train_vals_for_fit)
    
    # Transform all data
    all_vals = df["cpu_pct"].values.reshape(-1, 1)
    all_vals_clipped = np.clip(all_vals, scaler.data_min_[0], scaler.data_max_[0])
    scaled_vals = scaler.transform(all_vals_clipped).flatten()
    mins = df.index.values
    
    scaler_path = PROCESSED_DIR / "google_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"Đã lưu scaler: {scaler_path.name}")
    
    # ── Stage 5 & 3: Windowing then Splitting ──
    # Vì Google dataset chỉ có 106 phút, việc cắt time series rồi mới window sẽ làm tập Val/Test < 45 phút
    # -> Impossible to window. Nên ta window toàn bộ rồi cắt mảng X, y sau.
    log.info("Đang tạo Sliding Windows (gap-aware)...")
    try:
        X_all, y_all = build_windows_gap_aware(mins, scaled_vals, SEQ_LEN, HORIZONS)
        num_windows = len(X_all)
        log.info(f"Tạo được tổng cộng {num_windows} windows.")
        
        n_w_train = int(num_windows * TRAIN_RATIO)
        n_w_val = int(num_windows * VAL_RATIO)
        
        X_train, y_train = X_all[:n_w_train], y_all[:n_w_train]
        X_val, y_val = X_all[n_w_train : n_w_train + n_w_val], y_all[n_w_train : n_w_train + n_w_val]
        X_test, y_test = X_all[n_w_train + n_w_val :], y_all[n_w_train + n_w_val :]
        
        def save_tensor(X, y, name):
            if len(X) == 0:
                log.warning(f"  {name}: Không có dữ liệu để lưu.")
                return
            out_path = PROCESSED_DIR / f"google_{name}.pt"
            torch.save((torch.tensor(X), torch.tensor(y)), str(out_path))
            log.info(f"  Saved -> {out_path.name} (X={X.shape}, y={y.shape})")
            
        save_tensor(X_train, y_train, "train")
        save_tensor(X_val, y_val, "val")
        save_tensor(X_test, y_test, "test")
        
    except ValueError as e:
        log.error(f"Lỗi khi tạo window: {e}")
        
    return True


def main():
    t0 = time.time()
    log.info("BẮT ĐẦU PIPELINE TIỀN XỬ LÝ GOOGLE DATASET (PHASE 1)")
    
    if stage_1_raw_to_minute():
        if stage_2_quality_control():
            stage_3_4_5_split_scale_window()
            
    elapsed = time.time() - t0
    log.info(f"HOÀN THÀNH PIPELINE TỔNG THỜI GIAN: {elapsed/60:.2f} phút")

if __name__ == "__main__":
    main()
