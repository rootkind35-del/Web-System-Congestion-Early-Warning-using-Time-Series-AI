"""
make_dataset_bigdata.py
=======================
Tiền xử lý toàn bộ ~79 GB raw data thành PyTorch tensors sẵn sàng training.

Pipeline:
    1. Đọc chunk-by-chunk từng dataset → aggregate theo phút → pd.Series cpu_pct
    2. Ghép chronologically → unified time series
    3. Split Train(80%) / Val(10%) / Test(10%) theo thứ tự thời gian
    4. MinMaxScaler fit chỉ trên Train → transform Val + Test
    5. Sliding window seq_len=30, horizon=[1,5,15] → save *.pt tensors

Quy tắc (từ handoff):
    - KHÔNG dùng ổ C: (project trên F:)
    - KHÔNG dùng Calgary-HTTP data
    - Train/Val/Test tách biệt trước khi scale
    - MinMaxScaler chỉ fit trên Train

NOTE: Dùng gzip.open() thủ công thay vì pd.read_csv(file.gz, chunksize=...)
      để tránh segfault CPython 3.14 + pandas C-extension trên Windows.
"""

import os
import sys
import gc
import gzip
import json
import tarfile
import pickle
import time
import logging
import re
import collections
from io import StringIO
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_DIR   = RAW_DIR / "google"
AZURE_DIR    = RAW_DIR / "azure"
AMAZON_FILE  = RAW_DIR / "amazon_books_reviews.jsonl"
ALIBABA_FILE = RAW_DIR / "alibaba_container_usage.tar.gz"
NASA_GZ      = RAW_DIR / "NASA_access_log_Aug95.gz"

# ── Hyperparams ────────────────────────────────────────────────────────────────
SEQ_LEN     = 30             # 30-minute lookback window
HORIZONS    = [1, 5, 15]    # predict t+1, t+5, t+15 minutes
BATCH_LINES = 10_000        # lines per batch (safe for Python 3.14 memory)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10           # remaining 10% → test

# ── Pre-computed epoch offsets (Unix seconds) ──────────────────────────────────
GOOGLE_EPOCH_UNIX  = pd.Timestamp("2011-01-01", tz="UTC").timestamp()  # 1293840000.0
AZURE_EPOCH_UNIX   = pd.Timestamp("2017-03-01", tz="UTC").timestamp()  # 1488326400.0
ALIBABA_EPOCH_UNIX = pd.Timestamp("2018-01-01", tz="UTC").timestamp()  # 1514764800.0

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(PROCESSED_DIR / "make_dataset_bigdata.log"), mode="w"),
    ]
)
log = logging.getLogger(__name__)


def _ts_to_minute_series(ts_unix_sec: np.ndarray, values: np.ndarray) -> pd.Series:
    """
    Vectorized: unix-second timestamps + values → per-minute mean pd.Series.
    Uses direct pd.to_datetime(numpy_array, unit='s') — stable on Python 3.14.
    """
    dt = pd.to_datetime(ts_unix_sec, unit="s", utc=True)
    s  = pd.Series(values, index=dt, dtype="float64", name="cpu_pct")
    s.index = s.index.floor("min")
    return s.groupby(s.index).mean()


def _read_gz_chunks(fpath: Path, batch_size: int = BATCH_LINES):
    """
    Generator: open gzip manually, yield batches of raw text lines.
    Avoids pandas C-extension gzip decompression that segfaults in CPython 3.14.
    """
    with gzip.open(str(fpath), "rt", encoding="utf-8", errors="replace") as gz:
        buf = []
        for line in gz:
            buf.append(line)
            if len(buf) >= batch_size:
                yield "".join(buf)
                buf = []
        if buf:
            yield "".join(buf)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GOOGLE CLUSTER DATA 2011 — task_usage
# Schema (no header, 20 cols):
#   col[4]  = end_time (microseconds since cluster epoch = 2011-01-01 UTC)
#   col[5]  = mean CPU rate (fraction 0–1)
# ═══════════════════════════════════════════════════════════════════════════════
def process_google() -> pd.Series:
    log.info("=== Processing Google Cluster Data 2011 ===")
    files = sorted(GOOGLE_DIR.glob("part-*.csv.gz"))
    log.info(f"Found {len(files)} files")

    all_minute_series = []
    total_rows = 0

    for fpath in files:
        log.info(f"  {fpath.name}")
        try:
            for text_batch in _read_gz_chunks(fpath, BATCH_LINES):
                try:
                    chunk = pd.read_csv(
                        StringIO(text_batch), header=None,
                        usecols=[4, 5], on_bad_lines="skip"
                    )
                    chunk.columns = ["ts_us", "cpu_frac"]
                    chunk["ts_us"]    = pd.to_numeric(chunk["ts_us"],    errors="coerce")
                    chunk["cpu_frac"] = pd.to_numeric(chunk["cpu_frac"], errors="coerce")
                    chunk.dropna(inplace=True)
                    chunk = chunk[(chunk["cpu_frac"] >= 0) & (chunk["cpu_frac"] <= 1.0)]
                    if chunk.empty:
                        continue
                    ts  = chunk["ts_us"].to_numpy("float64") / 1e6 + GOOGLE_EPOCH_UNIX
                    cpu = chunk["cpu_frac"].to_numpy("float64") * 100.0
                    all_minute_series.append(_ts_to_minute_series(ts, cpu))
                    total_rows += len(chunk)
                except Exception as e:
                    log.debug(f"    batch error: {e}")
                    continue
        except Exception as e:
            log.warning(f"  Error reading {fpath.name}: {e} — skipping")
            continue

    log.info(f"  Google total: {total_rows:,} rows")
    if not all_minute_series:
        log.warning("  Google: no data!")
        return pd.Series(dtype="float64", name="cpu_pct")

    combined = pd.concat(all_minute_series).groupby(level=0).mean().resample("1min").mean()
    log.info(f"  Google: {len(combined):,} minutes | {combined.index[0]} → {combined.index[-1]}")
    log.info(f"  cpu%: min={combined.min():.2f} max={combined.max():.2f} mean={combined.mean():.2f}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 2. AZURE VM TRACES 2017 — vm_cpu_readings
# Schema (no header, 5 cols):
#   col[0] = timestamp (seconds relative to trace start ≈ 2017-03-01)
#   col[4] = avg_cpu (%)
# ═══════════════════════════════════════════════════════════════════════════════
def process_azure() -> pd.Series:
    log.info("=== Processing Azure VM Traces 2017 ===")
    files = sorted(AZURE_DIR.glob("*.csv.gz"))
    log.info(f"Found {len(files)} files")

    all_minute_series = []
    total_rows = 0

    for fpath in files:
        log.info(f"  {fpath.name}")
        try:
            for text_batch in _read_gz_chunks(fpath, BATCH_LINES):
                try:
                    chunk = pd.read_csv(
                        StringIO(text_batch), header=None,
                        usecols=[0, 4], on_bad_lines="skip"
                    )
                    chunk.columns = ["ts_sec", "avg_cpu"]
                    chunk["ts_sec"]  = pd.to_numeric(chunk["ts_sec"],  errors="coerce")
                    chunk["avg_cpu"] = pd.to_numeric(chunk["avg_cpu"], errors="coerce")
                    chunk.dropna(inplace=True)
                    chunk = chunk[(chunk["avg_cpu"] >= 0) & (chunk["avg_cpu"] <= 100.0)]
                    if chunk.empty:
                        continue
                    ts  = chunk["ts_sec"].to_numpy("float64") + AZURE_EPOCH_UNIX
                    cpu = chunk["avg_cpu"].to_numpy("float64")
                    all_minute_series.append(_ts_to_minute_series(ts, cpu))
                    total_rows += len(chunk)
                except Exception as e:
                    log.debug(f"    batch error: {e}")
                    continue
        except Exception as e:
            log.warning(f"  Error reading {fpath.name}: {e} — skipping")
            continue

    log.info(f"  Azure total: {total_rows:,} rows")
    if not all_minute_series:
        log.warning("  Azure: no data!")
        return pd.Series(dtype="float64", name="cpu_pct")

    combined = pd.concat(all_minute_series).groupby(level=0).mean().resample("1min").mean()
    log.info(f"  Azure: {len(combined):,} minutes | {combined.index[0]} → {combined.index[-1]}")
    log.info(f"  cpu%: min={combined.min():.2f} max={combined.max():.2f} mean={combined.mean():.2f}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AMAZON REVIEWS 2023 — Books (proxy traffic load)
# Schema: JSONL — timestamp(ms), rating(float), helpful_vote(int)
# ═══════════════════════════════════════════════════════════════════════════════
def process_amazon() -> pd.Series:
    log.info("=== Processing Amazon Books Reviews 2023 ===")
    if not AMAZON_FILE.exists():
        log.warning(f"  Not found: {AMAZON_FILE}")
        return pd.Series(dtype="float64", name="cpu_pct")

    all_minute_series = []
    buf_ts, buf_val = [], []
    FLUSH_EVERY = 200_000
    total_rows  = 0

    def flush():
        if not buf_ts:
            return
        ts  = np.array(buf_ts, dtype="float64")
        val = np.array(buf_val, dtype="float64")
        all_minute_series.append(_ts_to_minute_series(ts, val))

    log.info(f"  Streaming {AMAZON_FILE.name} ...")
    with open(AMAZON_FILE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj    = json.loads(line)
                ts_ms  = obj.get("timestamp")
                rating = obj.get("rating")
                hvote  = obj.get("helpful_vote", 0)
                if ts_ms is None or rating is None:
                    continue
                ts_unix = float(ts_ms) / 1000.0
                rating  = float(rating)
                hvote   = max(0, int(hvote))
                proxy   = float(np.clip(((rating - 1.0) / 4.0) * (1.0 + np.log1p(hvote)) * 20.0, 0.0, 100.0))
                buf_ts.append(ts_unix)
                buf_val.append(proxy)
                total_rows += 1

                if len(buf_ts) >= FLUSH_EVERY:
                    flush()
                    buf_ts.clear()
                    buf_val.clear()
                    if total_rows % 5_000_000 == 0:
                        log.info(f"  Amazon: {total_rows:,} rows ...")
            except Exception:
                continue

    flush()
    log.info(f"  Amazon total: {total_rows:,} rows")
    if not all_minute_series:
        log.warning("  Amazon: no data!")
        return pd.Series(dtype="float64", name="cpu_pct")

    combined = pd.concat(all_minute_series).groupby(level=0).mean().resample("1min").mean()
    log.info(f"  Amazon: {len(combined):,} minutes | {combined.index[0]} → {combined.index[-1]}")
    log.info(f"  traffic proxy: min={combined.min():.2f} max={combined.max():.2f} mean={combined.mean():.2f}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ALIBABA CONTAINER USAGE 2018
# Schema: single CSV inside tar.gz, no header
#   col[2] = timestamp (seconds relative to trace start ≈ 2018-01-01)
#   col[5] = cpu_util (fraction or %)
# ═══════════════════════════════════════════════════════════════════════════════
def process_alibaba() -> pd.Series:
    log.info("=== Processing Alibaba Container Usage 2018 ===")
    if not ALIBABA_FILE.exists():
        log.warning(f"  Not found: {ALIBABA_FILE}")
        return pd.Series(dtype="float64", name="cpu_pct")

    all_minute_series = []
    total_rows = 0
    buf = []
    BATCH = 200_000

    def flush_batch(lines):
        nonlocal total_rows
        if not lines:
            return
        try:
            chunk = pd.read_csv(
                StringIO("\n".join(lines)), header=None,
                usecols=[2, 5], on_bad_lines="skip"
            )
            chunk.columns = ["ts_sec", "cpu_val"]
            chunk["ts_sec"]  = pd.to_numeric(chunk["ts_sec"],  errors="coerce")
            chunk["cpu_val"] = pd.to_numeric(chunk["cpu_val"], errors="coerce")
            chunk.dropna(inplace=True)
            if chunk["cpu_val"].max() <= 1.5:
                chunk["cpu_val"] *= 100.0
            chunk = chunk[(chunk["cpu_val"] >= 0) & (chunk["cpu_val"] <= 100.0)]
            if chunk.empty:
                return
            ts  = chunk["ts_sec"].to_numpy("float64") + ALIBABA_EPOCH_UNIX
            cpu = chunk["cpu_val"].to_numpy("float64")
            all_minute_series.append(_ts_to_minute_series(ts, cpu))
            total_rows += len(chunk)
        except Exception as ex:
            log.debug(f"  Alibaba batch error: {ex}")

    try:
        tf = tarfile.open(str(ALIBABA_FILE), mode="r|gz")
        for member in tf:
            if not member.name.endswith(".csv"):
                continue
            log.info(f"  Member: {member.name}")
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            try:
                for raw_line in fobj:
                    try:
                        buf.append(raw_line.decode("utf-8", errors="replace").rstrip())
                    except Exception:
                        continue
                    if len(buf) >= BATCH:
                        flush_batch(buf)
                        buf.clear()
                        if total_rows % 5_000_000 == 0 and total_rows > 0:
                            log.info(f"  Alibaba: {total_rows:,} rows ...")
            except EOFError:
                log.warning("  Alibaba: EOFError (truncated gzip) — using data so far")
            flush_batch(buf)
            buf.clear()
            break
    except Exception as e:
        log.warning(f"  Alibaba: {e}")

    log.info(f"  Alibaba total: {total_rows:,} rows")
    if not all_minute_series:
        log.warning("  Alibaba: no data!")
        return pd.Series(dtype="float64", name="cpu_pct")

    combined = pd.concat(all_minute_series).groupby(level=0).mean().resample("1min").mean()
    log.info(f"  Alibaba: {len(combined):,} minutes | {combined.index[0]} → {combined.index[-1]}")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 5. NASA HTTP LOGS (test seed / ground truth)
# ═══════════════════════════════════════════════════════════════════════════════
def process_nasa() -> pd.Series:
    log.info("=== Processing NASA HTTP Log Aug 1995 ===")
    if not NASA_GZ.exists():
        log.warning(f"  Not found: {NASA_GZ}")
        return pd.Series(dtype="float64", name="cpu_pct")

    pattern = re.compile(r'\[(\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2})')
    counts  = collections.Counter()

    with gzip.open(str(NASA_GZ), "rt", encoding="latin1") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                counts[m.group(1)[:-3]] += 1

    start = datetime(1995, 8, 1, 0, 0)
    end   = datetime(1995, 8, 31, 23, 59)
    curr  = start
    timestamps, req_rates = [], []
    while curr <= end:
        timestamps.append(curr)
        req_rates.append(counts.get(curr.strftime("%d/%b/%Y:%H:%M"), 0))
        curr += timedelta(minutes=1)

    req_arr = np.array(req_rates, dtype="float64") * 12.0
    cpu_pct = np.clip(req_arr / 6000.0 * 100.0, 0.0, 100.0)
    idx    = pd.DatetimeIndex(timestamps).tz_localize("UTC")
    series = pd.Series(cpu_pct, index=idx, name="cpu_pct")
    log.info(f"  NASA: {len(series):,} minutes")
    return series


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BUILD SLIDING WINDOWS
# ═══════════════════════════════════════════════════════════════════════════════
def build_windows(values: np.ndarray, seq_len: int, horizons: list):
    """X: (N, seq_len, 1)  y: (N, len(horizons))"""
    max_h = max(horizons)
    N     = len(values) - seq_len - max_h + 1
    if N <= 0:
        raise ValueError(f"Not enough data: len={len(values)}, seq_len={seq_len}, max_h={max_h}")

    X = np.lib.stride_tricks.sliding_window_view(
        values[: N + seq_len - 1 + max_h], seq_len
    )[:N, :, np.newaxis].astype("float32")

    y = np.zeros((N, len(horizons)), dtype="float32")
    for j, h in enumerate(horizons):
        y[:, j] = values[seq_len - 1 + h: seq_len - 1 + h + N].astype("float32")

    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    log.info("╔══════════════════════════════════════════════════════╗")
    log.info("║  make_dataset_bigdata.py — Big Data Preprocessing    ║")
    log.info("╚══════════════════════════════════════════════════════╝")
    log.info(f"  Project root : {PROJECT_ROOT}")
    log.info(f"  Output dir   : {PROCESSED_DIR}")
    log.info(f"  SEQ_LEN={SEQ_LEN}  HORIZONS={HORIZONS}  BATCH={BATCH_LINES:,}")

    # ── Step 1: Process datasets ───────────────────────────────────────────────
    series_list = []

    s_google = process_google()
    if not s_google.empty:
        series_list.append(s_google)

    s_azure = process_azure()
    if not s_azure.empty:
        series_list.append(s_azure)

    s_amazon = process_amazon()
    if not s_amazon.empty:
        series_list.append(s_amazon)

    s_alibaba = process_alibaba()
    if not s_alibaba.empty:
        series_list.append(s_alibaba)

    s_nasa = process_nasa()  # held-out test seed

    if not series_list:
        log.error("No data from any dataset!")
        sys.exit(1)

    # ── Step 2: Merge ──────────────────────────────────────────────────────────
    log.info("=== Merging all datasets ===")
    combined = pd.concat(series_list).sort_index()
    combined = combined.resample("1min").mean()
    combined = combined.ffill(limit=60).bfill(limit=60).dropna()
    log.info(f"  Combined: {len(combined):,} minutes")
    log.info(f"  Range: {combined.index[0]} → {combined.index[-1]}")
    log.info(f"  Stats: min={combined.min():.3f} max={combined.max():.3f} mean={combined.mean():.3f}")

    # ── Step 3: Chronological split ────────────────────────────────────────────
    n       = len(combined)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_s = combined.iloc[:n_train]
    val_s   = combined.iloc[n_train: n_train + n_val]
    test_s  = combined.iloc[n_train + n_val:]

    log.info(f"  Train: {len(train_s):,}  Val: {len(val_s):,}  Test: {len(test_s):,}")

    # ── Step 4: Scale (fit on Train only) ──────────────────────────────────────
    log.info("=== Fitting MinMaxScaler (Train only) ===")
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_s.values.reshape(-1, 1)).flatten()
    val_scaled   = scaler.transform(val_s.values.reshape(-1, 1)).flatten()
    test_scaled  = scaler.transform(
        np.clip(test_s.values, scaler.data_min_[0], scaler.data_max_[0]).reshape(-1, 1)
    ).flatten()

    scaler_path = PROCESSED_DIR / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"  Scaler: data_min={scaler.data_min_[0]:.4f} data_max={scaler.data_max_[0]:.4f}")
    log.info(f"  Saved: {scaler_path}")

    if not s_nasa.empty:
        nasa_scaled = scaler.transform(
            np.clip(s_nasa.values, scaler.data_min_[0], scaler.data_max_[0]).reshape(-1, 1)
        ).flatten()

    # ── Step 5: Sliding windows + save ────────────────────────────────────────
    log.info("=== Building sliding windows ===")

    def save_split(vals, name):
        try:
            X, y = build_windows(vals, SEQ_LEN, HORIZONS)
            log.info(f"  {name}: X={X.shape}  y={y.shape}")
            out = PROCESSED_DIR / f"{name}.pt"
            torch.save((torch.tensor(X), torch.tensor(y)), str(out))
            log.info(f"  Saved: {out.name} ({out.stat().st_size/1e6:.1f} MB)")
        except ValueError as e:
            log.warning(f"  {name}: skipped — {e}")

    save_split(train_scaled, "train")
    save_split(val_scaled,   "val")
    save_split(test_scaled,  "test")
    if not s_nasa.empty:
        save_split(nasa_scaled, "test_nasa")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info(f"✅  DONE in {elapsed/60:.1f} minutes ({elapsed:.0f}s)")
    log.info(f"   Output: {PROCESSED_DIR}")
    for p in sorted(PROCESSED_DIR.glob("*.pt")):
        log.info(f"   {p.name}  ({p.stat().st_size/1e6:.1f} MB)")
    log.info("   scaler.pkl")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
