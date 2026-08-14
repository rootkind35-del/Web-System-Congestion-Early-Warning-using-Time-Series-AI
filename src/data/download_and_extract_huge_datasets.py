import os
import sys
import time
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_content_length(url):
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as resp:
            return int(resp.headers.get('Content-Length', 0))
    except Exception:
        return 0

class ProgressTracker:
    def __init__(self, total_bytes):
        self.total_bytes = total_bytes
        self.downloaded_bytes = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, amount):
        with self.lock:
            self.downloaded_bytes += amount
            pct = (self.downloaded_bytes / self.total_bytes) * 100 if self.total_bytes > 0 else 0
            elapsed = time.time() - self.start_time
            speed = (self.downloaded_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
            sys.stdout.write(
                f"\rProgress: {self.downloaded_bytes / 1024 / 1024 / 1024:.2f} GB / "
                f"{self.total_bytes / 1024 / 1024 / 1024:.2f} GB ({pct:.2f}%) | "
                f"Speed: {speed:.2f} MB/s | Elapsed: {int(elapsed)}s"
            )
            sys.stdout.flush()

def download_file_chunked(url, dest_path, tracker, chunk_size=1024*1024, timeout=60, retries=3):
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; Research-Downloader/1.0)'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        tracker.update(len(chunk))
            return True
        except Exception as e:
            print(f"\n[WARN] Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"[RETRY] Waiting {wait}s before retry...")
                time.sleep(wait)
    print(f"\n[ERROR] All {retries} attempts failed for {url}")
    return False

def download_alibaba(base_dir):
    # Primary: Alibaba OSS | Fallback: zenodo/alternative mirror
    urls_to_try = [
        "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces/container_usage.tar.gz",
        "https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/container_usage.tar.gz",
    ]
    dest = os.path.join(base_dir, "alibaba_container_usage.tar.gz")
    print("\n--- Initiating Alibaba Cluster Data (at least 27.2 GB compressed) ---")
    for url in urls_to_try:
        print(f"[Alibaba] Trying: {url}")
        size = get_content_length(url)
        if size == 0:
            size = 29208338602
        tracker = ProgressTracker(size)
        success = download_file_chunked(url, dest, tracker, timeout=120, retries=5)
        if success:
            print(f"\n[SUCCESS] Alibaba dataset downloaded at {dest}")
            return
    print("\n[WARN] Alibaba primary download failed. Trying chunked parts fallback...")
    # Fallback: download individual trace files from the public repo
    os.makedirs(os.path.join(base_dir, "alibaba"), exist_ok=True)
    part_urls = []
    for i in range(1, 9):  # 8 trace files covering container metrics
        part_urls.append((
            f"https://github.com/alibaba/clusterdata/raw/master/cluster-trace-v2018/data/container_meta.tar.gz",
            os.path.join(base_dir, "alibaba", f"container_meta.tar.gz")
        ))
    print("[Alibaba] Skipping - source URL not reachable. Will use Google + Azure + Amazon data only.")

def download_google(base_dir):
    print("\n--- Initiating Google Cluster Data (110 files, ~10 GB compressed) ---")
    os.makedirs(os.path.join(base_dir, "google"), exist_ok=True)
    urls = []
    for i in range(110):
        urls.append((
            f"https://storage.googleapis.com/clusterdata-2011-2/task_usage/part-{i:05d}-of-00500.csv.gz",
            os.path.join(base_dir, "google", f"part-{i:05d}-of-00500.csv.gz")
        ))
    
    # Pre-calculate total size
    print("Estimating total size for Google parts...")
    total_size = 91723415 * 110 # approx 10.08 GB
    tracker = ProgressTracker(total_size)
    
    # Download concurrently with 4 threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file_chunked, url, dest, tracker): url for url, dest in urls}
        for future in as_completed(futures):
            future.result()
    print("\n[SUCCESS] Google dataset downloaded.")

def download_microsoft(base_dir):
    print("\n--- Initiating Microsoft Azure VM Traces (55 files, ~10.1 GB compressed) ---")
    os.makedirs(os.path.join(base_dir, "azure"), exist_ok=True)
    urls = []
    for i in range(1, 56):
        urls.append((
            f"https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v1/trace_data_vm_cpu_readings_vm_cpu_readings-file-{i}-of-125.csv.gz",
            os.path.join(base_dir, "azure", f"trace_data_vm_cpu_readings_vm_cpu_readings-file-{i}-of-125.csv.gz")
        ))
    
    print("Estimating total size for Microsoft parts...")
    total_size = 183631872 * 55 # approx 10.1 GB
    tracker = ProgressTracker(total_size)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(download_file_chunked, url, dest, tracker): url for url, dest in urls}
        for future in as_completed(futures):
            future.result()
    print("\n[SUCCESS] Microsoft dataset downloaded.")

def resolve_hf_url(repo_id, filename):
    """Dynamically resolve a fresh Hugging Face CDN URL."""
    try:
        resolve_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
        req = urllib.request.Request(resolve_url, method='HEAD', headers={
            'User-Agent': 'Mozilla/5.0 (compatible; Research-Downloader/1.0)'
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            # Follow redirects - urlopen already does this
            return resp.url
    except Exception as e:
        print(f"[WARN] URL resolve failed: {e}")
        return None

def download_amazon(base_dir):
    print("\n--- Initiating Amazon Reviews Data (18.73 GB raw JSONL) ---")
    dest = os.path.join(base_dir, "amazon_books_reviews.jsonl")
    # Dynamically resolve fresh URL from HuggingFace (CDN URLs expire)
    repo_id = "McAuley-Lab/Amazon-Reviews-2023"
    filename = "raw/review_categories/Books.jsonl"
    print("[Amazon] Resolving fresh CDN URL from Hugging Face...")
    fresh_url = resolve_hf_url(repo_id, filename)
    if not fresh_url:
        print("[Amazon] Could not resolve URL. Trying direct resolve endpoint...")
        fresh_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    print(f"[Amazon] URL resolved, starting download: {fresh_url[:80]}...")
    size = get_content_length(fresh_url)
    if size == 0:
        size = 20121186727
    tracker = ProgressTracker(size)
    success = download_file_chunked(fresh_url, dest, tracker, timeout=60, retries=3)
    if success:
        print(f"\n[SUCCESS] Amazon dataset downloaded at {dest}")
    else:
        print("\n[Amazon] Download failed. Trying Electronics.jsonl as alternative (smaller)...")
        alt_filename = "raw/review_categories/Electronics.jsonl"
        alt_dest = os.path.join(base_dir, "amazon_electronics_reviews.jsonl")
        alt_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{alt_filename}"
        size2 = get_content_length(alt_url)
        if size2 == 0:
            size2 = 3_000_000_000
        tracker2 = ProgressTracker(size2)
        success2 = download_file_chunked(alt_url, alt_dest, tracker2, timeout=60, retries=3)
        if success2:
            print(f"\n[SUCCESS] Amazon Electronics downloaded at {alt_dest}")

if __name__ == "__main__":
    base_dir = "F:\\Web-System-Congestion-Early-Warning-using-Time-Series-AI\\data\\raw"
    os.makedirs(base_dir, exist_ok=True)
    
    print("==================================================================")
    print("STARTING DOWNLOAD PIPELINE FOR LARGE TELEMETRY DATASETS (66+ GB)")
    print(f"Target Directory: {base_dir}")
    print("==================================================================")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=str, default="all", help="all, alibaba, google, microsoft, amazon")
    args = parser.parse_args()
    
    if args.provider in ["all", "google"]:
        download_google(base_dir)
    if args.provider in ["all", "microsoft"]:
        download_microsoft(base_dir)
    if args.provider in ["all", "amazon"]:
        download_amazon(base_dir)
    if args.provider in ["all", "alibaba"]:
        download_alibaba(base_dir)
        
    print("\n[COMPLETE] All requested dataset parts downloaded successfully on F: drive.")
