import os
import sys
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.tcn_multitask import TCNDualAttBiLSTM_MultiTask
from src.models.multitask_baselines import StandardLSTM_MultiTask, SGTCNLSTM_MultiTask, BiLSTMAttention_MultiTask, Transformer_MultiTask

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure_parquet_3GB")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "lab")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ─── HYPERPARAMETERS ────────────────────────────────────────────────
SEQ_LEN = 30
HORIZONS = [5, 10, 15]
BATCH_SIZE = 16384
EPOCHS = 120             # Up to 120 epochs
PATIENCE = 10            # Early Stopping
LEARNING_RATE = 0.001
NOISE_PROB = 0.20
ALPHA, BETA, GAMMA = 0.3, 0.3, 0.4 

# ─── DATASET ────────────────────────────────────────────────────────
class AzureStreamingDataset(IterableDataset):
    def __init__(self, file_paths, is_train=True, max_files=None):
        super().__init__()
        self.file_paths = file_paths
        if max_files:
            self.file_paths = self.file_paths[:max_files]
        self.is_train = is_train
        self.feature_cols = ['Request_rate', 'CPU_usage', 'Memory_usage', 'Disk_IO', 'Network_IO', 'Response_time', 'Error_Rate_5xx']
        self.max_horizon = max(HORIZONS)
        self.window_size = SEQ_LEN + self.max_horizon
        self.norm_factors = np.array([6000.0, 100.0, 100.0, 500.0, 15000.0, 3000.0, 100.0], dtype=np.float32)

    def inject_noise(self, features):
        if not self.is_train or np.random.rand() > NOISE_PROB:
            return features
        noise_type = np.random.randint(3)
        noisy = features.copy()
        if noise_type == 0:
            noisy += np.random.normal(0, 0.05, noisy.shape)
        elif noise_type == 1:
            idx = np.random.randint(0, noisy.shape[0])
            noisy[idx] = noisy[idx] * np.random.uniform(2, 5)
        elif noise_type == 2:
            idx = np.random.randint(0, noisy.shape[0])
            noisy[idx] = 0.0
        return noisy

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = int(np.ceil(len(self.file_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            files = self.file_paths[worker_id * per_worker : (worker_id + 1) * per_worker]
        else:
            files = self.file_paths

        for fpath in files:
            # log.info(f"Loading {os.path.basename(fpath)}")
            df = pd.read_parquet(fpath)
            vm_ids = df['vm_id'].values
            boundaries = np.where(vm_ids[:-1] != vm_ids[1:])[0] + 1
            boundaries = np.concatenate(([0], boundaries, [len(df)]))
            
            features = (df[self.feature_cols].values / self.norm_factors).astype(np.float32)
            lbl_binary = df['label_binary'].values.astype(np.float32)
            lbl_tri = df['label_tri'].values.astype(np.int64)
            lbl_risk = df['label_risk'].values.astype(np.float32)
            
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i+1]
                length = end - start
                
                if length < self.window_size:
                    continue
                
                vm_feats = features[start:end]
                vm_bin = lbl_binary[start:end]
                vm_tri = lbl_tri[start:end]
                vm_risk = lbl_risk[start:end]
                
                for w in range(length - self.window_size + 1):
                    X = vm_feats[w : w + SEQ_LEN]
                    if self.is_train:
                        X = self.inject_noise(X)
                        
                    y_b, y_t, y_r = [], [], []
                    for h in HORIZONS:
                        target_idx = w + SEQ_LEN - 1 + h
                        y_b.append(vm_bin[target_idx])
                        y_t.append(vm_tri[target_idx])
                        y_r.append(vm_risk[target_idx])
                        
                    yield (
                        torch.tensor(X, dtype=torch.float32), 
                        torch.tensor(y_b, dtype=torch.float32), 
                        torch.tensor(y_t, dtype=torch.long), 
                        torch.tensor(y_r, dtype=torch.float32)
                    )
            del df, features, lbl_binary, lbl_tri, lbl_risk
            
# ─── TRAINING & EVALUATION ──────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_b_preds, all_b_true = [], []
    all_r_preds, all_r_true = [], []
    
    bin_pos_weight = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32).to(device)
    tri_weights = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32).to(device)
    
    criterion_b = nn.BCEWithLogitsLoss(pos_weight=bin_pos_weight)
    criterion_t = nn.CrossEntropyLoss(weight=tri_weights)
    criterion_r = nn.MSELoss()
    
    with torch.no_grad():
        for i, (X, y_b, y_t, y_r) in enumerate(loader):
            X, y_b, y_t, y_r = X.to(device), y_b.to(device), y_t.to(device), y_r.to(device)
            p_b, p_t, p_r = model(X)
            
            loss_b = criterion_b(p_b, y_b)
            loss_t = criterion_t(p_t.permute(0, 2, 1), y_t)
            loss_r = criterion_r(p_r, y_r)
            loss = ALPHA * loss_b + BETA * loss_t + GAMMA * loss_r
            total_loss += loss.item()
            
            b_pred = (torch.sigmoid(p_b[:, 0]) > 0.5).cpu().numpy()
            b_true = y_b[:, 0].cpu().numpy()
            all_b_preds.extend(b_pred)
            all_b_true.extend(b_true)
            
            all_r_preds.extend(p_r[:, 0].cpu().numpy())
            all_r_true.extend(y_r[:, 0].cpu().numpy())
            
            if i >= 100: # Val limit for speed
                break
                
    f1 = f1_score(all_b_true, all_b_preds, zero_division=0)
    mae = mean_absolute_error(all_r_true, all_r_preds)
    return total_loss / 100.0, f1, mae

def train_model(model_name, model, train_loader, test_loader, device):
    log.info(f"\n{'='*50}\nSTARTING TRAINING: {model_name}\n{'='*50}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    bin_pos_weight = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32).to(device)
    tri_weights = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float32).to(device)
    
    criterion_b = nn.BCEWithLogitsLoss(pos_weight=bin_pos_weight)
    criterion_t = nn.CrossEntropyLoss(weight=tri_weights)
    criterion_r = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    save_path = os.path.join(MODEL_SAVE_DIR, f"best_{model_name}_120E.pth")
    
    if os.path.exists(save_path):
        log.info(f"[{model_name}] Found existing weights at {save_path}. Resuming training!")
        model.load_state_dict(torch.load(save_path, map_location=device))
        
    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            batch_count = 0
            
            pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{EPOCHS}")
            for X, y_b, y_t, y_r in pbar:
                X, y_b, y_t, y_r = X.to(device), y_b.to(device), y_t.to(device), y_r.to(device)
                
                optimizer.zero_grad()
                p_b, p_t, p_r = model(X)
                
                loss_b = criterion_b(p_b, y_b)
                loss_t = criterion_t(p_t.permute(0, 2, 1), y_t)
                loss_r = criterion_r(p_r, y_r)
                
                loss = ALPHA * loss_b + BETA * loss_t + GAMMA * loss_r
                
                if torch.isnan(loss):
                    log.error(f"[{model_name}] LOSS IS NaN! Architecture might be exploding. Stopping this model.")
                    return False
                    
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                    
            val_loss, f1, mae = evaluate(model, test_loader, device)
            log.info(f"[{model_name}] Epoch {epoch+1} | Train Loss: {epoch_loss/batch_count:.4f} | Val Loss: {val_loss:.4f} | F1: {f1:.4f} | MAE: {mae:.4f}")
            
            # Early Stopping check
            if val_loss < best_loss - 0.0001:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), save_path)
                log.info(f"[{model_name}] Saved better model (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                log.info(f"[{model_name}] No improvement. Patience: {patience_counter}/{PATIENCE}")
                
            if patience_counter >= PATIENCE:
                log.info(f"[{model_name}] EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                break
                
        return True
        
    except Exception as e:
        log.error(f"[{model_name}] Crashed during training: {e}")
        traceback.print_exc()
        return False

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    train_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "train_*.parquet")))
    test_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "test_*.parquet")))
    
    if not train_files:
        log.error("No training files found!")
        return

    # Train on all 44 files, evaluate on all 11 files (FULL DATA)
    train_ds = AzureStreamingDataset(train_files, is_train=True, max_files=44)
    test_ds = AzureStreamingDataset(test_files, is_train=False, max_files=11)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)
    
    models = {
        # 'StandardLSTM_MultiTask': StandardLSTM_MultiTask(input_dim=7), # DONE
        # 'SGTCNLSTM_MultiTask': SGTCNLSTM_MultiTask(input_dim=7), # DONE
        'BiLSTMAttention_MultiTask': BiLSTMAttention_MultiTask(input_dim=7),
        'Transformer_MultiTask': Transformer_MultiTask(input_dim=7),
        'TCNDualAttBiLSTM_MultiTask': TCNDualAttBiLSTM_MultiTask(input_dim=7)
    }
    
    results = {}
    for name, model in models.items():
        success = train_model(name, model, train_loader, test_loader, device)
        results[name] = "SUCCESS" if success else "FAILED/NAN"
        
    log.info("\n" + "="*50)
    log.info("ALL MODELS COMPLETED")
    for k, v in results.items():
        log.info(f" - {k}: {v}")
    log.info("="*50)

if __name__ == "__main__":
    main()
