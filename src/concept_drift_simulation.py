import os
import sys
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
import pandas as pd
from sklearn.metrics import mean_absolute_error
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from src.models.tcn_multitask import TCNDualAttBiLSTM_MultiTask
from src.train_all_massive_azure import AzureStreamingDataset

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "azure_parquet_3GB")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lab", "best_TCNDualAttBiLSTM_MultiTask_120E.pth")

class PageHinkley:
    def __init__(self, min_instances=30, delta=0.005, threshold=0.1, alpha=0.9999):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.n = 0
        self.sum = 0.0
        
    def add_element(self, x):
        self.n += 1
        self.x_mean = self.x_mean + (x - self.x_mean) / self.n
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)
        
        if self.n > self.min_instances and self.sum > self.threshold:
            return True # Drift detected
        return False
        
    def reset(self):
        self.x_mean = 0.0
        self.n = 0
        self.sum = 0.0

def online_retrain(model, X_batch, y_r_batch, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _, _, p_r = model(X_batch)
    loss = criterion(p_r, y_r_batch)
    loss.backward()
    optimizer.step()
    model.eval()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        log.error(f"Base model {MODEL_PATH} not found. Train TCNDualAttBiLSTM first!")
        return
        
    model = TCNDualAttBiLSTM_MultiTask(input_dim=7).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) # Smaller LR for online
    criterion = nn.MSELoss()
    
    test_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "test_*.parquet")))
    test_ds = AzureStreamingDataset(test_files, is_train=False, max_files=1)
    # Batch size 128 for sequential online stream processing
    test_loader = DataLoader(test_ds, batch_size=128, num_workers=0) 
    
    ph = PageHinkley(threshold=0.15)
    
    mae_history = []
    drift_points = []
    retrain_events = []
    
    log.info("Starting Streaming Evaluation with Concept Drift Injection...")
    
    for i, (X, y_b, y_t, y_r) in enumerate(test_loader):
        X, y_r = X.to(device), y_r.to(device)
        
        # INJECT DRIFT: At batch 100, simulate a massive cloud flash crowd (Shift CPU & RAM up)
        if i >= 100 and i < 200:
            X[:, :, 1] = X[:, :, 1] + 20.0 # Shift CPU up
            X[:, :, 2] = X[:, :, 2] + 10.0 # Shift RAM up
            y_r = torch.clamp(y_r + 0.3, 0.0, 1.0) # Risk goes up drastically
            
        with torch.no_grad():
            _, _, p_r = model(X)
            
        # Calculate batch MAE
        batch_mae = mean_absolute_error(y_r[:, 0].cpu().numpy(), p_r[:, 0].cpu().numpy())
        mae_history.append(batch_mae)
        
        # Page-Hinkley detection
        if ph.add_element(batch_mae):
            log.warning(f"CONCEPT DRIFT DETECTED at batch {i}! MAE spiked to {batch_mae:.4f}")
            drift_points.append(i)
            
            # TRIGGER ONLINE RETRAINING
            log.info("Triggering Online Retraining for 5 batches...")
            for _ in range(5):
                online_retrain(model, X, y_r, optimizer, criterion)
                
            retrain_events.append(i)
            ph.reset() # Reset detector after retraining
            
        if i % 50 == 0:
            log.info(f"Batch {i} | MAE: {batch_mae:.4f}")
            
        if i >= 300: # End simulation
            break
            
    # Save simulation results for visualization
    df = pd.DataFrame({'batch': range(len(mae_history)), 'mae': mae_history})
    df.to_csv(os.path.join(PROJECT_ROOT, "data", "reports", "drift_simulation.csv"), index=False)
    log.info("Simulation Complete. Drift and Retrain data saved.")

if __name__ == "__main__":
    main()
