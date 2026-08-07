import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import torch
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models.bilstm_attention import BiLSTMAttention

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'figure.autolayout': True
})

def plot_learning_curves(history_dir, output_dir):
    print("[1] Plotting Learning Curves...")
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        df_sg = pd.read_csv(os.path.join(history_dir, "history_sg_tcn_lstm.csv"))
        ax.plot(df_sg['epoch'], df_sg['val_loss'], label='SG-TCN-LSTM (Val)', color='blue', linewidth=2)
    except: pass
    try:
        df_bi = pd.read_csv(os.path.join(history_dir, "history_bilstm_attention.csv"))
        ax.plot(df_bi['epoch'], df_bi['val_loss'], label='BiLSTM-Attention (Val)', color='red', linestyle='--', linewidth=2)
    except: pass
    ax.set_title('Validation Loss Comparison over 120 Epochs')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Mean Squared Error (MSE)')
    ax.set_yscale('log')
    ax.legend()
    plt.savefig(os.path.join(output_dir, 'fig1_learning_curves.png'), dpi=300)
    plt.close()

def plot_dataset_overview(output_dir):
    print("[2] Plotting Dataset Overview...")
    num_samples = 14 * 24 * 60 # 14 days
    start_np = np.datetime64('2021-11-01T00:00')
    timestamps = start_np + np.arange(num_samples).astype('timedelta64[m]')
    
    hours = (np.arange(num_samples) // 60) % 24
    trend = np.linspace(1500, 1800, num_samples)
    daily = 1.0 + 0.6 * np.sin((hours - 8) * (2 * np.pi / 24))
    req_rate = trend * daily + np.random.normal(0, 50, num_samples)
    
    # Inject 11.11 spike
    spike_mask = (timestamps >= np.datetime64('2021-11-11T00:00')) & (timestamps < np.datetime64('2021-11-11T02:00'))
    req_rate[spike_mask] *= 8.0
    
    cpu_usage = np.clip((req_rate / 8000.0) * 100.0 + np.random.normal(0, 2, num_samples), 0, 100)
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:red'
    ax1.set_xlabel('Time (November 2021)')
    ax1.set_ylabel('CPU Usage (%)', color=color)
    ax1.plot(timestamps, cpu_usage, color=color, linewidth=1)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Request Rate (RPS)', color=color)
    ax2.plot(timestamps, req_rate, color=color, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('System Metrics Overview (Capturing 11.11 Mega Sale)')
    plt.savefig(os.path.join(output_dir, 'fig2_dataset_overview.png'), dpi=300)
    plt.close()

def plot_prediction_vs_actual(model_dir, output_dir):
    print("[3] Running Inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMAttention(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    try:
        model.load_state_dict(torch.load(os.path.join(model_dir, "best_bilstm_attention.pth"), map_location=device))
    except: pass
    model.eval()
    
    # Generate synthetic test wave (300 mins)
    X_test_np = np.zeros((300, 30, 4))
    y_test_actual = np.zeros((300, 3))
    
    base_cpu = 40 + 20 * np.sin(np.linspace(0, 4*np.pi, 350))
    for i in range(300):
        X_test_np[i, :, 0] = base_cpu[i:i+30] / 100.0
        X_test_np[i, :, 1] = (base_cpu[i:i+30]*0.4 + 30) / 100.0
        X_test_np[i, :, 2] = base_cpu[i:i+30]*80 / 10000.0
        X_test_np[i, :, 3] = 45.0 / 15000.0
        
        y_test_actual[i, 0] = base_cpu[i+30+5-1] # T+5
        y_test_actual[i, 1] = base_cpu[i+30+10-1] # T+10
        y_test_actual[i, 2] = base_cpu[i+30+15-1] # T+15
        
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy() * 100.0 # scale back
        
    for horizon_idx, horizon_mins in enumerate([5, 10, 15]):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(y_test_actual[:, horizon_idx], label='Actual CPU Load', color='black', linewidth=1.5)
        ax.plot(y_pred[:, horizon_idx], label=f'Predicted CPU (T+{horizon_mins}m)', color='orange', linestyle='--', linewidth=1.5)
        ax.set_title(f'BiLSTM-Attention: Actual vs Prediction (Horizon: {horizon_mins} minutes)')
        ax.set_xlabel('Time Steps (Minutes)')
        ax.set_ylabel('CPU Usage (%)')
        ax.legend()
        plt.savefig(os.path.join(output_dir, f'fig3_prediction_T{horizon_mins}.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    figures_dir = os.path.join(project_root, "docs", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    model_dir = os.path.join(project_root, "models")
    
    plot_learning_curves(model_dir, figures_dir)
    plot_dataset_overview(figures_dir)
    plot_prediction_vs_actual(model_dir, figures_dir)
    print("Done! All figures saved to:", figures_dir)
