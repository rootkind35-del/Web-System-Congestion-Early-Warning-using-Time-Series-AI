import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

from src.models.tcn_dualatt_bilstm import TCNDualAttBiLSTM

def plot_telemetry_overview():
    print("Generating telemetry overview plot...")
    df = pd.read_csv(os.path.join(project_root, "data", "raw", "multi_year_web_metrics.csv"))
    
    # Take a 3-day sample (July 10, 11, 12) using integer index slicing
    sample_df = df.iloc[12960:17280].copy()
    sample_df['timestamp'] = pd.to_datetime(sample_df['timestamp'])
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 1. Request Rate
    axes[0].plot(sample_df['timestamp'], sample_df['req_rate'], color='#1f77b4', linewidth=1.5)
    axes[0].set_ylabel('Requests / Min', fontsize=11, fontweight='bold')
    axes[0].set_title('Web System Workload Telemetry Overview (NASA July 1995 Access Trace)', fontsize=13, fontweight='bold', pad=15)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. CPU Usage
    axes[1].plot(sample_df['timestamp'], sample_df['cpu_usage'], color='#d62728', linewidth=1.5)
    axes[1].set_ylabel('CPU Usage (%)', fontsize=11, fontweight='bold')
    axes[1].set_ylim(0, 105)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. RAM Usage
    axes[2].plot(sample_df['timestamp'], sample_df['ram_usage'], color='#2ca02c', linewidth=1.5)
    axes[2].set_ylabel('RAM Usage (%)', fontsize=11, fontweight='bold')
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    # 4. Response Latency
    axes[3].plot(sample_df['timestamp'], sample_df['latency_ms'], color='#9467bd', linewidth=1.5)
    axes[3].set_ylabel('Latency (ms)', fontsize=11, fontweight='bold')
    axes[3].set_yscale('log')
    axes[3].set_xlabel('Timestamp', fontsize=11, fontweight='bold')
    axes[3].grid(True, linestyle=':', alpha=0.6)
    
    # Highlight Mega Sale event on July 11
    for ax in axes:
        ax.axvspan(pd.Timestamp('1995-07-11 00:00:00'), pd.Timestamp('1995-07-11 23:59:00'), color='yellow', alpha=0.15, label='Injected Mega Sale')
        
    # Remove duplicate legends
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.95))
    
    fig.tight_layout()
    os.makedirs(os.path.join(project_root, "docs", "figures"), exist_ok=True)
    out_path = os.path.join(project_root, "docs", "figures", "telemetry_overview.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved telemetry overview to {out_path}")

def plot_prediction_vs_groundtruth():
    print("Generating prediction vs ground truth plot...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_file = os.path.join(project_root, "data", "processed", "processed_dataset.npz")
    scaler_file = os.path.join(project_root, "data", "processed", "minmax_scaler.pkl")
    model_file = os.path.join(project_root, "models", "best_tcn_dualatt_bilstm.pth")
    
    data = np.load(data_file)
    X_test, y_test = data['X_test'], data['y_test']
    scaler = joblib.load(scaler_file)
    
    model = TCNDualAttBiLSTM(input_dim=4, hidden_dim=64, num_layers=2, output_dim=3).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        inp = torch.tensor(X_test[:300], dtype=torch.float32).to(device)
        out = model(inp).cpu().numpy()
        
    # Scale back to original values
    y_test_actual = np.zeros_like(y_test[:300])
    y_pred_actual = np.zeros_like(out)
    for h in range(3):
        dummy_true = np.zeros((300, 4))
        dummy_pred = np.zeros((300, 4))
        dummy_true[:, 0] = y_test[:300, h]
        dummy_pred[:, 0] = out[:, h]
        y_test_actual[:, h] = scaler.inverse_transform(dummy_true)[:, 0]
        y_pred_actual[:, h] = scaler.inverse_transform(dummy_pred)[:, 0]
        
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    horizons = ["T+5 (5 min)", "T+10 (10 min)", "T+15 (15 min)"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(3):
        axes[i].plot(y_test_actual[:, i], label="Ground Truth CPU (%)", color='black', linewidth=1.5)
        axes[i].plot(y_pred_actual[:, i], label=f"Predicted CPU ({horizons[i]})", color=colors[i], linestyle='--', linewidth=1.5)
        axes[i].set_ylabel('CPU Utilization (%)', fontsize=10, fontweight='bold')
        axes[i].set_title(f'Multi-Horizon Forecast Horizon {horizons[i]}', fontsize=11, fontweight='bold')
        axes[i].legend(loc='upper right')
        axes[i].set_ylim(0, 105)
        axes[i].grid(True, linestyle=':', alpha=0.6)
        
    axes[2].set_xlabel('Time Step (Minutes)', fontsize=11, fontweight='bold')
    
    fig.suptitle('Proactive Multi-Step Congestion Prediction vs. Ground Truth', fontsize=14, fontweight='bold', y=0.98)
    fig.tight_layout()
    out_path = os.path.join(project_root, "docs", "figures", "prediction_vs_groundtruth.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction vs ground truth to {out_path}")

def plot_sota_bar_chart():
    print("Generating SOTA comparison bar chart...")
    models_to_compare = {
        'tcn_dualatt_bilstm': 'TCN-DualAtt-BiLSTM (Ours)',
        'bilstm_attention': 'BiLSTM-Attention (Ablation)',
        'sg_tcn_lstm': 'SG-TCN-LSTM (Baseline)',
        'standard_lstm': 'Standard LSTM (Baseline)',
        'itransformer': 'iTransformer (SOTA 2024)',
        'dlinear': 'DLinear (SOTA 2023)'
    }
    
    models = []
    t5_list = []
    t10_list = []
    t15_list = []
    
    for model_key, model_name in models_to_compare.items():
        stats_file = os.path.join(project_root, "models", f"summary_stats_{model_key}.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            models.append(model_name)
            t5_list.append(stats.get('T+5_MAE_mean'))
            t10_list.append(stats.get('T+10_MAE_mean'))
            t15_list.append(stats.get('T+15_MAE_mean'))
            
    if not models:
        print("No statistics found. Skipping bar chart.")
        return
        
    x = np.arange(len(models))
    width = 0.25
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    rects1 = ax.bar(x - width, t5_list, width, label='T+5 MAE', color='#1f77b4')
    rects2 = ax.bar(x, t10_list, width, label='T+10 MAE', color='#ff7f0e')
    rects3 = ax.bar(x + width, t15_list, width, label='T+15 MAE', color='#2ca02c')
    
    ax.set_ylabel('Mean Absolute Error (MAE) - CPU percentage points ↓', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model Architecture', fontsize=12, fontweight='bold')
    ax.set_title('Exhaustive Architecture Benchmark (2020-2024) on NASA Workload Trace', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height is not None:
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    out_path = os.path.join(project_root, "docs", "figures", "sota_architecture_benchmark.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved SOTA comparison bar chart to {out_path}")

def plot_alert_ablation_vis():
    print("Generating alert ablation threshold visualization...")
    ablation_file = os.path.join(project_root, "models", "threshold_ablation_results.csv")
    if not os.path.exists(ablation_file):
        print("No threshold ablation results CSV found. Skipping.")
        return
        
    df_thresh = pd.read_csv(ablation_file)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Bar chart comparing False Alarms
    configs = df_thresh['Config'].tolist()
    false_alarms = df_thresh['False_Alarms'].tolist()
    
    rects = ax1.bar(configs, false_alarms, color=['#d62728', '#ff7f0e', '#bcbd22', '#1f77b4'])
    ax1.set_title('False Alarm Count Comparison (Noise Suppression)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Unnecessary False Alerts (Count) ↓', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Alerting Threshold Configuration', fontsize=11, fontweight='bold')
    ax1.set_xticklabels(configs, rotation=15, ha='right')
    
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(f'{int(height):,}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    # 2. Precision-Recall-F1 comparison
    x = np.arange(len(configs))
    width = 0.25
    
    p_list = df_thresh['Precision'].tolist()
    r_list = df_thresh['Recall'].tolist()
    f1_list = df_thresh['F1-Score'].tolist()
    
    rects1 = ax2.bar(x - width, p_list, width, label='Precision', color='#1f77b4')
    rects2 = ax2.bar(x, r_list, width, label='Recall', color='#aec7e8')
    rects3 = ax2.bar(x + width, f1_list, width, label='F1-Score', color='#ff7f0e')
    
    ax2.set_title('Operational Alert Efficiency Metrics (T+5 Horizon)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Metric Value [0 - 1.0] ↑', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Alerting Threshold Configuration', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15, ha='right')
    ax2.set_ylim(0, 1.15)
    ax2.legend(fontsize=10)
    
    def autolabel_metric(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel_metric(rects1)
    autolabel_metric(rects2)
    autolabel_metric(rects3)
                        
    fig.tight_layout()
    out_path = os.path.join(project_root, "docs", "figures", "alert_ablation_metrics.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved alert ablation chart to {out_path}")

if __name__ == "__main__":
    plot_telemetry_overview()
    plot_prediction_vs_groundtruth()
    plot_sota_bar_chart()
    plot_alert_ablation_vis()
    print("All paper figures generated successfully.")
