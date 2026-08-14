import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "data", "reports")
ARTIFACTS_DIR = os.path.join("C:", os.sep, "Users", "dhp01", ".gemini", "antigravity", "brain", "a5ab832a-ba55-4ae3-8a7b-7148a645de24")

benchmark_csv = os.path.join(REPORTS_DIR, "full_benchmark_results.csv")
drift_csv = os.path.join(REPORTS_DIR, "drift_simulation.csv")

if os.path.exists(benchmark_csv):
    df_bm = pd.read_csv(benchmark_csv)
    # Simplify model names for charts
    df_bm['ShortName'] = df_bm['Model'].str.replace('_MultiTask', '').str.replace('TCNDualAttBiLSTM', 'TCNDualAttBiLSTM (Ours)')

    # Chart 1: F1-Score & R2-Score Comparison
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_bm))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df_bm['F1_Binary'], width, label='F1 Binary Score', color='#1f77b4', alpha=0.85)
    rects2 = ax1.bar(x + width/2, df_bm['R2_Score'], width, label='R² Score (Risk)', color='#2ca02c', alpha=0.85)

    ax1.set_ylabel('Score (Higher is Better)')
    ax1.set_title('Model Performance Comparison (Azure Cloud Data)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_bm['ShortName'], rotation=15, ha='right')
    ax1.set_ylim(0.75, 0.90)
    ax1.legend(loc='upper right')

    # Add values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    chart1_path = os.path.join(ARTIFACTS_DIR, "fig1_model_performance.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved {chart1_path}")

    # Chart 2: Inference Latency vs VRAM Footprint
    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(df_bm['Inference_ms'], df_bm['VRAM_MB'], s=df_bm['MAE']*50000, c=df_bm['F1_Binary'], 
                         cmap='viridis', alpha=0.8, edgecolors='black', linewidth=1.5)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('F1 Score', rotation=270, labelpad=15)

    for i, txt in enumerate(df_bm['ShortName']):
        ax.annotate(txt, (df_bm['Inference_ms'].iloc[i] + 0.02, df_bm['VRAM_MB'].iloc[i] + 0.5), fontsize=10, fontweight='bold')

    ax.set_xlabel('Inference Latency per Batch (ms)')
    ax.set_ylabel('Peak VRAM Allocation (MB)')
    ax.set_title('Production Constraints: Latency vs. Memory Footprint', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    chart2_path = os.path.join(ARTIFACTS_DIR, "fig2_latency_vs_vram.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved {chart2_path}")

if os.path.exists(drift_csv):
    df_drift = pd.read_csv(drift_csv)
    
    # Chart 3: Concept Drift Page-Hinkley & Recovery
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_drift['batch'], df_drift['mae'], color='#d62728', linewidth=2, label='Batch MAE')
    
    # Annotate Drift Injection and Retrain
    ax.axvline(x=100, color='orange', linestyle='--', linewidth=1.5, label='Drift Injected (Flash Crowd)')
    ax.axvline(x=105, color='green', linestyle='-', linewidth=2, label='Page-Hinkley Triggered (Online Retrain)')
    
    ax.annotate('Flash Crowd Shift\n(MAE Spike to 0.16)', xy=(100, 0.16), xytext=(30, 0.14),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="b", lw=1, alpha=0.5))

    ax.annotate('Online Retraining\n(MAE Recovered to 0.016)', xy=(300, 0.016), xytext=(210, 0.07),
                arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=6),
                fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="g", lw=1, alpha=0.5))

    ax.set_xlabel('Streaming Mini-Batch Index')
    ax.set_ylabel('Mean Absolute Error (MAE)')
    ax.set_title('Concept Drift Adaptation: Page-Hinkley Test & Online Retraining', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart3_path = os.path.join(ARTIFACTS_DIR, "fig3_concept_drift_recovery.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved {chart3_path}")
