"""
generate_thesis_charts.py
─────────────────────────
Script xuất biểu đồ chất lượng cao cho luận văn:
  1. So sánh MAE giữa 4 Model trên từng Domain (Grouped Bar)
  2. Heatmap RMSE toàn cục (Model × Domain)
  3. So sánh R² (khả năng giải thích) trên Domain Web NASA
  4. Phân tích Dung lượng dữ liệu qua Pipeline
  5. Kiến trúc Pipeline tổng quan (Diagram placeholder)
"""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── Paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
REPORT_DIR = os.path.join(PROJECT_ROOT, "data", "reports")
CHART_DIR  = os.path.join(REPORT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

CSV_PATH = os.path.join(REPORT_DIR, "lab_results_8D.csv")
df = pd.read_csv(CSV_PATH)

# ─── Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'StandardLSTM':        '#6C757D',
    'SG-TCN-LSTM':         '#0D6EFD',
    'BiLSTM-Attention':    '#FFC107',
    'TCN-DualAtt-BiLSTM':  '#DC3545',
}

DOMAIN_LABELS = {
    'cloud_azure':        'Cloud Azure\n(37 GB)',
    'web_nasa':           'Web NASA\n(HTTP Logs)',
    'micro_rs_anomic':    'RS-Anomic\n(Microservices)',
    'micro_train_ticket': 'Train Ticket\n(Microservices)',
}

MODELS = list(COLORS.keys())

# ═════════════════════════════════════════════════════════════════════
# CHART 1 – Grouped Bar: MAE @ T+5 across all domains
# ═════════════════════════════════════════════════════════════════════
def chart_mae_grouped_bar():
    sub = df[df['Horizon'] == 'T+5'].copy()
    domains = list(DOMAIN_LABELS.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(domains))
    width = 0.18

    for i, model in enumerate(MODELS):
        vals = [sub[(sub['Domain'] == d) & (sub['Model'] == model)]['MAE'].values[0]
                for d in domains]
        bars = ax.bar(x + i * width, vals, width, label=model,
                      color=COLORS[model], edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=7.5, rotation=45)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in domains])
    ax.set_ylabel('MAE (Mean Absolute Error)')
    ax.set_title('So sánh MAE @ T+5 giữa 4 Kiến trúc trên 4 Miền Dữ liệu')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(sub['MAE']) * 1.25)
    fig.savefig(os.path.join(CHART_DIR, 'chart1_mae_grouped_bar.png'))
    plt.close(fig)
    print("[SAVED] chart1_mae_grouped_bar.png")


# ═════════════════════════════════════════════════════════════════════
# CHART 2 – Heatmap: RMSE (Model × Domain), averaged over horizons
# ═════════════════════════════════════════════════════════════════════
def chart_rmse_heatmap():
    pivot = df.groupby(['Model', 'Domain'])['RMSE'].mean().unstack()
    pivot = pivot.reindex(index=MODELS, columns=list(DOMAIN_LABELS.keys()))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(DOMAIN_LABELS)))
    ax.set_xticklabels([DOMAIN_LABELS[d] for d in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10)

    for i in range(len(MODELS)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if v > 0.20 else 'black')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('RMSE trung bình (3 chân trời)')
    ax.set_title('Bản đồ Nhiệt RMSE: Mô hình × Miền Dữ liệu')
    fig.savefig(os.path.join(CHART_DIR, 'chart2_rmse_heatmap.png'))
    plt.close(fig)
    print("[SAVED] chart2_rmse_heatmap.png")


# ═════════════════════════════════════════════════════════════════════
# CHART 3 – Line: MAE vs Horizon for each model on Cloud Azure
# ═════════════════════════════════════════════════════════════════════
def chart_mae_horizon_line():
    sub = df[df['Domain'] == 'cloud_azure'].copy()
    horizons_int = [5, 10, 15]

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in MODELS:
        m = sub[sub['Model'] == model].sort_values('Horizon')
        ax.plot(horizons_int, m['MAE'].values, marker='o', linewidth=2.2,
                label=model, color=COLORS[model], markersize=7)

    ax.set_xticks(horizons_int)
    ax.set_xticklabels([f'T+{h}' for h in horizons_int])
    ax.set_xlabel('Chân trời dự báo (Forecast Horizon)')
    ax.set_ylabel('MAE')
    ax.set_title('Cloud Azure: Xu hướng MAE theo Chân trời Dự báo')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(CHART_DIR, 'chart3_mae_horizon_azure.png'))
    plt.close(fig)
    print("[SAVED] chart3_mae_horizon_azure.png")


# ═════════════════════════════════════════════════════════════════════
# CHART 4 – Bar: R² on web_nasa domain (the domain with positive R²)
# ═════════════════════════════════════════════════════════════════════
def chart_r2_web_nasa():
    sub = df[(df['Domain'] == 'web_nasa') & (df['Horizon'] == 'T+5')].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(sub['Model'], sub['R2'],
                  color=[COLORS[m] for m in sub['Model']],
                  edgecolor='white', linewidth=0.8, width=0.55)
    for bar, v in zip(bars, sub['R2'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('R² (Coefficient of Determination)')
    ax.set_title('Web NASA: Khả năng Giải thích Phương sai (R²) @ T+5')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(-0.05, 0.5)
    fig.savefig(os.path.join(CHART_DIR, 'chart4_r2_web_nasa.png'))
    plt.close(fig)
    print("[SAVED] chart4_r2_web_nasa.png")


# ═════════════════════════════════════════════════════════════════════
# CHART 5 – Pipeline Data Volume Waterfall
# ═════════════════════════════════════════════════════════════════════
def chart_data_pipeline():
    stages  = ['Dữ liệu thô\n(Nén .gz)', 'Giải nén\n(CSV)', 'Trích xuất\n500K dòng', 'Tiêm nhiễu\n20%', 'Windowing\n(Seq=30)', 'Tensor .pt\n(Train+Test)']
    sizes   = [37.0, 120.0, 0.015, 0.015, 0.408, 0.408]  # GB (approx)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors_bar = ['#1565C0', '#1E88E5', '#42A5F5', '#FFA726', '#EF5350', '#AB47BC']
    bars = ax.bar(stages, sizes, color=colors_bar, edgecolor='white', linewidth=0.8, width=0.6)

    for bar, s in zip(bars, sizes):
        label = f'{s:.1f} GB' if s >= 1.0 else f'{s*1024:.0f} MB'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Dung lượng (GB)')
    ax.set_title('Quy trình Xử lý Dữ liệu: Từ 37 GB Thô → Tensor GPU-Ready')
    ax.set_yscale('symlog', linthresh=1)
    ax.grid(axis='y', alpha=0.3)
    fig.savefig(os.path.join(CHART_DIR, 'chart5_data_pipeline.png'))
    plt.close(fig)
    print("[SAVED] chart5_data_pipeline.png")


# ═════════════════════════════════════════════════════════════════════
# CHART 6 – Model Params comparison
# ═════════════════════════════════════════════════════════════════════
def chart_model_params():
    import torch, torch.nn as nn
    sys.path.insert(0, PROJECT_ROOT)
    from src.models.baselines import StandardLSTM
    from src.models.sg_tcn_lstm import MultiHorizonTCNLSTM
    from src.models.bilstm_attention import BiLSTMAttention
    from src.models.tcn_dualatt_bilstm import TCNDualAttBiLSTM

    specs = [
        ('StandardLSTM',       StandardLSTM(7, 64, 2, 3)),
        ('SG-TCN-LSTM',        MultiHorizonTCNLSTM(7, 64, 2, 3)),
        ('BiLSTM-Attention',   BiLSTMAttention(7, 64, 2, 3)),
        ('TCN-DualAtt-BiLSTM', TCNDualAttBiLSTM(7, 64, 2, 3)),
    ]
    names  = [s[0] for s in specs]
    params = [sum(p.numel() for p in s[1].parameters()) for s in specs]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, params,
                   color=[COLORS[n] for n in names],
                   edgecolor='white', linewidth=0.8)
    for bar, p in zip(bars, params):
        ax.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
                f'{p:,}', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Tổng số Tham số (Parameters)')
    ax.set_title('So sánh Quy mô Mô hình (Số lượng Tham số)')
    ax.grid(axis='x', alpha=0.3)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))
    fig.savefig(os.path.join(CHART_DIR, 'chart6_model_params.png'))
    plt.close(fig)
    print("[SAVED] chart6_model_params.png")


# ═════════════════════════════════════════════════════════════════════
# CHART 7 – Radar: Multi-metric comparison on Cloud Azure T+5
# ═════════════════════════════════════════════════════════════════════
def chart_radar_azure():
    sub = df[(df['Domain'] == 'cloud_azure') & (df['Horizon'] == 'T+5')].copy()
    metrics = ['MSE', 'MAE', 'RMSE']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for _, row in sub.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'],
                color=COLORS[row['Model']])
        ax.fill(angles, values, alpha=0.1, color=COLORS[row['Model']])

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_title('Cloud Azure T+5: Đa chỉ số (MSE / MAE / RMSE)', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.savefig(os.path.join(CHART_DIR, 'chart7_radar_azure.png'))
    plt.close(fig)
    print("[SAVED] chart7_radar_azure.png")


# ═════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    chart_mae_grouped_bar()
    chart_rmse_heatmap()
    chart_mae_horizon_line()
    chart_r2_web_nasa()
    chart_data_pipeline()
    chart_model_params()
    chart_radar_azure()
    print("\n✅ ALL 7 CHARTS GENERATED SUCCESSFULLY!")
