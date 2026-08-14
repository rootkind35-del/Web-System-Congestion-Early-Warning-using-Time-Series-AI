import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
csv_path = os.path.join(project_root, "data", "reports", "lab_results.csv")
fig_dir = os.path.join(project_root, "docs", "figures")

os.makedirs(fig_dir, exist_ok=True)

# Load Data
df = pd.read_csv(csv_path)

# Filter only In-Domain (Azure) for the paper
df_azure = df[df["Domain"] == "In-Domain (Azure)"].copy()

# Ensure chronological ordering of horizons
df_azure["Horizon"] = pd.Categorical(df_azure["Horizon"], categories=["T+5", "T+10", "T+15"], ordered=True)
df_azure = df_azure.sort_values(by=["Model", "Horizon"])

# Style settings suitable for Scientific Papers
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42})

# 1. Bar Chart: MAE Comparison
plt.figure(figsize=(10, 6))
bar_plot = sns.barplot(
    data=df_azure,
    x="Horizon",
    y="MAE",
    hue="Model",
    palette="viridis"
)
plt.title("So sánh Sai số Tuyệt đối Trung bình (MAE) theo Horizon", fontsize=14, pad=15)
plt.xlabel("Horizon (Phút)", fontsize=12)
plt.ylabel("MAE (càng thấp càng tốt)", fontsize=12)
plt.legend(title="Kiến trúc Mô hình", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
mae_fig_path = os.path.join(fig_dir, "mae_comparison_bar.png")
plt.savefig(mae_fig_path, dpi=300, bbox_inches='tight')
plt.close()

# 2. Line Chart: RMSE Trend across Horizons
plt.figure(figsize=(10, 6))
line_plot = sns.lineplot(
    data=df_azure,
    x="Horizon",
    y="RMSE",
    hue="Model",
    marker="o",
    linewidth=2.5,
    markersize=8,
    palette="tab10"
)
plt.title("Xu hướng Lỗi RMSE khi tăng Horizon", fontsize=14, pad=15)
plt.xlabel("Horizon (Phút)", fontsize=12)
plt.ylabel("RMSE", fontsize=12)
plt.legend(title="Kiến trúc Mô hình")
plt.tight_layout()
rmse_fig_path = os.path.join(fig_dir, "rmse_trend_line.png")
plt.savefig(rmse_fig_path, dpi=300, bbox_inches='tight')
plt.close()

# 3. Bar Chart: MSE Comparison
plt.figure(figsize=(10, 6))
bar_plot_mse = sns.barplot(
    data=df_azure,
    x="Horizon",
    y="MSE",
    hue="Model",
    palette="magma"
)
plt.title("So sánh Lỗi Bình phương Trung bình (MSE) theo Horizon", fontsize=14, pad=15)
plt.xlabel("Horizon (Phút)", fontsize=12)
plt.ylabel("MSE", fontsize=12)
plt.legend(title="Kiến trúc Mô hình", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
mse_fig_path = os.path.join(fig_dir, "mse_comparison_bar.png")
plt.savefig(mse_fig_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Successfully generated scientific plots in: {fig_dir}")
