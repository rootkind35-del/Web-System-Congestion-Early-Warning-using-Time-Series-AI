import matplotlib.pyplot as plt
import numpy as np
import os

def plot_sota_mae_comparison():
    # Data from Exhaustive Benchmark
    models = [
        'TCN-DualAtt-BiLSTM (Ours)', 
        'iTransformer (2024)', 
        'DLinear (2023)', 
        'CNN-Patch (2023)', 
        'TS-Mixer (2023)'
    ]
    
    t5_mae = [2.282, 2.282, 2.339, 2.331, 2.604]
    t10_mae = [2.410, 2.449, 2.462, 2.519, 2.671]
    t15_mae = [2.649, 2.698, 2.707, 2.753, 3.015]

    x = np.arange(len(models))
    width = 0.25

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Colorblind friendly colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    rects1 = ax.bar(x - width, t5_mae, width, label='T+5 (5 min)', color=colors[0], alpha=0.9)
    rects2 = ax.bar(x, t10_mae, width, label='T+10 (10 min)', color=colors[1], alpha=0.9)
    rects3 = ax.bar(x + width, t15_mae, width, label='T+15 (15 min)', color=colors[2], alpha=0.9)

    # Add text for labels, title and custom x-axis tick labels
    ax.set_ylabel('Mean Absolute Error (MAE) ↓ Lower is Better', fontsize=12, fontweight='bold')
    ax.set_title('Exhaustive Architecture Benchmark (2020-2024) - Web Congestion Prediction', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, rotation=15, ha='right')
    ax.legend(fontsize=11, loc='upper left')

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Highlight our model
    for patch in rects1.patches + rects2.patches + rects3.patches:
        if patch.get_x() < 0: # The first group
            patch.set_hatch('//')

    fig.tight_layout()
    
    # Save figure
    os.makedirs('docs/figures', exist_ok=True)
    out_path = 'docs/figures/sota_architecture_benchmark.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved benchmark plot to {out_path}")

if __name__ == "__main__":
    plot_sota_mae_comparison()
