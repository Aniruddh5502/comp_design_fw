import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Professional style setup as per .momo/skills/data-visualization/SKILL.md
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams["font.family"] = "Serif"
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 16,
})

PALETTE_CATEGORICAL = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

def visualize_dataset(file_path, plot_dir):
    df = pd.read_csv(file_path)
    df_num = df.select_dtypes(include=['number'])
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Correlation Heatmap (Correlation many variables)
    plt.figure(figsize=(12, 9))
    sns.heatmap(df_num.corr(), annot=True, cmap='viridis', center=0, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, 'correlation_heatmap.svg'), bbox_inches='tight')
    plt.close()
    
    # 2. Distributions (Histogram)
    for col in df_num.columns:
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.histplot(df_num[col], bins=30, color=PALETTE_CATEGORICAL[0], edgecolor='white', alpha=0.8, ax=ax)
        
        mean_val = df_num[col].mean()
        median_val = df_num[col].median()
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median: {median_val:.2f}')
        
        ax.set_title(f'Distribution of {col}', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'dist_{col}.png'), dpi=300, bbox_inches='tight', pad_inches=1)
        plt.savefig(os.path.join(plot_dir, f'dist_{col}.svg'), bbox_inches='tight', pad_inches=1)
        plt.close()

if __name__ == "__main__":
    visualize_dataset('data/processed/full_processed_dataset.csv', 'analysis/plots')
