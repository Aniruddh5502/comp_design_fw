import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Path Setup
    SCRIPT_DIR = Path(__file__).parent
    CSV_PATH = SCRIPT_DIR.parent / "sweep_results.csv"
    PLOT_DIR = SCRIPT_DIR.parent / "plots"
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CSV_PATH.exists():
        print(f"Error: Results file not found at {CSV_PATH}")
        return

    # Load data
    df = pd.read_csv(CSV_PATH)
    df = df.sort_values("lipschitz")
    
    # Design Preferences (High-quality research style)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    
    # Define metrics to plot
    metrics = [
        ("mean_rmse", "Mean RMSE", "Lower is Better", "#CC79A7"),
        ("mean_spearman", "Mean Spearman $\\rho$", "Higher is Better", "#009E73"),
        ("mean_ratio", "Mean Ratio (RMSE/Std)", "Close to 1.0 is Better", "#56B4E9")
    ]
    
    # Create 3 separate plots for better clarity in reports
    for col, label, desc, color in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.scatter(df["lipschitz"], df[col], 
                marker='o', s=60,
                color=color, label=label)
        
        ax.set_title(f"Effect of Lipschitz Constant on {label}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Lipschitz Constant ($L$)", fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        
        # Add markers for the trade-off points
        for x, y in zip(df["lipschitz"], df[col]):
            ax.text(x, y, f'{y:.4f}', ha='center', va='bottom', fontsize=9)
            
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xticks(df["lipschitz"])
        
        # Save as SVG
        file_name = f"sweep_{col}.svg"
        fig.savefig(PLOT_DIR / file_name, bbox_inches='tight', format='svg')
        plt.close(fig)
        print(f"Saved plot: {file_name}")

    # Summary Plot (Combined)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color_rmse = "#CC79A7"
    ax1.set_xlabel("Lipschitz Constant ($L$)")
    ax1.set_ylabel("Mean RMSE", color=color_rmse, fontweight='bold')
    ax1.plot(df["lipschitz"], df["mean_rmse"], color=color_rmse, marker='o', linewidth=2, label="RMSE")
    ax1.tick_params(axis='y', labelcolor=color_rmse)
    
    ax2 = ax1.twinx()
    color_rho = "#009E73"
    ax2.set_ylabel("Mean Spearman $\\rho$", color=color_rho, fontweight='bold')
    ax2.plot(df["lipschitz"], df["mean_spearman"], color=color_rho, marker='s', linewidth=2, label="Spearman")
    ax2.tick_params(axis='y', labelcolor=color_rho)
    
    plt.title("The Reliability Tax: Accuracy vs. Distance-Awareness", fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    fig.savefig(PLOT_DIR / "reliability_tax_tradeoff.svg", bbox_inches='tight', format='svg')
    plt.close(fig)
    print("Saved Summary Plot: reliability_tax_tradeoff.svg")

if __name__ == "__main__":
    main()
