import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
con = Console()

root = Path(__file__).parent.parent
convergence_file = root / "results" / "convergence.csv"
plot_file = root / "plots" / "convergence_plot.svg"

def main():
    # loading dataframe
    df = pd.read_csv(convergence_file)
    
    # imports
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AnthropicSerif"
    
    fig, ax = plt.subplots()
    fig.set_figheight(3)
    fig.set_figwidth(4)
    
    ax.errorbar(df["size"], df["mean_r"], df["std_r"], fmt="-o", color="#4477AA", ecolor="#CC79A7", elinewidth=3, capsize=5, markersize=8, label='Mean R-test')


    ax.set_title("Ensemble Surrogate Convergence Study", fontsize=14)
    ax.set_xlabel("Training Set Size (N)", fontsize=10)
    ax.set_ylabel("Mean Pearson Correlation (R_test)", fontsize=10)
    ax.grid(True, alpha=0.8)
    ax.legend()
    
    fig.tight_layout(pad=0.1)
    fig.savefig(plot_file, bbox_inches='tight', pad_inches=0.1)
    con.print(f"[x]    Plot has been saved to {plot_file}")
    
    
if __name__ == "__main__":
    main()
