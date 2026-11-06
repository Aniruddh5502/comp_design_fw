"""
Evaluate ensemble on test set.
- Spearman & Pearson correlation (variance vs abs error) per output
- Reliability diagram: variance vs abs error, 4 quadrants
- Training curves: train/val loss
- Correlation heatmap
"""
import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from utils.data_processing import load_splits, inverse_transform
from core.model import Ensemble
from core.trainer import create_dataloaders, PROJECT_ROOT


TARGET_COLS = [
    "max_stress_von_mises",
    "max_deformation",
    "mode_1_freq",
    "mode_2_freq",
    "mode_3_freq",
    "mode_4_freq",
]

CHECKPOINT_DIR = PROJECT_ROOT / "tests/ensemble_test/checkpoints"
RESULTS_DIR = PROJECT_ROOT / "tests/ensemble_test/results"
DATA_DIR = PROJECT_ROOT / "tests/ensemble_test/data"

# Colorblind-safe palette (Wong palette)
PALETTE_CATEGORICAL = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
PALETTE_SEQUENTIAL = 'viridis'
QUADRANT_COLORS = {
    'q1': '#C44E52',      # High var, High error - red
    'q2': '#DD8452',      # High var, Low error - orange
    'q3': '#8B0000',      # Low var, High error - dark red (dangerous)
    'q4': '#55A868',      # Low var, Low error - green
}

# Font configuration
FONT_SERIF = 'Anthropic Serif Web Text'
FONT_MONO = 'Anthropic Mono Variable'
FONT_SIZE_BASE = 14
FONT_SIZE_SMALL = 12
FIG_DPI = 300
MIN_FIG_SIZE = (8, 4)


def setup_style():
    """Apply professional white-theme style with Anthropic fonts."""
    plt.rcParams.update({
        'figure.figsize': MIN_FIG_SIZE,
        'figure.dpi': FIG_DPI,
        'font.size': FONT_SIZE_BASE,
        'font.family': FONT_SERIF,
        'axes.titlesize': FONT_SIZE_BASE + 2,
        'axes.titleweight': 'bold',
        'axes.labelsize': FONT_SIZE_BASE,
        'xtick.labelsize': FONT_SIZE_SMALL,
        'ytick.labelsize': FONT_SIZE_SMALL,
        'legend.fontsize': FONT_SIZE_SMALL,
        'figure.titlesize': FONT_SIZE_BASE + 4,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'axes.labelcolor': '#333333',
        'text.color': '#333333',
        'grid.color': '#CCCCCC',
        'grid.alpha': 0.8,
        'grid.linestyle': '-',
        'grid.linewidth': 1,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })
    # Monospace for specific elements (handled per-element via fontfamily kwarg)


def load_ensemble(ensemble_size: int, checkpoint_path: str, device: str) -> Ensemble:
    """Load trained ensemble from checkpoint."""
    ensemble = Ensemble(ensemble_size=ensemble_size).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for i, model in enumerate(ensemble.models):
        model.load_state_dict(checkpoint["model_state"][f"model_{i}"])
    ensemble.eval()
    return ensemble


@torch.no_grad()
def evaluate(ensemble: Ensemble, test_loader, device: str):
    """Get predictions, variance, and errors on test set."""
    ensemble.eval()
    all_mean = []
    all_var = []
    all_true = []
    all_individual = []

    for xb, yb in test_loader:
        xb = xb.to(device)
        mean, var, individual = ensemble.predict_with_uncertainty(xb)
        all_mean.append(mean.cpu().numpy())
        all_var.append(var.cpu().numpy())
        all_true.append(yb.numpy())
        all_individual.append(individual.cpu().numpy())

    mean = np.vstack(all_mean)        # [N, 6]
    var = np.vstack(all_var)          # [N, 6]
    true = np.vstack(all_true)        # [N, 6]
    individual = np.concatenate(all_individual, axis=1)  # [E, N, 6]

    abs_error = np.abs(mean - true)   # [N, 6]
    return mean, var, true, abs_error, individual


def correlation_analysis(var: np.ndarray, abs_error: np.ndarray, target_names: list) -> pd.DataFrame:
    """Compute Spearman & Pearson per target."""
    results = []
    for i, name in enumerate(target_names):
        v = var[:, i]
        e = abs_error[:, i]
        rho_s, p_s = spearmanr(v, e)
        rho_p, p_p = pearsonr(v, e)
        results.append({
            "target": name,
            "spearman_rho": rho_s,
            "spearman_p": p_s,
            "pearson_r": rho_p,
            "pearson_p": p_p,
        })
    return pd.DataFrame(results)


def plot_reliability_diagram(var: np.ndarray, abs_error: np.ndarray, target_names: list, save_dir: Path):
    """Reliability diagram: variance vs abs error, 4 quadrants with median splits.
    Generates one image per target."""
    setup_style()
    
    for i, name in enumerate(target_names):
        fig, ax = plt.subplots(figsize=MIN_FIG_SIZE)
        v = var[:, i]
        e = abs_error[:, i]

        v_med = np.median(v)
        e_med = np.median(e)

        # Quadrant masks
        mask_q1 = (v >= v_med) & (e >= e_med)  # Uncertain, Wrong
        mask_q2 = (v >= v_med) & (e < e_med)   # Uncertain, Right
        mask_q3 = (v < v_med) & (e >= e_med)   # Confident, Wrong (DANGEROUS)
        mask_q4 = (v < v_med) & (e < e_med)    # Confident, Right

        # Scatter with distinct colors
        ax.scatter(v[mask_q1], e[mask_q1], c=QUADRANT_COLORS['q1'], alpha=0.6, s=18, label='High Var, High Err', edgecolors='none')
        ax.scatter(v[mask_q2], e[mask_q2], c=QUADRANT_COLORS['q2'], alpha=0.6, s=18, label='High Var, Low Err', edgecolors='none')
        ax.scatter(v[mask_q3], e[mask_q3], c=QUADRANT_COLORS['q3'], alpha=0.6, s=18, label='Low Var, High Err', edgecolors='none')
        ax.scatter(v[mask_q4], e[mask_q4], c=QUADRANT_COLORS['q4'], alpha=0.6, s=18, label='Low Var, Low Err', edgecolors='none')

        # Median lines
        ax.axvline(v_med, color='#888888', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(e_med, color='#888888', linestyle='--', linewidth=1, alpha=0.7)

        ax.set_xlabel('Ensemble Variance', fontfamily=FONT_SERIF)
        ax.set_ylabel('Absolute Error', fontfamily=FONT_SERIF)
        ax.set_title(name.replace('_', ' ').title(), fontfamily=FONT_SERIF, fontweight='bold', fontsize=FONT_SIZE_BASE + 2)
        
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='#CCCCCC', fontsize=FONT_SIZE_SMALL - 1,
                 prop={'family': FONT_SERIF})

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#CCCCCC')

        plt.tight_layout()
        safe_name = name.replace(' ', '_')
        plt.savefig(save_dir / f"reliability_diagram_{safe_name}.svg", bbox_inches='tight', pad_inches=0.25, facecolor='white')
        plt.close()


def plot_training_curves(history_path: Path, save_dir: Path):
    """Plot train/val loss curves with professional styling."""
    setup_style()
    
    import json
    with open(history_path) as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=(max(MIN_FIG_SIZE[0] * 1.2, 10), max(MIN_FIG_SIZE[1] * 1.2, 5)))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label='Train Loss', color=PALETTE_CATEGORICAL[0], linewidth=2.5)
    ax.plot(epochs, history["val_loss"], label='Val Loss', color=PALETTE_CATEGORICAL[1], linewidth=2.5)
    
    # Mark best epoch
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_val = min(history["val_loss"])
    ax.axvline(best_epoch, color='#888888', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.plot(best_epoch, best_val, 'o', color=PALETTE_CATEGORICAL[2], markersize=10, label=f'Best Val (Epoch {best_epoch})')

    ax.set_xlabel('Epoch', fontfamily=FONT_SERIF)
    ax.set_ylabel('MSE Loss', fontfamily=FONT_SERIF)
    ax.set_title('Training Curves', fontfamily=FONT_SERIF, fontweight='bold', fontsize=FONT_SIZE_BASE + 2)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#CCCCCC', fontsize=FONT_SIZE_SMALL,
              prop={'family': FONT_SERIF})
    ax.set_yscale('log')
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#CCCCCC')

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.svg", bbox_inches='tight', pad_inches=0.25, facecolor='white')
    plt.close()


def plot_correlation_heatmap(corr_df: pd.DataFrame, save_dir: Path):
    """Plot correlation heatmap (Spearman & Pearson)."""
    setup_style()
    
    # Prepare data for heatmap
    targets = corr_df['target'].values
    spearman_vals = corr_df['spearman_rho'].values
    pearson_vals = corr_df['pearson_r'].values
    
    data = np.vstack([spearman_vals, pearson_vals])
    row_labels = ['Spearman rho', 'Pearson r']
    
    fig_w = max(MIN_FIG_SIZE[0] * 1.2, 10)
    fig_h = max(MIN_FIG_SIZE[1] * 0.8, 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    im = ax.imshow(data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t.replace('_', ' ').title() for t in targets], rotation=45, ha='right', 
                       fontfamily=FONT_SERIF, fontsize=FONT_SIZE_SMALL)
    ax.set_yticks(range(2))
    ax.set_yticklabels(row_labels, fontfamily=FONT_SERIF, fontsize=FONT_SIZE_SMALL)
    
    # Add text annotations
    for i in range(2):
        for j in range(len(targets)):
            val = data[i, j]
            color = 'white' if abs(val) > 0.5 else '#333333'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontweight='bold', 
                    fontsize=FONT_SIZE_SMALL, fontfamily=FONT_MONO)
    
    ax.set_title('Variance vs Error Correlation by Target', fontfamily=FONT_SERIF, fontweight='bold', 
                 fontsize=FONT_SIZE_BASE + 2, pad=15)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontfamily=FONT_SERIF, fontsize=FONT_SIZE_SMALL, 
                   rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=FONT_SIZE_SMALL - 1)
    
    plt.tight_layout()
    plt.savefig(save_dir / "correlation_heatmap.svg", bbox_inches='tight', pad_inches=0.25, facecolor='white')
    plt.close()


def plot_error_distribution(abs_error: np.ndarray, target_names: list, save_dir: Path):
    """Plot error distribution per target (small multiples)."""
    setup_style()
    
    n_targets = len(target_names)
    n_cols = 3
    n_rows = (n_targets + n_cols - 1) // n_cols
    
    fig_w = max(MIN_FIG_SIZE[0] * n_cols / 1.8, 10)
    fig_h = max(MIN_FIG_SIZE[1] * n_rows / 1.2, 6)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    axes = axes.flatten() if n_targets > 1 else [axes]

    for i, name in enumerate(target_names):
        ax = axes[i]
        errors = abs_error[:, i]
        
        # Histogram
        n, bins, patches = ax.hist(errors, bins=30, color=PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)], 
                                    edgecolor='white', alpha=0.85, linewidth=0.5)
        
        # Mean and median lines
        mean_err = errors.mean()
        median_err = np.median(errors)
        ax.axvline(mean_err, color='#C44E52', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_err:.3f}')
        ax.axvline(median_err, color='#55A868', linestyle='--', linewidth=2, 
                   label=f'Median: {median_err:.3f}')
        
        ax.set_xlabel('Absolute Error', fontfamily=FONT_SERIF)
        ax.set_ylabel('Frequency', fontfamily=FONT_SERIF)
        ax.set_title(name.replace('_', ' ').title(), fontfamily=FONT_SERIF, fontweight='bold', fontsize=FONT_SIZE_BASE + 1)
        
        ax.legend(fontsize=FONT_SIZE_SMALL - 2, frameon=True, facecolor='white', edgecolor='#CCCCCC',
                  prop={'family': FONT_SERIF})
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#CCCCCC')

    for j in range(n_targets, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Absolute Error Distribution by Target', fontfamily=FONT_SERIF, fontweight='bold', 
                 fontsize=FONT_SIZE_BASE + 4, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "error_distribution.svg", bbox_inches='tight', pad_inches=0.25, facecolor='white')
    plt.close()


def save_predictions(mean, var, true, abs_error, individual, bounds, target_names, save_dir: Path):
    """Save all predictions to CSV."""
    # Inverse transform to original scale
    mean_orig = inverse_transform(pd.DataFrame(mean, columns=target_names), bounds, target_names)
    var_orig = inverse_transform(pd.DataFrame(var, columns=target_names), bounds, target_names)
    true_orig = inverse_transform(pd.DataFrame(true, columns=target_names), bounds, target_names)
    abs_err_orig = inverse_transform(pd.DataFrame(abs_error, columns=target_names), bounds, target_names)

    # Individual predictions: [E, N, D] -> save each ensemble member
    E, N, D = individual.shape
    indiv_orig = []
    for e in range(E):
        df_e = inverse_transform(pd.DataFrame(individual[e], columns=target_names), bounds, target_names)
        df_e["ensemble_member"] = e
        indiv_orig.append(df_e)
    indiv_df = pd.concat(indiv_orig, ignore_index=True)

    # Main predictions
    pred_df = pd.DataFrame({
        f"{t}_pred": mean_orig[t] for t in target_names
    })
    pred_df = pd.concat([pred_df,
                         pd.DataFrame({f"{t}_var": var_orig[t] for t in target_names}),
                         pd.DataFrame({f"{t}_true": true_orig[t] for t in target_names}),
                         pd.DataFrame({f"{t}_abs_err": abs_err_orig[t] for t in target_names})], axis=1)

    pred_df.to_csv(save_dir / "test_predictions.csv", index=False)
    indiv_df.to_csv(save_dir / "individual_predictions.csv", index=False)
    print(f"Saved predictions to {save_dir}")


def main(
    ensemble_size: int = 10,
    checkpoint_path: str = None,
    device: str = "cpu",
):
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_DIR / "best_ensemble.pt"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, val_df, test_df, bounds = load_splits(DATA_DIR)
    _, _, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size=32, device=device)

    # Load ensemble
    ensemble = load_ensemble(ensemble_size, checkpoint_path, device)

    # Evaluate
    print("Evaluating on test set...")
    mean, var, true, abs_error, individual = evaluate(ensemble, test_loader, device)

    # Correlation analysis
    print("\n=== Correlation Analysis (Variance vs Abs Error) ===")
    corr_df = correlation_analysis(var, abs_error, TARGET_COLS)
    print(corr_df.to_string(index=False))
    corr_df.to_csv(RESULTS_DIR / "correlation_analysis.csv", index=False)

    # Generate plots
    print("\nGenerating reliability diagram...")
    plot_reliability_diagram(var, abs_error, TARGET_COLS, RESULTS_DIR)

    print("Generating training curves...")
    plot_training_curves(CHECKPOINT_DIR / "training_history.json", RESULTS_DIR)

    print("Generating correlation heatmap...")
    plot_correlation_heatmap(corr_df, RESULTS_DIR)

    print("Generating error distribution...")
    plot_error_distribution(abs_error, TARGET_COLS, RESULTS_DIR)

    # Save predictions
    save_predictions(mean, var, true, abs_error, individual, bounds, TARGET_COLS, RESULTS_DIR)

    print("\nDone. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(device=device)