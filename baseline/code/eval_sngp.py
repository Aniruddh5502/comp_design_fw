import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from baseline.code.sngp import SNGP

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "baseline" / "results"
PLOTS = ROOT / "baseline" / "plots"
CKPT = ROOT  / "baseline" / "models" / "sngp_rff_model.pth"


def corr(a, b, method):
    """Correlation, NaN when either side is constant (never a fake 0)."""
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method=method))


def main():
    test_df = pd.read_csv(PROC / "test_dataset.csv")
    X_test = torch.tensor(test_df.iloc[:, :4].values, dtype=torch.float32)
    Y_test = torch.tensor(test_df.iloc[:, 4:].values, dtype=torch.float32)

    if not CKPT.exists():
        print(f"No checkpoint at {CKPT}. Run train_sngp_rff.py first.")
        return

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = SNGP(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    mean, var = model.predict(X_test)
    mu = mean.numpy()
    latent_var = var.numpy()
    y = Y_test.numpy()
    noise_var = float(model.gp.noise_var.item())

    # Sanity checks: variance should vary across inputs and grow for inputs far from the data.
    print("Latent variance min/max/std:",
          latent_var.min(), latent_var.max(), latent_var.std())
    _, var_far = model.predict(X_test * 2.0)  # heuristic far-from-data shift
    print(f"Mean latent variance in-distribution: {latent_var.mean():.4e} | "
          f"scaled x2: {var_far.numpy().mean():.4e}")

    err = np.abs(y - mu)
    n_out = y.shape[1]
    pearson = [corr(err[:, i], latent_var[:, i], "pearson") for i in range(n_out)]
    spearman = [corr(err[:, i], latent_var[:, i], "spearman") for i in range(n_out)]

    # Gaussian NLL of observations under the predictive N(mu, noise + latent var).
    pred_var = latent_var + noise_var
    nll = 0.5 * (np.log(2 * math.pi * pred_var) + (y - mu) ** 2 / pred_var)

    rmse_per_dim = np.sqrt(np.mean((y - mu) ** 2, axis=0)).tolist()
    results = {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "rmse_per_dim": rmse_per_dim,
        "nll": float(nll.mean()),
        "noise_var": noise_var,
        "pearson_per_dim": pearson,
        "spearman_per_dim": spearman,
        "mean_pearson": float(np.nanmean(pearson)),
        "mean_rho": float(np.nanmean(spearman)),  # Spearman, headline metric
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "sngp_rff_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_out):
        ax.scatter(latent_var[:, i], err[:, i], alpha=0.4, label=f"Dim {i+1}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latent predictive variance (log)")
    ax.set_ylabel("Absolute prediction error (log)")
    ax.set_title(f"SNGP-RFF distance awareness (mean Spearman $\\rho_s$={results['mean_rho']:.3f})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.savefig(PLOTS / "sngp_rff_uq_correlation.png", dpi=150)
    fig.savefig(PLOTS / "sngp_rff_uq_correlation.svg")
    plt.close(fig)

    print(json.dumps(results, indent=2))
    print(f"Results saved to {RESULTS / 'sngp_rff_metrics.json'}")


if __name__ == "__main__":
    main()