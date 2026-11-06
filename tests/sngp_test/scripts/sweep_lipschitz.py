import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import stats

from comp_design_fw.tests.sngp_test.core.model import setup_model
from comp_design_fw.tests.sngp_test.core.trainer import SNGPTrainer
from comp_design_fw.tests.sngp_test.utils.data_cleaning import load_and_clean_data

SEED = 0


def collect(model, loader, device):
    """Returns (means, vars, targets) as numpy arrays over the whole loader."""
    means, variances, targets = [], [], []
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            m, v = model.predict(X.to(device), device)
            means.append(m.cpu())
            variances.append(v.cpu())
            targets.append(y)
    return (torch.cat(means).numpy(),
            torch.cat(variances).numpy(),
            torch.cat(targets).numpy())


def metrics(means, variances, targets):
    """Per-output RMSE, Spearman(|error|, variance), and RMSE/avg-std; averaged over outputs."""
    rmses, rhos, ratios = [], [], []
    for i in range(targets.shape[1]):
        error = targets[:, i] - means[:, i]
        abs_error = np.abs(error)
        var_i = variances[:, i]

        rmse = np.sqrt(np.mean(error ** 2))
        avg_std = np.sqrt(np.mean(var_i))
        rho, _ = stats.spearmanr(abs_error, var_i)

        rmses.append(rmse)
        rhos.append(rho)
        ratios.append(rmse / avg_std if avg_std != 0 else 0.0)

    return {
        "mean_rmse": float(np.mean(rmses)),
        "mean_spearman": float(np.mean(rhos)),
        "mean_ratio": float(np.mean(ratios)),
    }


def variance_stats(variances, noise_var):
    """
    Splits predicted variance into the total and the GP part (total minus noise floor),
    and reports min / median / max plus how often the GP part is below the noise floor.
    """
    noise = np.asarray(noise_var).reshape(1, -1)
    gp_part = variances - noise

    def summary(x):
        return float(np.min(x)), float(np.median(x)), float(np.max(x))

    total_min, total_med, total_max = summary(variances)
    gp_min, gp_med, gp_max = summary(gp_part)
    frac_below_floor = float(np.mean(gp_part < noise))

    return {
        "total_min": total_min, "total_med": total_med, "total_max": total_max,
        "gp_min": gp_min, "gp_med": gp_med, "gp_max": gp_max,
        "frac_gp_below_floor": frac_below_floor,
        "noise_floor": float(np.mean(noise)),
    }


def print_variance_stats(tag, s):
    print(f"  [{tag}] total var  min={s['total_min']:.3e}  med={s['total_med']:.3e}  max={s['total_max']:.3e}")
    print(f"  [{tag}] GP part    min={s['gp_min']:.3e}  med={s['gp_med']:.3e}  max={s['gp_max']:.3e}")
    print(f"  [{tag}] noise floor={s['noise_floor']:.3e}  "
          f"fraction of GP part below floor={s['frac_gp_below_floor']:.3f}")


def main():
    SCRIPT_DIR = Path(__file__).parent
    ROOT = SCRIPT_DIR.parent.parent.parent.parent
    CSV_PATH = ROOT / "data" / "raw" / "inputs.csv"
    RESULTS_PATH = SCRIPT_DIR.parent / "sweep_results.csv"

    # Spectral-norm coefficient per layer. Network Lipschitz bound is c^3 (3 layers, 1-Lipschitz ReLU).
    C_VALUES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    train_loader, val_loader, test_loader = load_and_clean_data(
        csv_path=str(CSV_PATH), batch_size=32)
    train_samples = len(train_loader.dataset)
    val_samples = len(val_loader.dataset)
    test_samples = len(test_loader.dataset)
    RFF_DIM = 1024

    print(f"N_train={train_samples}  N_val={val_samples}  N_test={test_samples}  rff_dim={RFF_DIM}")
    if train_samples < RFF_DIM:
        print("WARNING: N_train < rff_dim, Phi^T Phi is rank-deficient.")

    results = []
    for c in C_VALUES:
        print(f"\n--- Spectral-norm coefficient c = {c} (Lipschitz bound <= {c**3:g}) ---")

        # Same seed for every c, so the random Fourier features and init are identical
        # and only c changes.
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        model, criterion, optimizer, device = setup_model(
            input_dim=4, output_dim=6,
            lipschitz_constant=c,
            num_samples=train_samples,
        )

        # Early stopping on validation loss
        trainer = SNGPTrainer(model, criterion, optimizer, device, patience=20)
        trainer.train(train_loader, val_loader, epochs=500)

        # Fit the GP posterior on the training features
        model.fit_gp(train_loader, device)

        val_arrays = collect(model, val_loader, device)
        test_arrays = collect(model, test_loader, device)
        val_m = metrics(*val_arrays)
        test_m = metrics(*test_arrays)

        noise_var = model.noise_var.cpu().numpy()
        val_vs = variance_stats(val_arrays[1], noise_var)
        test_vs = variance_stats(test_arrays[1], noise_var)

        row = {"lipschitz": c}
        row.update(test_m)
        row.update({f"val_{k}": v for k, v in val_m.items()})
        results.append(row)

        print(f"c={c} | "
              f"Test RMSE: {test_m['mean_rmse']:.5f} "
              f"Spearman: {test_m['mean_spearman']:.4f} "
              f"Ratio: {test_m['mean_ratio']:.3f} | "
              f"Val Spearman: {val_m['mean_spearman']:.4f}")
        print_variance_stats("test", test_vs)
        print_variance_stats("val ", val_vs)

    df = pd.DataFrame(results)
    df = df[["lipschitz", "mean_rmse", "mean_spearman", "mean_ratio",
             "val_mean_rmse", "val_mean_spearman", "val_mean_ratio"]]
    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSweep complete. Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()