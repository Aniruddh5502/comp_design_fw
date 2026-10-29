from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from baseline.code.sngp import SNGP

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "processed" / "train_1500.csv"
TEST_PATH = ROOT / "data" / "processed" / "test_dataset.csv"
SAVE_PATH = ROOT / "baseline" / "models" / "sngp_rff_model.pth"
PLOT_PATH = ROOT / "baseline" / "plots" / "sngp_rff_loss.png"
PLOT_PATH_SVG = ROOT / "baseline" / "plots" / "sngp_rff_loss.svg"

MODEL_CONFIG = dict(hidden_dim=64, n_blocks=2, coeff=6.0,
                    n_rff=1024, lengthscale=1.0, ridge=1.0)

SEED = 42
BATCH_SIZE = 256
LR = 1e-3
VAL_FRACTION = 0.20      # 20% of train_1500.csv, used only to choose the epoch count
MAX_EPOCHS = 2000
PATIENCE = 100           # stop if validation MSE has not improved for this many epochs


@torch.no_grad()
def mse(model, X, Y):
    model.eval()
    return F.mse_loss(model(X), Y).item()


def fit(X, Y, cfg, epochs, X_val=None, Y_val=None, X_test=None, Y_test=None,
        patience=None):
    """Train for up to `epochs`. If validation data is given, stop when it stops improving."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = SNGP(**cfg)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))

    hist = {"epoch": [], "train": [], "val": [], "test": []}
    best_val, bad_epochs = float("inf"), 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = F.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()

        # Log everything for this epoch first, so all lists stay the same length.
        hist["epoch"].append(epoch)
        hist["train"].append(mse(model, X, Y))
        if X_test is not None:
            hist["test"].append(mse(model, X_test, Y_test))

        stop = False
        if X_val is not None:
            val = mse(model, X_val, Y_val)
            hist["val"].append(val)
            if val < best_val:
                best_val, bad_epochs = val, 0
            else:
                bad_epochs += 1
            if patience is not None and bad_epochs >= patience:
                print(f"Early stop at epoch {epoch}: no validation improvement for {patience} epochs.")
                stop = True

        if epoch % 50 == 0 or epoch == 1:
            msg = f"Epoch {epoch} - train MSE {hist['train'][-1]:.5f}"
            if X_val is not None:
                msg += f" - val MSE {hist['val'][-1]:.5f}"
            if X_test is not None:
                msg += f" - test MSE {hist['test'][-1]:.5f}"
            print(msg)

        if stop:
            break

    return model, hist

def plot_curves(hist, best_epoch):
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams["font.family"] = "Anthropic Serif Web"
    plt.rcParams["font.size"] = 14
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hist["epoch"], hist["train"], label="training MSE (80%)")
    ax.plot(hist["epoch"], hist["val"], label="validation MSE (20%)")
    if hist["test"]:
        ax.plot(hist["epoch"], hist["test"], label="test MSE (display only)", linestyle="--")
    ax.axvline(best_epoch, color="gray", linestyle=":", label=f"chosen epoch = {best_epoch}")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log scale)")
    ax.set_title("SNGP-RFF training curves")
    ax.grid(True, linestyle="-", alpha=0.6)
    ax.legend()
    ax.grid(visible=True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    fig.savefig(PLOT_PATH_SVG)
    plt.close(fig)
    print(f"Loss plot saved to {PLOT_PATH}")


def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print("Error: check TRAIN_PATH and TEST_PATH at the top of the file.")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    X = torch.tensor(train_df.iloc[:, :4].values, dtype=torch.float32)
    Y = torch.tensor(train_df.iloc[:, 4:].values, dtype=torch.float32)

    test_df = pd.read_csv(TEST_PATH)
    X_test = torch.tensor(test_df.iloc[:, :4].values, dtype=torch.float32)
    Y_test = torch.tensor(test_df.iloc[:, 4:].values, dtype=torch.float32)

    # 80/20 split of train_1500.csv. The validation part only chooses the epoch count.
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X), generator=g)
    n_val = int(round(VAL_FRACTION * len(X)))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    cfg = {**MODEL_CONFIG, "in_dim": X.shape[1], "out_dim": Y.shape[1]}

    # Stage 1: find the epoch count on validation.
    print("Stage 1: training on 80%, stopping on validation")
    _, hist = fit(X_tr, Y_tr, cfg, MAX_EPOCHS,
                  X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test,
                  patience=PATIENCE)
    best_idx = int(np.argmin(hist["val"]))
    best_epoch = hist["epoch"][best_idx]
    print(f"Best validation epoch: {best_epoch} (val MSE {hist['val'][best_idx]:.5f})")
    plot_curves(hist, best_epoch)

    # Stage 2: retrain on all of train_1500.csv for that many epochs.
    print(f"Stage 2: retraining on 100% for {best_epoch} epochs")
    model, _ = fit(X, Y, cfg, best_epoch)
    model.fit_laplace(X, Y)

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "chosen_epochs": best_epoch}, SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()