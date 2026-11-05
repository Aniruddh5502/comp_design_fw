import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from torch.nn.utils.parametrizations import spectral_norm

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
SHARPNESS = [2.0, 5.0, 20.0]          # target = sin(x)cos(y) + exp(-k r^2); larger k = sharper peak
MODEL_KINDS = ["vanilla", "sn", "sn_gain5"]
WIDTH = 64
LR = 1e-3
EPOCHS = 5000
N_TRAIN, N_VAL = 1500, 500
SILU_LIP = 1.0998                     # max |d/dx SiLU(x)|


def target(x, k):
    return torch.sin(x[:, 0]) * torch.cos(x[:, 1]) + torch.exp(-k * (x[:, 0] ** 2 + x[:, 1] ** 2))


class Gain(nn.Module):
    """Multiplies the network output by a fixed scalar, raising the Lipschitz ceiling."""
    def __init__(self, net, gain):
        super().__init__()
        self.net = net
        self.gain = gain

    def forward(self, x):
        return self.gain * self.net(x)


def make_mlp(sn):
    def lin(i, o):
        layer = nn.Linear(i, o)
        return spectral_norm(layer) if sn else layer
    return nn.Sequential(lin(2, WIDTH), nn.SiLU(), lin(WIDTH, WIDTH), nn.SiLU(), lin(WIDTH, 1))


def build(kind):
    if kind == "vanilla":
        return make_mlp(False)
    if kind == "sn":
        return make_mlp(True)
    if kind == "sn_gain5":
        return Gain(make_mlp(True), 5.0)
    raise ValueError(kind)


def lipschitz_ceiling(model):
    """Upper bound on the network's Lipschitz constant. inf if unconstrained."""
    gain = model.gain if isinstance(model, Gain) else 1.0
    core = model.net if isinstance(model, Gain) else model
    layers = [m for m in core.modules() if isinstance(m, nn.Linear)]
    if not any(hasattr(m, "parametrizations") for m in layers):
        return float("inf")
    prod = 1.0
    for m in layers:
        prod *= torch.linalg.matrix_norm(m.weight, ord=2).item()
    return gain * prod * SILU_LIP ** 2


def grid_max_grad(f, n=201):
    """Max gradient norm of f over [-1,1]^2 (a numerical Lipschitz estimate)."""
    g = torch.linspace(-1, 1, n)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1).requires_grad_(True)
    out = f(pts).reshape(-1)
    grad, = torch.autograd.grad(out.sum(), pts)
    return grad.norm(dim=1).max().item()


def make_data(seed, k):
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.uniform(-1, 1, (N_TRAIN + N_VAL, 2)), dtype=torch.float32)
    X = X[torch.tensor(rng.permutation(N_TRAIN + N_VAL))]
    with torch.no_grad():
        y = target(X, k).unsqueeze(1)
    return X[:N_TRAIN], y[:N_TRAIN], X[N_TRAIN:], y[N_TRAIN:]


def make_ood(seed, k, n=500):
    """Points outside the training box (x0 in [1, 1.5]); secondary metric."""
    rng = np.random.default_rng(seed + 1000)
    X = torch.tensor(np.stack([rng.uniform(1.0, 1.5, n), rng.uniform(-1.5, 1.5, n)], 1),
                     dtype=torch.float32)
    with torch.no_grad():
        y = target(X, k).unsqueeze(1)
    return X, y


def train(model, Xtr, ytr, Xva, yva):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = []
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            hist.append(nn.functional.mse_loss(model(Xva), yva).item())
    return np.array(hist)


# ---------- main experiment ----------
rows, curves, target_lip = [], {}, {}

for k in SHARPNESS:
    target_lip[k] = grid_max_grad(lambda p: target(p, k))
    for seed in SEEDS:
        Xtr, ytr, Xva, yva = make_data(seed, k)
        Xood, yood = make_ood(seed, k)
        for kind in MODEL_KINDS:
            torch.manual_seed(seed)
            model = build(kind)
            hist = train(model, Xtr, ytr, Xva, yva)
            model.eval()
            with torch.no_grad():
                val_mse = nn.functional.mse_loss(model(Xva), yva).item()
                ood_mse = nn.functional.mse_loss(model(Xood), yood).item()
            rows.append({
                "k": k, "seed": seed, "model": kind,
                "val_mse": val_mse, "ood_mse": ood_mse,
                "ceiling": lipschitz_ceiling(model),
                "grid_lip": grid_max_grad(lambda p: model(p).reshape(-1)),
            })
            curves[(k, kind, seed)] = hist
            print(f"k={k} seed={seed} {kind:9s} val_mse={val_mse:.5f} ood_mse={ood_mse:.5f}")

df = pd.DataFrame(rows)
df.to_csv(RESULTS / "per_seed_results.csv", index=False)


# ---------- statistics ----------
def compare(k, base, other, metric="val_mse"):
    xa = df[(df.k == k) & (df.model == base)][metric].values
    xb = df[(df.k == k) & (df.model == other)][metric].values
    _, p = stats.ttest_ind(xb, xa, equal_var=False)
    return xa.mean(), xa.std(ddof=1), xb.mean(), xb.std(ddof=1), p


def verdict(p, mean_other, mean_base):
    if p >= 0.05:
        return "no significant difference"
    return "constrained model worse" if mean_other > mean_base else "constrained model better"


# ---------- report (all numbers computed) ----------
lines = [
    "# Spectral Normalization Capacity Test (multi-seed)",
    "",
    f"Seeds: {SEEDS} | Epochs: {EPOCHS} | Adam lr: {LR} | Width: {WIDTH}",
    "Metric: final-epoch validation MSE, mean ± std over seeds. Welch t-test against vanilla.",
    "",
]
for k in SHARPNESS:
    lines.append(f"## Target sharpness k = {k}")
    lines.append("")
    lines.append(f"Target Lipschitz (grid estimate): {target_lip[k]:.3f}")
    lines.append("")
    lines.append("| Model | Val MSE (mean ± std) | Lipschitz ceiling | Grid Lipschitz | OOD MSE (mean) | p vs vanilla | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    van_mean = df[(df.k == k) & (df.model == "vanilla")].val_mse.mean()
    van_ood = df[(df.k == k) & (df.model == "vanilla")].ood_mse.mean()
    lines.append(f"| vanilla | {van_mean:.5f} ± {df[(df.k == k) & (df.model == 'vanilla')].val_mse.std(ddof=1):.5f} | none | "
                 f"{df[(df.k == k) & (df.model == 'vanilla')].grid_lip.mean():.2f} | {van_ood:.5f} | — | baseline |")
    for kind in MODEL_KINDS[1:]:
        sub = df[(df.k == k) & (df.model == kind)]
        mb, sb, mo, so, p = compare(k, "vanilla", kind)
        ceil = sub.ceiling.iloc[0]
        ceil_s = "none" if np.isinf(ceil) else f"{ceil:.2f}"
        lines.append(f"| {kind} | {mo:.5f} ± {so:.5f} | {ceil_s} | {sub.grid_lip.mean():.2f} | "
                     f"{sub.ood_mse.mean():.5f} | {p:.4f} | {verdict(p, mo, mb)} |")
    lines.append("")

report_path = RESULTS / "report.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# ---------- learning curves ----------
fig, axes = plt.subplots(1, len(SHARPNESS), figsize=(6 * len(SHARPNESS), 5), sharey=True)
for ax, k in zip(axes, SHARPNESS):
    for kind in MODEL_KINDS:
        arr = np.stack([curves[(k, kind, s)] for s in SEEDS])
        mean, std = arr.mean(0), arr.std(0)
        ax.plot(mean, label=kind)
        ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    ax.set_yscale("log")
    ax.set_title(f"k = {k} (target Lip ≈ {target_lip[k]:.2f})")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.5)
axes[0].set_ylabel("Val MSE")
axes[0].legend()
fig.savefig(RESULTS / "learning_curves.svg", bbox_inches="tight")
print(f"Saved to {RESULTS}")