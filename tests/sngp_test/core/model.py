import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm
from torch.nn.utils import parametrize


class _ScaleBy(nn.Module):
    """Parametrization that multiplies a weight by a fixed coefficient c."""
    def __init__(self, c):
        super().__init__()
        self.c = c

    def forward(self, W):
        return self.c * W


def sn_linear(in_features, out_features, coeff):
    """Linear layer with spectral norm fixed to `coeff`."""
    layer = spectral_norm(nn.Linear(in_features, out_features))
    parametrize.register_parametrization(layer, "weight", _ScaleBy(coeff))
    return layer


def _as_per_output(value, output_dim, dtype=torch.float32):
    """Scalar or list -> per-output tensor of shape [output_dim]."""
    if isinstance(value, (int, float)):
        return torch.full((output_dim,), float(value), dtype=dtype)
    t = torch.as_tensor(value, dtype=dtype)
    if t.dim() == 0:
        t = t.expand(output_dim)
    return t.clone().contiguous()


class SNGPSurrogate(nn.Module):
    """
    SNGP with PER-OUTPUT kernels.

    Key changes vs. the original:
      * One random Fourier feature map per output  ->  each output has its own
        posterior variance field (fixes the shared `Sigma_0 == ... == Sigma_5` issue).
      * Per-output lengthscales.
      * Per-output noise variance s_j^2.
      * Per-output mean heads (mean_W of shape [O, D], mean_b of shape [O]).
    """
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 6,
                 hidden_dim: int = 128,
                 rff_dim: int = 1024,
                 noise_var=1e-4,          # scalar OR list of length output_dim
                 prior_prec: float = 1.0,
                 lipschitz_constant: float = 1.0,
                 lengthscale=1.0):        # scalar OR list of length output_dim
        super().__init__()

        self.output_dim = output_dim
        self.rff_dim = rff_dim
        self.prior_prec = prior_prec

        # Shared backbone
        self.feature_extractor = nn.Sequential(
            sn_linear(input_dim, hidden_dim, lipschitz_constant),
            nn.ReLU(),
            sn_linear(hidden_dim, hidden_dim, lipschitz_constant),
            nn.ReLU(),
            sn_linear(hidden_dim, hidden_dim, lipschitz_constant),
            nn.ReLU(),
        )

        # ---- Per-output lengthscales ----
        ls = _as_per_output(lengthscale, output_dim)                # [O]
        self.register_buffer("lengthscales", ls)

        # ---- Per-output random Fourier features ----
        # W[j] ~ N(0, I) / ls[j]; shape [O, rff_dim, hidden_dim]
        W = torch.randn(output_dim, rff_dim, hidden_dim)
        W = W / ls.view(output_dim, 1, 1)
        self.register_buffer("W", W)

        b = torch.rand(output_dim, rff_dim) * 2 * math.pi           # [O, rff_dim]
        self.register_buffer("b", b)

        # ---- Per-output observation noise s_j^2 ----
        self.register_buffer("noise_var", _as_per_output(noise_var, output_dim))

        # ---- Per-output posterior covariance Sigma_j ----
        self.register_buffer(
            "Sigmas",
            torch.zeros(output_dim, rff_dim, rff_dim, dtype=torch.float64),
        )
        self.fitted = False

        # ---- Per-output mean heads:  mean_o(x) = psi_o(x)^T w_o + b_o ----
        bound = 1.0 / math.sqrt(rff_dim)
        self.mean_W = nn.Parameter(
            torch.empty(output_dim, rff_dim).uniform_(-bound, bound)
        )
        self.mean_b = nn.Parameter(
            torch.empty(output_dim).uniform_(-bound, bound)
        )

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def features(self, X):
        return self.feature_extractor(X)

    def rff(self, h):
        """h: [B, hidden_dim]  ->  psi: [B, O, rff_dim]"""
        proj = torch.einsum("bh,ojh->boj", h, self.W) + self.b.unsqueeze(0)
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(proj)

    def mean_from_psi(self, psi):
        """psi: [B, O, rff_dim]  ->  mean: [B, O]"""
        return torch.einsum("bod,od->bo", psi, self.mean_W) + self.mean_b

    def forward(self, X):
        psi = self.rff(self.features(X))
        return self.mean_from_psi(psi)

    def forward_var(self, X):
        """Differentiable (mean, var) — used by the NLL refinement stage."""
        psi = self.rff(self.features(X))
        mean = self.mean_from_psi(psi)
        psi64 = psi.double()
        Sigmas = self.Sigmas.to(psi64.device)
        var = torch.einsum("bod,ode,boe->bo", psi64, Sigmas, psi64)
        var = var + self.noise_var.double().unsqueeze(0)
        return mean, var.float()

    # ------------------------------------------------------------------ #
    #  GP posterior fitting
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fit_gp(self, loader, device):
        """
        Posterior precision per output j:
            Lambda_j = Phi_j^T Phi_j / s_j^2 + tau I
        We compute Sigma_j = s_j^2 (Phi_j^T Phi_j + tau s_j^2 I)^(-1) in float64.
        """
        self.eval()
        Phi_list = []
        for X, _ in loader:
            h = self.feature_extractor(X.to(device))
            psi = self.rff(h)                                        # [B, O, D]
            Phi_list.append(psi.detach().cpu().double())
        Phi = torch.cat(Phi_list, dim=0)                             # [N, O, D]

        D = self.rff_dim
        eye = torch.eye(D, dtype=torch.float64)

        for j in range(self.output_dim):
            Phi_j = Phi[:, j, :]                                     # [N, D]
            PhiTPhi = Phi_j.T @ Phi_j
            s2 = float(self.noise_var[j].item())
            A = PhiTPhi + (self.prior_prec * s2) * eye
            L = torch.linalg.cholesky(A)
            A_inv = torch.cholesky_inverse(L)
            self.Sigmas[j].copy_(s2 * A_inv)

        self.fitted = True

    # ------------------------------------------------------------------ #
    #  Prediction
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, X, device):
        assert self.fitted, "Must call fit_gp() first"
        self.eval()
        return self.forward_var(X.to(device))

    @torch.no_grad()
    def calibrate_noise(self, loader, device, factor=1.0):
        """
        Set per-output noise_var from residual variance on `loader`.
        Fixes the RMSE/Std ratio (calibration) but does NOT change the
        rank correlation — that depends on the spatial shape of Sigma.
        """
        self.eval()
        preds, ys = [], []
        for X, y in loader:
            X = X.to(device)
            psi = self.rff(self.feature_extractor(X))
            mean = self.mean_from_psi(psi)
            preds.append(mean.cpu())
            ys.append(y.cpu())
        pred = torch.cat(preds, 0)
        y = torch.cat(ys, 0)
        self.noise_var.copy_(((y - pred) ** 2).mean(dim=0) * factor)


# ---------------------------------------------------------------------- #
#  Helpers (kept compatible with the old API)
# ---------------------------------------------------------------------- #
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path, input_dim=4, output_dim=6, lipschitz_constant=1.0,
               lengthscale=1.0, noise_var=1e-4, prior_prec=1.0):
    device = get_device()
    model = SNGPSurrogate(
        input_dim=input_dim, output_dim=output_dim, hidden_dim=128,
        lipschitz_constant=lipschitz_constant, lengthscale=lengthscale,
        noise_var=noise_var, prior_prec=prior_prec,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device


def setup_model(input_dim=4, output_dim=6, lipschitz_constant=1.0, lengthscale=1.0,
                noise_var=1e-4, prior_prec=1.0, num_samples=1000):
    device = get_device()
    model = SNGPSurrogate(
        input_dim=input_dim, output_dim=output_dim, hidden_dim=128,
        lipschitz_constant=lipschitz_constant, lengthscale=lengthscale,
        noise_var=noise_var, prior_prec=prior_prec,
    ).to(device)

    criterion = nn.MSELoss()
    mean_nv = float(model.noise_var.mean().item())
    head_weight_decay = (prior_prec * mean_nv) / num_samples

    optimizer = optim.Adam([
        {"params": model.feature_extractor.parameters(), "weight_decay": 0.0},
        {"params": [model.mean_W, model.mean_b],         "weight_decay": head_weight_decay},
    ], lr=0.001)
    return model, criterion, optimizer, device