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
    """
    Linear layer with spectral norm fixed to `coeff`:
        W_eff = coeff * W / sigma_max(W)   =>   ||W_eff||_2 = coeff
    With 1-Lipschitz ReLUs, each block is coeff-Lipschitz.
    """
    layer = spectral_norm(nn.Linear(in_features, out_features))
    parametrize.register_parametrization(layer, "weight", _ScaleBy(coeff))
    return layer


class SNGPSurrogate(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 6,
                 hidden_dim: int = 128,
                 rff_dim: int = 1024,
                 noise_var: float = 1e-4,
                 prior_prec: float = 1.0,
                 lipschitz_constant: float = 1.0,   # spectral-norm coefficient per layer
                 lengthscale: float = 1.0):
        super().__init__()

        self.output_dim = output_dim
        self.rff_dim = rff_dim
        self.prior_prec = prior_prec

        self.feature_extractor = nn.Sequential(
            sn_linear(input_dim, hidden_dim, lipschitz_constant),
            nn.ReLU(),
            sn_linear(hidden_dim, hidden_dim, lipschitz_constant),
            nn.ReLU(),
            sn_linear(hidden_dim, hidden_dim, lipschitz_constant),
            nn.ReLU(),
        )

        # Per-output observation noise s_j^2, stored as a buffer so it follows the model's device
        self.register_buffer(
            "noise_var",
            torch.as_tensor(noise_var, dtype=torch.float32).expand(output_dim).clone(),
        )

        # Fixed random Fourier features (not learned, as in SNGP)
        self.register_buffer("W", torch.randn(rff_dim, hidden_dim) / lengthscale)
        self.register_buffer("b", torch.rand(rff_dim) * 2 * math.pi)

        # Posterior covariance per output, kept in float64 for numerical accuracy
        self.register_buffer("Sigmas", torch.zeros(output_dim, rff_dim, rff_dim, dtype=torch.float64))
        self.fitted = False

        self.mean_head = nn.Linear(rff_dim, output_dim)

    def rff(self, h):
        return math.sqrt(2.0 / self.rff_dim) * torch.cos(h @ self.W.T + self.b)

    def forward(self, X):
        psi = self.rff(self.feature_extractor(X))
        return self.mean_head(psi)

    @torch.no_grad()
    def fit_gp(self, loader, device):
        """
        Posterior precision per output j:  Lambda_j = Phi^T Phi / s_j^2 + tau I
        Its inverse, in a better-conditioned form:
            Sigma_j = s_j^2 (Phi^T Phi + tau s_j^2 I)^(-1)
        Computed in float64 on CPU using a Cholesky factorization.
        """
        self.eval()
        Phi_list = []
        for X, _ in loader:
            psi = self.rff(self.feature_extractor(X.to(device)))
            Phi_list.append(psi.detach().cpu().double())
        Phi = torch.cat(Phi_list)                      # [N, D], float64

        D = self.rff_dim
        PhiTPhi = Phi.T @ Phi                          # [D, D], float64
        eye = torch.eye(D, dtype=torch.float64)

        for j in range(self.output_dim):
            s2 = float(self.noise_var[j].item())
            A = PhiTPhi + (self.prior_prec * s2) * eye
            L = torch.linalg.cholesky(A)
            A_inv = torch.cholesky_inverse(L)
            self.Sigmas[j].copy_(s2 * A_inv)

        self.fitted = True

    @torch.no_grad()
    def predict(self, X, device):
        assert self.fitted, "Must call fit_gp() first"
        self.eval()
        X = X.to(device)
        psi = self.rff(self.feature_extractor(X))      # [B, D], float32
        mean = self.mean_head(psi)

        # var_j = psi^T Sigma_j psi + s_j^2, computed in float64
        psi64 = psi.double()
        Sigmas = self.Sigmas.to(psi64.device)
        var = torch.einsum("bd,ode,be->bo", psi64, Sigmas, psi64)
        var = var + self.noise_var.double().to(psi64.device)
        return mean, var.float()


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(path, input_dim=4, output_dim=6, lipschitz_constant=1.0,
               lengthscale=1.0, noise_var=1e-4, prior_prec=1.0):
    device = get_device()
    # lipschitz_constant must match the value used in training
    model = SNGPSurrogate(input_dim=input_dim, output_dim=output_dim,
                          hidden_dim=128,
                          lipschitz_constant=lipschitz_constant,
                          lengthscale=lengthscale,
                          noise_var=noise_var,
                          prior_prec=prior_prec).to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def setup_model(input_dim=4, output_dim=6, lipschitz_constant=1.0, lengthscale=1.0,
                noise_var=1e-4, prior_prec=1.0, num_samples=1000):
    device = get_device()
    model = SNGPSurrogate(input_dim=input_dim, output_dim=output_dim,
                          hidden_dim=128,
                          lipschitz_constant=lipschitz_constant,
                          lengthscale=lengthscale,
                          noise_var=noise_var,
                          prior_prec=prior_prec).to(device)
    criterion = nn.MSELoss()

    # MAP weight decay on the head: lambda = (tau * sigma^2) / N
    head_weight_decay = (prior_prec * noise_var) / num_samples

    optimizer_params = [
        {"params": model.feature_extractor.parameters(), "weight_decay": 0.0},
        {"params": model.mean_head.parameters(), "weight_decay": head_weight_decay},
    ]
    optimizer = optim.Adam(optimizer_params, lr=0.001)
    return model, criterion, optimizer, device