"""
SNGP for regression, following Liu et al. (NeurIPS 2020),
"Simple and Principled Uncertainty Estimation with Deterministic Deep Learning
via Distance Awareness".

1. Feature extractor: dense input layer, then residual blocks h <- h + ReLU(SN(W h + b)).
   Every dense layer is spectrally normalised with coefficient c, so the Lipschitz
   constant of each layer is bounded by c. The residual connection keeps the map
   bi-Lipschitz, so distances in input space are preserved in feature space.
   No LayerNorm: it would remove the distance information that the GP variance uses.

2. GP output layer: random Fourier features approximating an RBF kernel,
       phi(h) = sqrt(2 / D) * cos(W h + b),   W ~ N(0, I / l^2),  b ~ U[0, 2 pi]
   W and b are fixed (not trained). The mean is phi(h) beta + bias, with beta trained.

3. Laplace approximation (after training, on the training set): the posterior over beta
   is N(beta_hat, Lambda^{-1}) with
       Lambda = ridge * I + Phi^T Phi / sigma_n^2
   and the latent predictive variance at h is
       var(h) = phi(h)^T Lambda^{-1} phi(h).
   The variance is large where the features are not covered by the training data.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralNormLinear(nn.Module):
    """Dense layer with spectral norm at most `coeff`.

    Power iteration (one step per training forward pass), as in Miyato et al. (2018).
    The weight is rescaled by min(1, coeff / sigma), where sigma is the spectral norm.
    """

    def __init__(self, in_features, out_features, coeff=6.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.coeff = coeff
        self.register_buffer("u", F.normalize(torch.randn(out_features), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(in_features), dim=0))

    def forward(self, x):
        W = self.weight
        if self.training:
            with torch.no_grad():
                self.v.copy_(F.normalize(W.t() @ self.u, dim=0))
                self.u.copy_(F.normalize(W @ self.v, dim=0))
        sigma = self.u @ W @ self.v
        W_sn = W * torch.clamp(self.coeff / sigma, max=1.0)
        return F.linear(x, W_sn, self.bias)


class ResidualSNBlock(nn.Module):
    """h <- h + ReLU(SN(W h + b))."""

    def __init__(self, dim, coeff):
        super().__init__()
        self.dense = SpectralNormLinear(dim, dim, coeff)

    def forward(self, h):
        return h + F.relu(self.dense(h))


class FeatureExtractor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_blocks, coeff):
        super().__init__()
        self.input = SpectralNormLinear(in_dim, hidden_dim, coeff)
        self.blocks = nn.ModuleList(
            [ResidualSNBlock(hidden_dim, coeff) for _ in range(n_blocks)])

    def forward(self, x):
        h = F.relu(self.input(x))
        for block in self.blocks:
            h = block(h)
        return h


class RandomFeatureGP(nn.Module):
    """Random Fourier features with a trained linear output layer and a Laplace posterior."""

    def __init__(self, feat_dim, n_rff, out_dim, lengthscale=1.0, ridge=1.0):
        super().__init__()
        self.n_rff = n_rff
        self.ridge = ridge
        # Fixed random features (buffers, not trained).
        self.register_buffer("W", torch.randn(n_rff, feat_dim) / lengthscale)
        self.register_buffer("b", 2.0 * math.pi * torch.rand(n_rff))
        # Trained output layer.
        self.beta = nn.Parameter(torch.zeros(n_rff, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        # Laplace state, filled by fit_laplace and saved with the checkpoint.
        self.register_buffer("cov", torch.zeros(n_rff, n_rff))
        self.register_buffer("noise_var", torch.tensor(1.0))

    def features(self, h):
        return math.sqrt(2.0 / self.n_rff) * torch.cos(F.linear(h, self.W, self.b))

    def forward(self, h):
        return self.features(h) @ self.beta + self.bias

    @torch.no_grad()
    def fit_laplace(self, h, y):
        phi = self.features(h)
        resid = y - (phi @ self.beta + self.bias)
        # Shared Gaussian noise variance, estimated from training residuals.
        self.noise_var.fill_(resid.pow(2).mean().clamp_min(1e-8))
        eye = torch.eye(self.n_rff, dtype=phi.dtype, device=phi.device)
        precision = self.ridge * eye + phi.t() @ phi / self.noise_var
        chol = torch.linalg.cholesky(precision)
        self.cov.copy_(torch.cholesky_inverse(chol))

    @torch.no_grad()
    def latent_variance(self, h):
        """Variance of the latent function value (excludes observation noise)."""
        phi = self.features(h)
        return ((phi @ self.cov) * phi).sum(dim=1)


class SNGP(nn.Module):
    def __init__(self, in_dim=4, out_dim=6, hidden_dim=64, n_blocks=2,
                 coeff=6.0, n_rff=1024, lengthscale=1.0, ridge=1.0):
        super().__init__()
        self.extractor = FeatureExtractor(in_dim, hidden_dim, n_blocks, coeff)
        self.gp = RandomFeatureGP(hidden_dim, n_rff, out_dim, lengthscale, ridge)

    def forward(self, x):
        return self.gp(self.extractor(x))

    @torch.no_grad()
    def fit_laplace(self, X, Y):
        """Compute the posterior precision on the full training set. Call after training."""
        self.eval()
        self.gp.fit_laplace(self.extractor(X), Y)

    @torch.no_grad()
    def predict(self, x):
        """Returns (mean, latent variance), each of shape (N, out_dim)."""
        self.eval()
        h = self.extractor(x)
        mean = self.gp(h)
        var = self.gp.latent_variance(h)
        return mean, var.unsqueeze(1).expand_as(mean)