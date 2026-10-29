import torch
import pytest
import numpy as np
from baseline.code.sngp import SNGP, SpectralNormLinear


def test_sngp_output_shape():
    """Verify output dimensions for single and batch inputs."""
    model = SNGP(in_dim=4, out_dim=6)
    model.eval()
    x_single = torch.randn(1, 4)
    out_single = model(x_single)
    assert out_single.shape == (1, 6)
    x_batch = torch.randn(10, 4)
    out_batch = model(x_batch)
    assert out_batch.shape == (10, 6)


def test_predict_returns_mean_and_var():
    model = SNGP(in_dim=4, out_dim=6)
    model.eval()
    X = torch.randn(5, 4)
    Y = torch.randn(5, 6)
    model.fit_laplace(X, Y)
    mean, var = model.predict(torch.randn(3, 4))
    assert mean.shape == (3, 6)
    assert var.shape == (3, 6)
    assert torch.all(var >= 0)


def test_spectral_norm_applied():
    """Verify SpectralNormLinear layers are present."""
    model = SNGP(in_dim=4, out_dim=6)
    sn_layers = [m for m in model.modules() if isinstance(m, SpectralNormLinear)]
    assert len(sn_layers) > 0, "No SpectralNormLinear layers found"


def test_distance_awareness_correlation():
    """Relative distances preserved in latent space."""
    model = SNGP(in_dim=4, out_dim=6)
    model.eval()
    x1 = torch.randn(1, 4)
    x2 = x1 + torch.randn(1, 4) * 0.01
    x3 = x1 + torch.randn(1, 4) * 10.0
    with torch.no_grad():
        z1 = model.extractor(x1)
        z2 = model.extractor(x2)
        z3 = model.extractor(x3)
        dist_12 = torch.dist(z1, z2)
        dist_13 = torch.dist(z1, z3)
    assert dist_13 > dist_12, "Spectral norm should preserve relative distances"


def test_fit_laplace_sets_cov():
    model = SNGP(in_dim=4, out_dim=6)
    model.eval()
    X = torch.randn(20, 4)
    Y = torch.randn(20, 6)
    model.fit_laplace(X, Y)
    assert model.gp.cov.shape == (model.gp.n_rff, model.gp.n_rff)
    nz = model.gp.noise_var.item()
    assert nz > 0 and np.isfinite(nz)


def test_eval_mode_preservation():
    model = SNGP(in_dim=4, out_dim=6)
    model.train()
    X = torch.randn(5, 4)
    Y = torch.randn(5, 6)
    model.fit_laplace(X, Y)
    _ = model.predict(torch.randn(1, 4))
    assert model.training is False