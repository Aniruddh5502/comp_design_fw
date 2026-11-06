"""
Ensemble of vanilla NNs for FEA surrogate modeling.
Architecture: 4 -> 32 -> 32 -> 6, ReLU hidden, no activation on output.
"""
import torch
import torch.nn as nn
from typing import List


class VanillaNN(nn.Module):
    """Single vanilla NN: 4 inputs -> 32 -> 32 -> 6 outputs."""

    def __init__(self, input_dim: int = 4, hidden_dims: List[int] = None, output_dim: int = 6):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        # Output layer: NO activation
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Ensemble(nn.Module):
    """Ensemble of VanillaNN models."""

    def __init__(
        self,
        ensemble_size: int = 10,
        input_dim: int = 4,
        hidden_dims: List[int] = None,
        output_dim: int = 6,
    ):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([
            VanillaNN(input_dim, hidden_dims, output_dim)
            for _ in range(ensemble_size)
        ])

    def forward(self, x):
        """Forward pass through all ensemble members.
        Returns: [ensemble_size, batch_size, output_dim]"""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs, dim=0)

    def predict_with_uncertainty(self, x):
        """Return mean prediction and variance across ensemble."""
        with torch.no_grad():
            outputs = self.forward(x)  # [E, B, D]
            mean = outputs.mean(dim=0)      # [B, D]
            var = outputs.var(dim=0, unbiased=False)  # [B, D]
            return mean, var, outputs

    def get_individual_predictions(self, x):
        """Return all individual member predictions: [E, B, D]"""
        with torch.no_grad():
            return self.forward(x)


if __name__ == "__main__":
    # Quick test
    ensemble = Ensemble(ensemble_size=10)
    x = torch.randn(16, 4)
    mean, var, all_preds = ensemble.predict_with_uncertainty(x)
    print(f"Mean: {mean.shape}, Var: {var.shape}, All: {all_preds.shape}")