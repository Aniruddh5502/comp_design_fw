"""
Trainer for ensemble with validation-based early stopping.
Tracks train/val loss per epoch.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import os
from core.model import Ensemble

# Project root (where this repo is)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class EnsembleTrainer:
    def __init__(
        self,
        ensemble: Ensemble,
        lr: float = 0.001,
        device: str = "cpu",
    ):
        self.ensemble = ensemble.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        # Separate optimizer for each model
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=lr)
            for model in ensemble.models
        ]
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch on all ensemble members."""
        self.ensemble.train()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            batch_loss = 0.0

            for i, (model, opt) in enumerate(zip(self.ensemble.models, self.optimizers)):
                opt.zero_grad()
                pred = model(xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                opt.step()
                batch_loss += loss.item()

            total_loss += batch_loss / len(self.ensemble.models)
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        """Validate on val set."""
        self.ensemble.eval()
        total_loss = 0.0
        n_batches = 0

        for xb, yb in val_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            batch_loss = 0.0
            for model in self.ensemble.models:
                pred = model(xb)
                loss = self.criterion(pred, yb)
                batch_loss += loss.item()
            total_loss += batch_loss / len(self.ensemble.models)
            n_batches += 1

        return total_loss / n_batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 200,
        patience: int = 50,
        save_dir: str = None,
    ) -> dict:
        """Train with early stopping on val loss."""
        if save_dir is None:
            save_dir = PROJECT_ROOT / "tests/ensemble_test/checkpoints"
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        best_val = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 20 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                best_state = {
                    "epoch": epoch,
                    "model_state": {f"model_{i}": m.state_dict() for i, m in enumerate(self.ensemble.models)},
                    "val_loss": val_loss,
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} (best val: {best_val:.6f})")
                    break

        # Load best model
        if best_state:
            for i, model in enumerate(self.ensemble.models):
                model.load_state_dict(best_state["model_state"][f"model_{i}"])
            torch.save(best_state, save_path / "best_ensemble.pt")

        # Save history
        with open(save_path / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history


def create_dataloaders(train_df, val_df, test_df, batch_size: int = 32, device: str = "cpu"):
    """Create DataLoaders from dataframes."""
    INPUT_COLS = ["beam_width", "beam_length", "fillet_radius", "beam_height"]
    TARGET_COLS = [
        "max_stress_von_mises",
        "max_deformation",
        "mode_1_freq",
        "mode_2_freq",
        "mode_3_freq",
        "mode_4_freq",
    ]

    def to_tensor(df, cols):
        return torch.tensor(df[cols].values, dtype=torch.float32, device=device)

    train_ds = TensorDataset(to_tensor(train_df, INPUT_COLS), to_tensor(train_df, TARGET_COLS))
    val_ds = TensorDataset(to_tensor(val_df, INPUT_COLS), to_tensor(val_df, TARGET_COLS))
    test_ds = TensorDataset(to_tensor(test_df, INPUT_COLS), to_tensor(test_df, TARGET_COLS))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    ensemble = Ensemble(ensemble_size=3)
    trainer = EnsembleTrainer(ensemble, lr=0.001)
    print("Trainer ready")