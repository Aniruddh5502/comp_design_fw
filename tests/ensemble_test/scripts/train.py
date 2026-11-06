"""
Train ensemble vanilla NN on FEA data.
Usage: python scripts/train.py
"""
import sys
sys.path.insert(0, ".")

import torch
import pandas as pd
from utils.data_processing import prepare_data, DATA_DIR
from core.model import Ensemble
from core.trainer import EnsembleTrainer, create_dataloaders, PROJECT_ROOT


def main(
    ensemble_size: int = 10,
    lr: float = 0.001,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 50,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str = "cpu",
):
    # Prepare data
    print("Preparing data...")
    train_df, val_df, test_df, bounds = prepare_data(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, batch_size=batch_size, device=device
    )

    # Build ensemble
    ensemble = Ensemble(ensemble_size=ensemble_size)
    print(f"Ensemble: {ensemble_size} models, {sum(p.numel() for p in ensemble.parameters())} total params")

    # Train
    trainer = EnsembleTrainer(ensemble, lr=lr, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)

    # Save test set for evaluation
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print(f"Training complete. Best val loss: {min(history['val_loss']):.6f}")
    return ensemble, test_loader, bounds


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    main(device=device)