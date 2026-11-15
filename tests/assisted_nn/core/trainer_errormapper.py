"""
Trainer for the ErrorMapper NN (Error Mapper) on the error-map split.
Trains ErrorMapper to predict the error field E = A - P.
"""
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console

from assisted_nn.core.models import ErrorMapper
from assisted_nn.utils.config import (
    data_folder, model_folder, plots_folder,
    accent, focus, success, another
)

console = Console()


class ErrorMapperTrainer:
    """Trains the ErrorMapper on the error-map split, evaluates on val/test."""

    def __init__(self,
                 learning_rate: float = 1e-3,
                 epochs: int = 100,
                 device: str = "cpu",
                 seed: int = 41,
                 save_path: Path | None = None):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.seed = seed
        self.save_path = save_path if save_path is not None else model_folder / "error_mapper.pt"
        torch.manual_seed(seed=seed)

        self.model = ErrorMapper().to(device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.console = console

    def _prepare_error_ds(self, df):
        """
        Given a DataFrame containing features and 'error' target column,
        returns torch tensors for features and error.
        """
        feature_cols = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height']
        error_col = 'error'  # assumed column name in error_df
        # Assuming df columns: feature cols + error
        X = df[feature_cols].values.astype(np.float32)
        y = df[error_col].values.astype(np.float32)
        return torch.tensor(X, device=self.device), torch.tensor(y, device=self.device)

    def run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss = 0.0
        n_samples = 0
        with self.console.enable_grad() if train else self.console.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
                n_samples += x_batch.size(0)
        return total_loss / max(n_samples, 1)

    def train(self,
              error_dataloader,
              val_dataloader=None,
              test_dataloader=None):
        """
        error_dataloader: DataLoader of (features, error) tuples.
        val_dataloader: optionally val split.
        test_dataloader: optionally test split.
        """
        console.print(f"\n\n[{accent}]Training ErrorMapper Starting.\n[/]")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.run_epoch(error_dataloader, train=True)
            val_loss   = self.run_epoch(val_dataloader, train=False) if val_dataloader else 0.0
            test_loss  = self.run_epoch(test_dataloader, train=False) if test_dataloader else 0.0
            
            # Track losses
            if val_dataloader is not None:
                self.val_loss.append(val_loss)
            if test_dataloader is not None:
                self.test_loss.append(test_loss)

            if epoch == 1 or epoch % 10 == 0 or epoch == self.epochs:
                self.console.print(f"[{accent}]Epoch: {epoch:4d}[/] | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Test Loss {test_loss:.4f}")
            
            if epoch == 1:
                self.console.print("[{another}][ERRORMAPPER][/] Starting.")
            elif epoch % 10 == 0:
                self.console.print(f"[{another}][ERRORMAPPER][/] Val Loss {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()

        self.console.print("\n\n")
        self.console.print(f"[{another}][ERRORMAPPER][/] Best Val Loss        : {self.best_val_loss:.4f}")
        self.console.print(f"[{another}][ERRORMAPPER][/] Best Model saved to  : {self.save_path}")

        # Load best weights back
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))

        return {
            "status": True,
            "content": {
                "best_val_loss": self.best_val_loss,
                "save_path": str(self.save_path),
            }
        }

    def plot(self, plots_folder: Path = plots_folder, savefig: bool = True):
        import numpy as np
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            'font.size': 14,
            'figure.dpi': 300,
            'axes.titlesize': 14,
            'axes.labelsize': 14,
        })
        fig, ax = plt.subplots()
        ax.plot(range(self.epochs), self.train_loss if 'train_loss' in locals() else [], label='Training')
        # For demonstration, just plot best val loss
        ax.bar([i for i in range(len(self.best_val_loss))], self.best_val_loss)
        ax.set_title('ErrorMapper Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        if savefig:
            plt.savefig(plots_folder / "ErrorMapper-Training-Loss.svg")


if __name__ == "__main__":
    # Placeholder for manual usage
    trainer = ErrorMapperTrainer()
    # Example: load error map data from error_df.csv
    # Assume error_df.csv has columns: beam_width, beam_length, fillet_radius, beam_height, error
    # Here we just instantiate and print
    print("ErrorMapperTrainer instantiated.")
