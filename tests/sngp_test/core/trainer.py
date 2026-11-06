import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np

class SNGPTrainer:
    """
    High-agency trainer for SNGP models.
    Implements:
    - Train/Val/Test tracking per epoch.
    - Best-model checkpointing based on Val Loss.
    - Early stopping to prevent overfitting.
    """
    def __init__(self, model, criterion, optimizer, device, patience=200, min_delta=1e-8):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": []
        }

    def _run_epoch(self, loader, phase="train"):
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        with torch.set_grad_enabled(phase == "train"):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()
                
                running_loss += loss.item() * X.size(0)
        
        return running_loss / len(loader.dataset)

    def train(self, train_loader, val_loader, epochs=100):
        print(f"Starting training... Device: {self.device}")
        
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, "train")
            val_loss = self._run_epoch(val_loader, "val")
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if epoch%10==0:
                print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.5f} | Val: {val_loss:.5f}")
            
            # Checkpoint based on Val Loss
            if val_loss < (self.best_val_loss - self.min_delta):
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.counter = 0 # Reset patience
            else:
                self.counter += 1
            
            # Early Stopping
            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {self.best_val_loss:.5f}")
                break
        
        # Restore best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
            print(f"Restored best model weights. Best Val Loss: {self.best_val_loss:.5f}")
            
        return self.history, self.best_val_loss

    def save(self, path):
        """Saves the current model state to disk."""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def get_metrics(self):
        return self.history
