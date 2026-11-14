"""
training pipeline for the two-stage error-mapping architecture

Stage 1: Train primary NN on train split
Stage 2: Freeze Primary, compute E = A-P on the error-map split, train the ErrorMapper on (X_map, E)
Stage 3: Evaluate both on val/test, report RMSE/MAE/Correlations
"""

import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from rich.console import Console

from assisted_nn.core.models import Primary
from assisted_nn.utils.config import data_folder, model_folder, plots_folder
from assisted_nn.utils.config import accent, focus, success, another

console = Console()


class PrimaryTrainer:
    """Trains the primary model on the train split, validates on val split"""
    def __init__(self,
                 learning_rate:float=1e-3,
                 epochs:int=100,
                 device:str="cpu",
                 seed:int=41,
                 save_path:Path|None=None):
        self.epochs         =   epochs
        self.learning_rate   =   learning_rate
        self.device         =   device
        self.seed           =   seed
        self.save_path      =   save_path if save_path is not None else model_folder / "primary.pt"
        
        # determinism
        torch.manual_seed(seed=seed)
        
        
        # setting up 1. model, 2. optimizer, 3. loss
        self.model = Primary().to(device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss  = nn.MSELoss()
        self.train_loss:list=[]
        self.val_loss:list=[]
        self.best_val_loss=float("inf")
        self.console = Console()
    
    def save_model(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
    
    def run_epoch(self, loader, train:bool)->float:
        self.model.train(train)
        total_loss = 0.0
        n_samples  = 0
        
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                if train:
                    self.optimizer.zero_grad()
                
                pred = self.model(x_batch)
                loss = self.loss(pred, y_batch)
                
                if train:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * x_batch.size(0)
                n_samples  += x_batch.size(0)
        return total_loss / max(n_samples, 1)
    
    def train(self, 
              train_loader:DataLoader|None=None, 
              val_loader:DataLoader|None=None):
        
        console.print(f"\n\n[{accent}]Training Starting.\n[/]")
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.run_epoch(train_loader, train=True)
            val_loss   = self.run_epoch(val_loader, train=False)
            
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            
            # save the best one
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()
            
            # printing
            if epoch == 1 or epoch%10==0 or epoch == self.epochs:
                self.console.print(f"[{accent}]Epoch: {epoch:4d}[/] | Train Loss {train_loss:.4f} | Validation Loss {val_loss:.4f}")
        
        self.console.print("\n\n")
        self.console.print(f"[{another}][PRIMARY TRAINER][/] Best Val Loss        : {self.best_val_loss:.4f}")
        self.console.print(f"[{another}][PRIMARY TRAINER][/] Best Model saved to  : {self.save_path}")
        self.console.print("\n\n")
        
        # load the best weights back to the instance is in its best state
        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))
        
        
        return {
            "status":True,
            "content":{
                "train_loss":self.train_loss,
                "val_loss":self.val_loss,
            }
        }

    def plot(self, plots_folder:Path=plots_folder, savefig:bool=True):
        
        # imports
        import numpy as np
        import matplotlib.pyplot as plt
        
        # style setting
        plt.rcParams["font.size"] = 14
        plt.rcParams["font.family"] = "Anthropic Serif Web Text"
        plt.rcParams.update({
            'figure.figsize': (8, 4),
            'figure.dpi': 300,
            'font.size': 14,
            'axes.titlesize': 14,
            'axes.titleweight': 400,
            'axes.labelsize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 16,
        })
        
        fig, ax = plt.subplots()
        ax.set_title("Primary Models Training & Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss (MSE)")
        ax.plot(range(self.epochs), self.train_loss, label="Training Loss")
        ax.plot(range(self.epochs), self.val_loss, label="Validation Loss")
        ax.legend(loc='upper right', frameon=True)
        ax.grid(alpha=0.8)
        
        if savefig:
            plt.savefig(plots_folder / "Primary-Training-Loss.svg",bbox_inches='tight', pad_inches=0.25)
     


if __name__=="__main__":
    
    primary_trainer = PrimaryTrainer()
    
    from assisted_nn.utils.data_loader import get_dataloaders, load_and_clean_dataset, scale_normal
    from assisted_nn.utils.config import dataset_raw_file, data_folder
    
    cleaned_df = load_and_clean_dataset(dataset_raw_file)
    scaled_df  = scale_normal(cleaned_df, data_folder)
    dataloaders = get_dataloaders(scaled_df)
    
    train_dataloader = dataloaders["content"]["train"]
    val_dataloader   = dataloaders["content"]["val"]
    # now we train the primary
    training_loss = primary_trainer.train(
        train_loader=train_dataloader,
        val_loader=val_dataloader)
    plot = primary_trainer.plot()
    
