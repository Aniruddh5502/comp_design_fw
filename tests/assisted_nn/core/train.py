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
                 learning_rate:float=1e-4,
                 epochs:int=4,
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
        ax.plot(range(self.epochs), self.train_loss, color='#56B4E9', label="Training Loss", marker='o', markersize=15, markerfacecolor='#56B4E9', markeredgecolor='black')
        ax.plot(range(self.epochs), self.val_loss , color='#009E73', label="Validation Loss", marker='o', markersize=15, markerfacecolor='#009E73', markeredgecolor='black')
        ax.legend(loc='upper right', frameon=True)
        ax.grid(alpha=0.8)
        
        if savefig:
            plt.savefig(plots_folder / "Primary-Training-Loss.svg",bbox_inches='tight', pad_inches=0.25)

    def train(self, 
              train_loader:DataLoader|None=None, 
              val_loader:DataLoader|None=None, 
              plot:bool=True):
        
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
        
        if plot:
            self.plot()
        
        # --- per-target error report on train + val ---
        console.print(f"\n[{accent}]Per-target error report (best weights)[/]\n")
        TARGETS = ["max_stress_von_mises", "max_deformation", "mode_1_freq", "mode_2_freq", "mode_3_freq", "mode_4_freq",]
        self.evaluate(train_loader, TARGETS, split_name="Train")
        self.evaluate(val_loader,   TARGETS, split_name="Val  ")
        return {
            "status":True,
            "content":{
                "train_loss":self.train_loss,
                "val_loss":self.val_loss,
            }
        }

    @torch.no_grad()
    def evaluate(self, loader:DataLoader, target_names: list[str], split_name: str = "", model:Primary|None=None) -> dict:
        """Compute per-target MAE, RMSE, nMAE on a dataloader."""
        import numpy as np
        import pandas as pd
        if model is not None:
            net = model
        else:
            net = self.model
        
        net.eval()
        preds_all, true_all = [], []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            preds_all.append(net(x_batch).cpu().numpy())
            true_all.append(y_batch.numpy())

        preds = np.concatenate(preds_all, axis=0)
        y_true = np.concatenate(true_all, axis=0)

        err = preds - y_true
        abs_err = np.abs(err)

        stats = {
            "MAE":     abs_err.mean(axis=0),
            "RMSE":    np.sqrt((err ** 2).mean(axis=0)),
            "MeanErr": err.mean(axis=0),
            "nMAE":    abs_err.mean(axis=0) / (y_true.std(axis=0) + 1e-12),
        }

        df = pd.DataFrame(stats, index=target_names).T

        # baseline: always-predict-mean
        baseline_mae = np.abs(y_true - y_true.mean(axis=0)).mean(axis=0)
        df.loc["mean_baseline_MAE"] = baseline_mae
        df.loc["model_MAE"]         = df.loc["MAE"]
        df.loc["beats_baseline"]    = df.loc["MAE"] < baseline_mae

        # pretty print
        from rich.table import Table
        table = Table(title=f"[{accent}]{split_name} Error Report[/]")
        table.add_column("Metric", justify="left")
        for t in target_names:
            table.add_column(t, justify="right")

        for metric in df.index:
            row = [metric] + [f"{v:.4f}" for v in df.loc[metric].values]
            table.add_row(*row)

        self.console.print(table)

        return {"preds": preds, "true": y_true, "df": df}
     


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
    primary_trainer.train(train_loader=train_dataloader, val_loader=val_dataloader, plot=True)
    primary_trainer.evaluate(loader=val_dataloader, target_names=["max_stress_von_mises", "max_deformation", "mode_1_freq", "mode_2_freq", "mode_3_freq", "mode_4_freq",])
