import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
import json
import sys

con = Console()

class SurrogateNN(nn.Module):
    def __init__(self, input_dim=4, output_dim=6):
        super(SurrogateNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

import __main__
__main__.SurrogateNN = SurrogateNN

class SurrogatePredictor:
    def __init__(self, model_path: str):
        self.root = Path(__file__).parent.parent
        
        with open(self.root / "config" / "design_space.json", "r") as f:
            self.design_space = json.load(f)
        
        with open(self.root / "data" / "processed" / "scaling_bounds.json", "r") as f:
            self.bounds = json.load(f)
        
        self.ensemble = torch.load(self.root / model_path, weights_only=False)
        for m in self.ensemble:
            m.eval()
        
        self.cols = self.bounds["columns"]
        self.input_cols = self.cols[:4] 
        self.mins = np.array([self.bounds["min"][c] for c in self.input_cols])
        self.maxs = np.array([self.bounds["max"][c] for c in self.input_cols])
        self.ranges = self.maxs - self.mins

    def _check_range(self, x):
        x = np.atleast_1d(x)
        if x.ndim == 1:
            for i, val in enumerate(x):
                col_name = self.input_cols[i]
                limit = self.design_space['input_parameters'][col_name]
                if val < limit["min"] or val > limit["max"]:
                    return False, col_name
            return True, None
        return True, None

    def predict(self, x_physical):
        x_physical = np.array(x_physical, dtype=np.float32)
        
        # Handle both single point and batch
        is_single = (x_physical.ndim == 1)
        if is_single:
            # Add batch dimension
            x_batch = x_physical[np.newaxis, :]
            in_bounds, failed_col = self._check_range(x_physical)
            if not in_bounds:
                raise ValueError(f"Input Out-of-Range: {failed_col}")
        else:
            x_batch = x_physical

        # Scale
        x_scaled = 2 * (x_batch - self.mins) / self.ranges - 1
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            # [20, batch, 6]
            preds_scaled = torch.stack([m(x_tensor) for m in self.ensemble]).numpy()

        # De-scale
        out_cols = self.cols[4:]
        out_mins = np.array([self.bounds["min"][c] for c in out_cols])
        out_ranges = np.array([self.bounds["max"][c] - self.bounds["min"][c] for c in out_cols])
        
        # broadcasting over [20, batch, 6]
        raw_ensemble_physical = ((preds_scaled + 1) * out_ranges / 2) + out_mins
        
        if is_single:
            # Squeeze out the batch dimension [20, 1, 6] -> [20, 6]
            raw_single = np.squeeze(raw_ensemble_physical, axis=1)
            mean_preds = np.mean(raw_single, axis=0)
            std_preds = np.std(raw_single, axis=0)
            
            return {
                "original_input": x_physical.tolist(),
                "scaled_input": x_scaled[0].tolist(),
                "raw_ensemble_preds": raw_single.tolist(),
                "mean_preds": mean_preds.tolist(),
                "std_preds": std_preds.tolist(),
                "range_status": "IN_BOUNDS"
            }
        else:
            # Batch: [20, batch, 6] -> mean/std over axis 0
            mean_preds = np.mean(raw_ensemble_physical, axis=0)
            std_preds = np.std(raw_ensemble_physical, axis=0)
            return {
                "mean_preds": mean_preds.tolist(),
                "std_preds": std_preds.tolist()
            }

    def get_train_covariance(self):
        df_train = pd.read_csv(self.root / "data" / "processed" / "train_full.csv")
        X_train = df_train.iloc[:, :4].values
        return np.cov(X_train, rowvar=False)

if __name__ == "__main__":
    try:
        predictor = SurrogatePredictor("models/ensemble_1500_42.pth")
        test_point = [0.5, 15.0, 0.5, 10.0] 
        res = predictor.predict(test_point)
        con.print("[bold green]Prediction Successful![/bold green]")
        con.print(f"Mean Preds: {res['mean_preds']}")
    except Exception as e:
        con.print(f"[bold red]Test failed: {e}[/bold red]")
