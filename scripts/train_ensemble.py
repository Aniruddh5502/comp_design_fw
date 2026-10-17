import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from rich.console import Console
import argparse
import json

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

def train_model(X_train, y_train, seed, epochs=100):
    g = torch.Generator()
    g.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SurrogateNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=g)
    
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_ensemble(ensemble, X_test, y_test_scaled, bounds):
    for m in ensemble: m.eval()
    
    with torch.no_grad():
        preds_scaled = torch.stack([m(X_test) for m in ensemble]).numpy()
    
    mean_pred_scaled = np.mean(preds_scaled, axis=0)
    std_pred_scaled = np.std(preds_scaled, axis=0)
    
    # Reverse scaling for the "Physical" error calculation
    # Formula: x = ((x_scaled + 1) * (max - min) / 2) + min
    cols = bounds["columns"]
    # We only care about the output columns (index 4 onwards)
    output_cols = cols[4:] 
    mins = np.array([bounds["min"][c] for c in output_cols])
    maxs = np.array([bounds["max"][c] for c in output_cols])
    ranges = maxs - mins
    
    # De-scale the target and the mean prediction
    y_test_raw = ((y_test_scaled + 1) * ranges / 2) + mins
    mean_pred_raw = ((mean_pred_scaled + 1) * ranges / 2) + mins
    
    # Disagreement in scaled space (normalized by range for consistency)
    abs_error = np.abs(y_test_raw - mean_pred_raw) / ranges
    disagreement = std_pred_scaled / (ranges / 2) # Normalize std by the half-range (the scale of the [-1, 1] window)
    
    correlations = []
    for i in range(mean_pred_raw.shape[1]):
        rho, _ = spearmanr(disagreement[:, i], abs_error[:, i])
        correlations.append(rho)
        
    return np.mean(correlations), mean_pred_raw, disagreement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True, help="Path to training CSV")
    parser.add_argument("--test_file", required=True, help="Path to test CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_metrics", type=str, help="JSON file to save result")
    parser.add_argument("--save_model", type=str, help="Path to save ensemble torch file")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    
    # Load scaling bounds
    with open(root / "data" / "processed" / "scaling_bounds.json", "r") as f:
        bounds = json.load(f)
    
    df_train = pd.read_csv(root / args.train_file)
    df_test = pd.read_csv(root / args.test_file)
    
    # Data is already scaled to [-1, 1] from process.py
    X_train = torch.tensor(df_train.iloc[:, :4].values, dtype=torch.float32)
    y_train = torch.tensor(df_train.iloc[:, 4:].values, dtype=torch.float32)
    X_test = torch.tensor(df_test.iloc[:, :4].values, dtype=torch.float32)
    y_test_scaled = torch.tensor(df_test.iloc[:, 4:].values, dtype=torch.float32)
    
    ensemble = []
    for i in range(20):
        m = train_model(X_train, y_train, seed=args.seed + i, epochs=args.epochs)
        ensemble.append(m)
    
    r_test, mean_p, disag = evaluate_ensemble(ensemble, X_test, y_test_scaled, bounds)
    
    # Stochasticity Audit
    if np.allclose(disag, 0, atol=1e-6):
        con.print("[bold red]CRITICAL FAILURE: Ensemble members are identical![/bold red]")
    else:
        con.print(f"[bold cyan]Stochasticity Audit Passed: Mean Disagreement = {np.mean(disag):.6f}[/bold cyan]")
    
    result = {"r_test": float(r_test), "train_file": args.train_file}
    
    if args.save_metrics:
        with open(root / args.save_metrics, 'w') as f:
            json.dump(result, f)
            
    if args.save_model:
        torch.save(ensemble, root / args.save_model)
        
    con.print(f"[bold green]R_test for {args.train_file}: {r_test:.4f}[/bold green]")
    return r_test

if __name__ == "__main__":
    main()
