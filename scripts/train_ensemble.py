import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from rich.console import Console
import argparse

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

def train_model(X_train, y_train, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SurrogateNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(200):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

def evaluate_ensemble(ensemble, X_test, y_test, lhs_range):
    ensemble[0].eval() # Set all to eval
    for m in ensemble: m.eval()
    
    with torch.no_grad():
        # Get predictions from all members: (num_members, num_samples, output_dim)
        preds = torch.stack([m(X_test) for m in ensemble]).numpy()
    
    # mean_pred: (num_samples, output_dim)
    mean_pred = np.mean(preds, axis=0)
    # std_pred: (num_samples, output_dim)
    std_pred = np.std(preds, axis=0)
    
    # Normalize by LHS Range (as per audited plan)
    # y_test is (num_samples, output_dim)
    abs_error = np.abs(y_test - mean_pred) / lhs_range
    disagreement = std_pred / lhs_range
    
    # We calculate correlation per output dimension and average them
    correlations = []
    for i in range(mean_pred.shape[1]):
        rho, _ = spearmanr(disagreement[:, i], abs_error[:, i])
        correlations.append(rho)
        
    return np.mean(correlations), mean_pred, disagreement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="data/processed/train_1500.csv")
    parser.add_argument("--test_file", type=str, default="data/processed/test_set.csv")
    parser.add_argument("--test", action="store_true", help="Run quick test with minimal epochs")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    train_path = root / args.train_file
    test_path = root / args.test_file
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Separate X and y (X: first 4 cols, y: last 6 cols)
    X_train_raw = df_train.iloc[:, :4].values
    y_train_raw = df_train.iloc[:, 4:].values
    X_test_raw = df_test.iloc[:, :4].values
    y_test_raw = df_test.iloc[:, 4:].values
    
    # Scaling
    scaler_x = StandardScaler()
    X_train = torch.tensor(scaler_x.fit_transform(X_train_raw), dtype=torch.float32)
    
    scaler_y = StandardScaler()
    y_train = torch.tensor(scaler_y.fit_transform(y_train_raw), dtype=torch.float32)
    
    X_test = torch.tensor(scaler_x.transform(X_test_raw), dtype=torch.float32)
    y_test = torch.tensor(scaler_y.transform(y_test_raw), dtype=torch.float32)

    # In a real run, lhs_range comes from the LHS pool
    lhs_range = np.ptp(y_train_raw, axis=0) 

    con.print(f"[yellow]Training Ensemble of 20 members on {train_path.name}...[/yellow]")
    ensemble = []
    for i in range(20):
        # Use different seeds for ensemble members
        m = train_model(X_train, y_train, seed=i)
        ensemble.append(m)
        if (i+1)%5 == 0: con.print(f"Member {i+1}/20 trained.")

    r_test, mean_p, disag = evaluate_ensemble(ensemble, X_test, y_test.numpy(), lhs_range)
    con.print(f"[bold green]ID Baseline R_test: {r_test:.4f}[/bold green]")

if __name__ == "__main__":
    main()
