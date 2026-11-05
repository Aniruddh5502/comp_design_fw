import torch
import numpy as np
from scipy import stats
from pathlib import Path
from sngp_test.core.model import load_model
from sngp_test.utils.data_cleaning import load_and_clean_data

# Path Resolution
# evaluate.py is in new_sota/tests/hybrid_arch/scripts/
SCRIPT_DIR = Path(__file__).parent
# root is 4 levels up from scripts/ (scripts -> hybrid_arch -> tests -> new_sota -> root)
# Wait, according to the previous read:
# C:\Users\Administrator\Desktop\code\comp_design_fw\new_sota\tests\hybrid_arch\scripts\evaluate.py
# .parent (scripts)
# .parent (hybrid_arch)
# .parent (tests)
# .parent (new_sota)
# .parent (root)
ROOT = SCRIPT_DIR.parent.parent.parent.parent

# Absolute paths for consistency
DEFAULT_MODEL_PATH = SCRIPT_DIR.parent / "model" / "sngp_best.pth"
DEFAULT_CSV_PATH = ROOT / "data" / "raw" / "inputs.csv"

def predict_single(input_vector, model_path=DEFAULT_MODEL_PATH, csv_path=DEFAULT_CSV_PATH):
    """
    Predicts mean and variance for a single input vector or a batch of vectors.
    
    Args:
        input_vector: List or numpy array of shape (dim,) or (batch, dim).
        model_path: Path to the .pth model file.
        csv_path: Path to the raw CSV for GP fitting.
    """
    # Convert Path objects to strings for the loaders
    m_path = str(model_path)
    c_path = str(csv_path)
    
    # 1. Load Model
    model, device = load_model(m_path)
    
    # 2. Fit GP (SNGP requirement: must fit on training data for variance)
    train_loader, _, _ = load_and_clean_data(csv_path=c_path, batch_size=32)
    model.fit_gp(train_loader, device)
    
    # 3. Prepare Input
    X = torch.tensor(input_vector, dtype=torch.float32).to(device)
    if X.ndim == 1:
        X = X.unsqueeze(0) # Add batch dimension [1, dim]
    
    # 4. Predict
    model.eval()
    with torch.no_grad():
        mean, var = model.predict(X, device)
    
    return mean.cpu().numpy(), var.cpu().numpy()

def evaluate_output(idx, targets, means, vars):
    """Computes metrics for a single output dimension."""
    error = targets[:, idx] - means[:, idx]
    abs_error = np.abs(error)
    variance = vars[:, idx]
    
    rmse = np.sqrt(np.mean(error**2))
    avg_std = np.sqrt(np.mean(variance))
    
    # Pearson: Linear correlation
    pearson_r, _ = stats.pearsonr(abs_error, variance)
    # Spearman: Monotonic correlation (critical for uncertainty/error relationship)
    spearman_r, _ = stats.spearmanr(abs_error, variance)
    
    return {
        "rmse": rmse,
        "avg_std": avg_std,
        "ratio": rmse / avg_std if avg_std != 0 else 0,
        "pearson": pearson_r,
        "spearman": spearman_r
    }

def main():
    # 1. Load Model
    try:
        model, device = load_model(str(DEFAULT_MODEL_PATH))
        print(f"Model loaded from {DEFAULT_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Load Data (Single call to get all loaders)
    try:
        train_loader, _, test_loader = load_and_clean_data(csv_path=str(DEFAULT_CSV_PATH), batch_size=32)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return

    # 3. Fit GP (Mandatory for variance calculation)
    model.fit_gp(train_loader, device)
    print("GP posterior fitted on training set.")

    # 4. Evaluation Loop
    model.eval()
    means, vars, targets = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            m, v = model.predict(X, device)
            means.append(m.cpu())
            vars.append(v.cpu())
            targets.append(y)

    # Concatenate all batches
    means = torch.cat(means).numpy()
    vars = torch.cat(vars).numpy()
    targets = torch.cat(targets).numpy()

    # 5. Calibration Report
    print("\n" + "="*85)
    print(f"{'Output':<10} | {'RMSE':<10} | {'Avg Std':<10} | {'Ratio':<10} | {'Pearson':<10} | {'Spearman':<10}")
    print("-" * 85)

    all_spearman = []
    for i in range(targets.shape[1]):
        m = evaluate_output(i, targets, means, vars)
        all_spearman.append(m['spearman'])
        print(f"Out {i:<5} | {m['rmse']:<10.5f} | {m['avg_std']:<10.5f} | {m['ratio']:<10.3f} | {m['pearson']:<10.3f} | {m['spearman']:<10.3f}")

    print("-" * 85)
    print(f"Mean Spearman Correlation (Abs Error vs Var): {np.mean(all_spearman):.4f}")
    print("="*85)

if __name__ == "__main__":
    main()
