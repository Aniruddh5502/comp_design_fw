"""
Data processing utilities for ensemble training.
Clean, scale [0,1], split 70/15/15 from raw inputs.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Project root (where this repo is)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_COLS = ["beam_width", "beam_length", "fillet_radius", "beam_height"]
TARGET_COLS = [
    "max_stress_von_mises",
    "max_deformation",
    "mode_1_freq",
    "mode_2_freq",
    "mode_3_freq",
    "mode_4_freq",
]

RAW_DATA_PATH = PROJECT_ROOT / "data/raw/inputs.csv"
DATA_DIR = PROJECT_ROOT / "tests/ensemble_test/data"


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    """Load raw CSV and drop rows with any NaN."""
    df = pd.read_csv(raw_path)
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def scale_minmax(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, dict]:
    """Scale specified columns to [0,1]. Return scaled df and bounds dict."""
    bounds = {}
    df_scaled = df.copy()
    for col in cols:
        vmin = df[col].min()
        vmax = df[col].max()
        bounds[col] = {"min": float(vmin), "max": float(vmax)}
        if vmax > vmin:
            df_scaled[col] = (df[col] - vmin) / (vmax - vmin)
        else:
            df_scaled[col] = 0.0
    return df_scaled, bounds


def split_train_val_test(df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Split dataframe into train/val/test with given ratios."""
    np.random.seed(seed)
    n = len(df)
    idx = np.random.permutation(n)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return df.iloc[train_idx].reset_index(drop=True), \
           df.iloc[val_idx].reset_index(drop=True), \
           df.iloc[test_idx].reset_index(drop=True)


def prepare_data(
    raw_path: Path = RAW_DATA_PATH,
    out_dir: Path = DATA_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """Full pipeline: load, clean, scale, split, save."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load and clean
    df = load_and_clean(raw_path)
    print(f"Loaded {len(df)} samples after cleaning")

    # Scale all columns (inputs + targets)
    all_cols = INPUT_COLS + TARGET_COLS
    df_scaled, bounds = scale_minmax(df, all_cols)

    # Save bounds
    with open(out_path / "scaling_bounds.json", "w") as f:
        json.dump(bounds, f, indent=2)

    # Split
    train_df, val_df, test_df = split_train_val_test(
        df_scaled, train_ratio, val_ratio, seed
    )
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Save splits
    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    return train_df, val_df, test_df, bounds


def load_splits(data_dir: Path = DATA_DIR):
    """Load pre-saved train/val/test splits."""
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    val_df = pd.read_csv(Path(data_dir) / "val.csv")
    test_df = pd.read_csv(Path(data_dir) / "test.csv")
    with open(Path(data_dir) / "scaling_bounds.json") as f:
        bounds = json.load(f)
    return train_df, val_df, test_df, bounds


def inverse_transform(df_scaled: pd.DataFrame, bounds: dict, cols: list) -> pd.DataFrame:
    """Convert scaled [0,1] values back to original scale."""
    df = df_scaled.copy()
    for col in cols:
        vmin = bounds[col]["min"]
        vmax = bounds[col]["max"]
        df[col] = df_scaled[col] * (vmax - vmin) + vmin
    return df


if __name__ == "__main__":
    prepare_data()