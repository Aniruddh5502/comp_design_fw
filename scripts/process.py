"""
Processing data for Ensemble Surrogate Research.
This script provides modular tools for cleaning, extracting, and scaling datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
import argparse
import json

con = Console()

def clean_csv(csv_filePath: Path, save_as_filename: str) -> pd.DataFrame:
    """Cleans the raw CSV by removing empty strings and dropping NaNs."""
    if not csv_filePath.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_filePath}")
    
    df = pd.read_csv(csv_filePath)
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna(axis=0, how='any')
    
    target = csv_filePath.parent / save_as_filename
    if not target.suffix:
        target = target.with_suffix('.csv')
    
    df.to_csv(target, index=False, encoding='utf-8')
    return df

def get_input_output_csv(cleaned_csvPath: Path, dataset_name: str) -> pd.DataFrame:
    """Extracts only the necessary columns for training and testing."""
    df = pd.read_csv(cleaned_csvPath, encoding='utf-8')
    # Selecting specific columns [1,2,3,4,6,7,8,9,10,11]
    training_df = df.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]].copy()
    
    root = Path(__file__).parent.parent
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = processed_dir / dataset_name
    if not target_path.suffix:
        target_path = target_path.with_suffix('.csv')
    
    training_df.to_csv(target_path, index=False, encoding='utf-8')
    return training_df

def scale_and_split(full_dataset: pd.DataFrame, sizes: list[int]):
    """Scales data to [-1, 1] based on global bounds and generates splits."""
    root = Path(__file__).parent.parent
    processed_dir = root / "data" / "processed"
    
    # 1. Calculate Global Bounds
    df_min = full_dataset.min()
    df_max = full_dataset.max()
    
    bounds = {
        "min": df_min.to_dict(),
        "max": df_max.to_dict(),
        "columns": full_dataset.columns.tolist()
    }
    
    with open(processed_dir / "scaling_bounds.json", "w") as f:
        json.dump(bounds, f, indent=4)
    con.print("[green]Scaling bounds saved to scaling_bounds.json[/green]")
    
    # 2. Scale to [-1, 1]
    # Formula: 2 * (x - min) / (max - min) - 1
    df_scaled = 2 * (full_dataset - df_min) / (df_max - df_min) - 1
    df_scaled.to_csv(processed_dir / "full_scaled.csv", index=False, encoding='utf-8')
    
    # 3. Split (80/20)
    np.random.seed(42)
    mask = np.random.rand(len(df_scaled)) < 0.8
    
    train_pool = df_scaled[mask].reset_index(drop=True)
    test_set = df_scaled[~mask].reset_index(drop=True)
    
    test_set.to_csv(processed_dir / "test_set.csv", index=False, encoding='utf-8')
    con.print(f"[green]Test set saved: {len(test_set)} samples[/green]")
    
    # 4. Generate Nested training sets from the scaled train pool
    shuffled_train = train_pool.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for size in sizes:
        subset = shuffled_train.iloc[:size]
        filename = f"train_{size}.csv"
        subset.to_csv(processed_dir / filename, index=False, encoding='utf-8')
        con.print(f"[blue]Nested scaled train set saved: {filename} ({len(subset)} samples)[/blue]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular Data Pipeline for Surrogate Research")
    parser.add_argument("--mode", choices=["clean", "split", "all"], default="all", help="Pipeline mode to execute")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    raw_input = root / "data" / "raw" / "inputs.csv"
    
    try:
        if args.mode in ["clean", "all"]:
            con.print("[yellow]Cleaning dataset...[/yellow]")
            clean_csv(raw_input, "cleaned.csv")
        
        if args.mode in ["split", "all"]:
            con.print("[yellow]Extracting IO columns...[/yellow]")
            cleaned_path = root / "data" / "raw" / "cleaned.csv"
            full_dataset = get_input_output_csv(cleaned_path, "full_processed")
            
            con.print("[yellow]Scaling and Generating nested splits...[/yellow]")
            training_sizes = [300, 600, 900, 1200, 1500]
            scale_and_split(full_dataset, training_sizes)
            
        con.print("[bold green]Process Complete![/bold green]")
    except Exception as e:
        con.print(f"[bold red]Pipeline Failed: {e}[/bold red]")
