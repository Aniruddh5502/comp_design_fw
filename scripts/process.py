"""
Processing data for Ensemble Surrogate Research.
This script performs a deterministic 80/20 split and generates nested training subsets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console

con = Console()

def clean_csv(csv_filePath: Path, save_as_filename: str) -> pd.DataFrame:
    """Cleans the raw CSV by removing empty strings and dropping NaNs."""
    if not csv_filePath.exists():
        raise FileNotFoundError(f"Dataset not found at: {csv_filePath}")
    
    df = pd.read_csv(csv_filePath)
    # Convert whitespace-only strings to NaN and drop
    df = df.replace(r'^\s*$', np.nan, regex=True).dropna(axis=0, how='any')
    
    target = csv_filePath.parent / save_as_filename
    if not target.suffix:
        target = target.with_suffix('.csv')
    
    df.to_csv(target, index=False, encoding='utf-8')
    return df

def get_input_output_csv(cleaned_csvPath: Path, dataset_name: str) -> pd.DataFrame:
    """Extracts only the necessary columns for training and testing."""
    df = pd.read_csv(cleaned_csvPath, encoding='utf-8')
    # Based on existing logic: selecting specific columns [1,2,3,4,6,7,8,9,10,11]
    training_df = df.iloc[:, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]].copy()
    
    root = Path(__file__).parent.parent
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = processed_dir / dataset_name
    if not target_path.suffix:
        target_path = target_path.with_suffix('.csv')
    
    training_df.to_csv(target_path, index=False, encoding='utf-8')
    return training_df

def generate_nested_splits(full_train_df: pd.DataFrame, sizes: list[int]):
    """
    Splits data into a fixed test set and nested training sets.
    Ensures nested subsets: train_300 is a subset of train_600, etc.
    """
    root = Path(__file__).parent.parent
    processed_dir = root / "data" / "processed"
    
    # 1. Deterministic 80/20 Split
    # We use a fixed seed to ensure the test set remains constant across all experiments
    np.random.seed(42)
    mask = np.random.rand(len(full_train_df)) < 0.8
    
    train_pool = full_train_df[mask].reset_index(drop=True)
    test_set = full_train_df[~mask].reset_index(drop=True)
    
    # Save the fixed test set
    test_set.to_csv(processed_dir / "test_set.csv", index=False, encoding='utf-8')
    con.print(f"[green]Test set saved: {len(test_set)} samples[/green]")
    
    # 2. Generate Nested Training Subsets
    # To ensure nesting, we shuffle the train_pool once and then slice
    shuffled_train = train_pool.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for size in sizes:
        # Take the first 'size' rows from the shuffled pool
        subset = shuffled_train.iloc[:size]
        filename = f"train_{size}.csv"
        subset.to_csv(processed_dir / filename, index=False, encoding='utf-8')
        con.print(f"[blue]Nested train set saved: {filename} ({len(subset)} samples)[/blue]")

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    raw_input = root / "data" / "raw" / "inputs.csv"
    
    try:
        # Step 1: Clean
        con.print("[yellow]Cleaning dataset...[/yellow]")
        clean_csv(raw_input, "cleaned.csv")
        
        # Step 2: Extract IO columns
        con.print("[yellow]Extracting IO columns...[/yellow]")
        cleaned_path = root / "data" / "raw" / "cleaned.csv"
        full_dataset = get_input_output_csv(cleaned_path, "full_processed")
        
        # Step 3: Nested Splits
        con.print("[yellow]Generating nested splits...[/yellow]")
        training_sizes = [300, 600, 900, 1200, 1500]
        generate_nested_splits(full_dataset, training_sizes)
        
        con.print("[bold green]Data Pipeline Complete![/bold green]")
        
    except Exception as e:
        con.print(f"[bold red]Pipeline Failed: {e}[/bold red]")
