import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_and_clean_data(
    csv_path:Path|None=None, 
    batch_size=32, 
    train_split=0.7, 
    val_split=0.15, 
    test_split=0.15
):
    """
    Professional data cleaning and loading pipeline for FEM surrogate models.
    Fixes Data Leakage by fitting scales on the training set only.
    """
    # 1. Robust Root Discovery
    # Find the project root by looking for the .momo directory
    root = Path(__file__).parent.parent.parent.parent.parent

    # Resolve paths relative to the discovered project root
    scaler_dir = Path(__file__).parent.parent / "data"
    plots_dir = root / "plots"
    
    for d in [scaler_dir, plots_dir]:
        d.mkdir(exist_ok=True, parents=True)

    # Handle csv_path: if it's relative, make it relative to root
    full_dataset_path = csv_path
    scaler_json_path = scaler_dir / "scaler.json"
    
    if not full_dataset_path.exists():
        raise FileNotFoundError(f"Input CSV not found at {full_dataset_path}. Please ensure data/inputs.csv exists in the project root.")

    # 2. Cleaning
    df = pd.read_csv(full_dataset_path, encoding='utf-8')
    df = df.replace("", np.nan).dropna()

    feature_cols = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height']
    label_cols = ['max_stress_von_mises', 'max_deformation', 'mode_1_freq', 'mode_2_freq', 'mode_3_freq', 'mode_4_freq']
    all_cols = feature_cols + label_cols
    df = df[all_cols]

    # 3. Splitting FIRST (Prevents Data Leakage)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_split)
    val_end = train_end + int(n * val_split)

    train_df_raw = df[:train_end]
    val_df_raw = df[train_end:val_end]
    test_df_raw = df[val_end:]
    
    print(f"Train Data Shape    :   {train_df_raw.shape}")
    print(f"Validation Shape    :   {val_df_raw.shape}")
    print(f"Tests Data Shape    :   {test_df_raw.shape}")
    # 4. Scaling (Fit on Train only)
    content = {}
    for col in all_cols:
        c_min = float(train_df_raw[col].min())
        c_max = float(train_df_raw[col].max())
        content[col] = {"max": c_max, "min": c_min}

    with open(scaler_json_path, 'w', encoding='utf-8') as f:
        json.dump(content, f, indent=4)

    def scale_df(df_to_scale):
        df_s = df_to_scale.copy()
        for col in all_cols:
            c_min = content[col]["min"]
            c_max = content[col]["max"]
            if c_max - c_min != 0:
                df_s[col] = (df_to_scale[col] - c_min) / (c_max - c_min)
            else:
                df_s[col] = 0.0
        return df_s

    train_df = scale_df(train_df_raw)
    val_df = scale_df(val_df_raw)
    test_df = scale_df(test_df_raw)

    # 5. PyTorch Dataset Conversion
    def df_to_loader(df, shuffle=False):
        X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        Y = torch.tensor(df[label_cols].values, dtype=torch.float32)
        ds = TensorDataset(X, Y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return df_to_loader(train_df, shuffle=True), df_to_loader(val_df), df_to_loader(test_df)

def generate_synthetic_data():
    return load_and_clean_data()
