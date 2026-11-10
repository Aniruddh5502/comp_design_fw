"""
Cleaning data and supplying the dataloaders for train, val, test
"""

import torch
import os, json
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from rich.console import Console

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

con = Console()

def get_data_loader(
    dataset_raw:Path,
    scale_json:Path,
    clean:bool=True,
    scale:bool=True,
    train:float=0.7, 
    val:float=0.15,
    seed:int=42,
    test:float=0.15,
    device:str='cpu')->dict:
    """
    Args:
        dataset_raw     # raw csv datasets path relative to project root
        scale_json      # scale.json file to save the scales for inputs.csv
        clean           # If the data needs to be cleaned.
        train           # training data ratio out of 1
        val             # validation data ratio out of 1
        seed            # seed for ranodmization of dataset
        test            # testing data ratio out of 1
    """
    # 1. Get the data from the csv file and clean
    try:
        with open(dataset_raw, 'r', encoding='utf-8') as file:
            df = pd.read_csv(file)
        
        if clean:
            df = df.replace("",np.nan)
            df = df.dropna()
        
        total_columns = [
            'beam_width', 
            'beam_length', 
            'fillet_radius',
            'beam_height', 
            'max_stress_von_mises', 
            'max_deformation',
            'mode_1_freq', 
            'mode_2_freq', 
            'mode_3_freq', 
            'mode_4_freq'
            ]
        feature_columns = [
            'beam_width', 
            'beam_length', 
            'fillet_radius',
            'beam_height'
            ]
        target_columns = [
            'max_stress_von_mises', 
            'max_deformation',
            'mode_1_freq', 
            'mode_2_freq', 
            'mode_3_freq', 
            'mode_4_freq'
            ]
        
        # Dropping unnecessary columns
        df = df[total_columns]
        content = {}
        for column in total_columns:
            content[column] = {
                "max":max(df[column]),
                "min":min(df[column]),
            }
        
        # Saving the scale json file in this test folders data
        with open(scale_json,'w',encoding='utf-8') as file:
            json.dump(content, file, indent=4)
        
        # Now we scale the data
        df_scaled = df.copy()
        if scale:
            for col in total_columns:
                df_scaled[col] = (df[col] - content[col]["min"])/(content[col]["max"] - content[col]["min"]) 
        
        # Now we slit the data along 0 axis
        n_train = int(train * len(df_scaled))
        n_val   = int(val * len(df_scaled))
        
        np.random.seed(seed)
        idx = np.random.permutation(len(df_scaled))
        
        train_idx   =   idx[:n_train]
        val_idx     =   idx[n_train:n_train+n_val]
        test_idx    =   idx[n_train+n_val:]
        
        train_df    =   df_scaled.iloc[train_idx].reset_index(drop=True)
        val_df      =   df_scaled.iloc[val_idx].reset_index(drop=True)
        test_df     =   df_scaled.iloc[test_idx].reset_index(drop=True)
        
        # creating numpy datarrays
        train_feature_np = train_df[feature_columns].values
        train_target_np  = train_df[target_columns].values
        val_feature_np   = val_df[feature_columns].values
        val_target_np  = val_df[target_columns].values
        test_feature_np  = test_df[feature_columns].values
        test_target_np   = test_df[target_columns].values
        
        # making them torch tensor
        train_feature_t =   torch.tensor(train_feature_np, dtype=torch.float32, device=device)
        train_target_t  =   torch.tensor(train_target_np, dtype=torch.float32, device=device)
        val_feature_t   =   torch.tensor(val_feature_np, dtype=torch.float32, device=device)
        val_target_t    =   torch.tensor(val_target_np, dtype=torch.float32, device=device)
        test_feature_t  =   torch.tensor(test_feature_np, dtype=torch.float32, device=device)
        test_target_t   =   torch.tensor(test_target_np, dtype=torch.float32, device=device)
        
        # combining feature and target into a tensor dataset
        train_ds        =   TensorDataset(train_feature_t, train_target_t)
        val_ds          =   TensorDataset(val_feature_t, val_target_t)
        test_ds         =   TensorDataset(test_feature_t, test_target_t)
        
        # Making these tensor datasets Dataloaders
        train_dataloader    =   DataLoader(train_ds, batch_size=32, shuffle=True)
        val_dataloader      =   DataLoader(val_ds, batch_size=32, shuffle=True)
        test_dataloader     =   DataLoader(test_ds, batch_size=32, shuffle=True)
        
        return {
            "status":True,
            "content":{
                "train_ds":train_dataloader,
                "val_ds":val_dataloader,
                "test_ds":test_dataloader,
            },
            "error":None
        }
    except Exception as e:
            con.print(f"x    [red]Dataset csv reading has failed:[/] [red]ERROR:[/] [dim]{e}[/]")
            return {
                "status":False,
                "content":None,
                "error":e
            }

if __name__ == "__main__":
    root = Path(__file__).parent.parent.parent.parent
    dataset_csv_file = root / "data"  / "raw"        / "inputs.csv"
    scales_json_file = root / "tests" / "ae_arch"    / "data" / "scales.json"
    print(get_data_loader(dataset_csv_file, scales_json_file))