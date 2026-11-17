"""
Plan for error mapper data making is.
1. We first run the Primary model to do inference over error_df
2. Get the predictions, compute error and make a new dataset with features+errors
3. Store the data as CSV and send the dataloader back
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# local imports 
from assisted_nn.utils.config import data_folder, model_folder
from assisted_nn.core.models import Primary

# now we get the dataloader from this get_dataloaders
# Load weights of the primary model and instantiate the model
# then run them through the Primary model to get prediction columns
# and then compute the errors and combine feature_cols+target_error_cols
# save the new df as CSV [this is the data on which the error mapper model will be trained upon]
# also we calculate errors on val set to create a validation set for the
# error mapper model, so it doesn't overfit also.

error_df_csv = data_folder / "error_df.csv"
val_df_csv  =   data_folder / "val_df.csv"

model_weights_path = model_folder / 'primary.pt'

def get_error_data(error_csv_file: Path = error_df_csv,
                   val_csv_file: Path = val_df_csv) -> dict:
    
    features = ["beam_width", "beam_length", "fillet_radius", "beam_height"]
    targets  = ["max_stress_von_mises", "max_deformation",
                "mode_1_freq", "mode_2_freq", "mode_3_freq", "mode_4_freq"]

    error_df    =   pd.read_csv(error_csv_file)
    val_df      =   pd.read_csv(val_csv_file)
    
    error_feature_t = torch.tensor(error_df[features].values, dtype=torch.float32)
    error_target_t  = torch.tensor(error_df[targets].values,  dtype=torch.float32)
    error_ds = TensorDataset(error_feature_t, error_target_t)
    error_dl = DataLoader(error_ds, batch_size=32, shuffle=False)

    val_feature_t = torch.tensor(val_df[features].values, dtype=torch.float32)
    val_target_t  = torch.tensor(val_df[targets].values, dtype=torch.float32)
    val_ds = TensorDataset(val_feature_t, val_target_t)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)

    device = torch.device("cpu")                                   # ← moved up

    try:
        model = Primary()
        model.load_state_dict(torch.load(model_weights_path, map_location=device))  # ← map_location
        print("Primary model successfully loaded.")
    except Exception as e:
        print(f"Model loading failed in assisted_nn.utils.error_data: {e}")
        raise

    model = model.to(device).eval()

    predictions, y_true_all = [], []
    with torch.no_grad():
        for x_batch, y_batch in error_dl:
            x_batch = x_batch.to(device)                           # ← move both
            y_batch = y_batch.to(device)
            predictions.append(model(x_batch).cpu().numpy())       # ← .cpu()
            y_true_all.append(y_batch.cpu().numpy())               # ← .cpu()

    predictions_val, y_true_all_val = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions_val.append(model(x_batch).cpu().numpy())
            y_true_all_val.append(y_batch.cpu().numpy())
    
    # ===============================================================
    # Error Traininf
    # ===============================================================
    preds  = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true_all, axis=0)

    abs_error = np.abs(preds - y_true)
    abs_error_df = pd.DataFrame(
        abs_error, columns=[f"{t}_error" for t in targets]         # ← derived
    )

    combined_error_df = pd.concat([error_df[features].reset_index(drop=True), abs_error_df],
                                    axis=1)

    combined_error_df.to_csv(data_folder / "error_mapper_training.csv", index=False)
    
    # ===============================================================
    # for val
    # ===============================================================
    preds_val = np.concat(predictions_val, axis=0)
    y_true_val = np.concat(y_true_all_val, axis=0)
    
    abs_error_val = np.abs(preds_val - y_true_val)
    abs_error_val_df = pd.DataFrame(abs_error_val, columns=[f"{t}_error" for t in targets])
    combined_error_val_df = pd.concat([val_df[features].reset_index(drop=True), abs_error_val_df], 
                                      axis=1)
    combined_error_val_df.to_csv(data_folder / "error_mapper_val.csv", index=False)
    
    return {
        "train":combined_error_df,
        "val":combined_error_val_df,
    }

def get_error_dataloaders(error_csv_file: Path = error_df_csv,
                   val_csv_file: Path = val_df_csv):
    """
    This function uses the get_error_data function to get the 
    error_mapper_train and error_mapper_val dataframes and then
    makes dataloaders from them
    """
    # get error dataframes
    dataframes = get_error_data()
    
    train_df = dataframes['train']
    val_df   = dataframes['val']
    
    features = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height']
    targets = ['max_stress_von_mises_error', 'max_deformation_error', 'mode_1_freq_error', 'mode_2_freq_error', 'mode_3_freq_error', 'mode_4_freq_error']
    
    # Now make tensors from these dataframes-> numpy values using .values
    train_features_t    =   torch.tensor(train_df[features].values, dtype=torch.float32)
    train_targets_t     =   torch.tensor(train_df[targets].values, dtype=torch.float32)
    val_features_t      =   torch.tensor(val_df[features].values, dtype=torch.float32)
    val_targets_t       =   torch.tensor(val_df[targets].values, dtype=torch.float32)
    
    # making tensor datasets
    train_t             =   TensorDataset(train_features_t, train_targets_t)
    val_t               =   TensorDataset(val_features_t, val_targets_t)
    
    # making dataloaders
    train_dl            =   DataLoader(train_t, batch_size=32, shuffle=True)
    val_dl              =   DataLoader(val_t, batch_size=32, shuffle=True)
    
    # now we return the dataloaders as a dict
    return {
        "train":train_dl,
        "val":val_dl
    }
    
    
if __name__=="__main__":
    print(f"ErrorMapper models training data details: ")
    print(f"ErrorMapper models training data columns")
    print(f"{get_error_data()['val'].columns}")
    dataloaders = get_error_dataloaders()
    print(f"ErrorMapper models training and validation data descriptions:")
    print(f"{dataloaders}")