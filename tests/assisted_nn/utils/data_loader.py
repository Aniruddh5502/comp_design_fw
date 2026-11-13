import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype

# local imports 
from assisted_nn.utils.config import dataset_raw_file, data_folder

# tasks in sequence that need to be done
# [x] 1. load and clean the data
# [x] 2. Filter columns that are needed
# [x] 3. Scale the data, store scales
# [x] 4. Seperate the data to train, error map, val, test 50:30:10:10
# [x] 5. Convert to Numpy values -> Tensor -> DataLoader -> Return

# Loading and cleaning the dataset from csv
def load_and_clean_dataset(dataset_path:Path)->pd.DataFrame:
    """This function loads, cleans and filters columns to create a filtered dataframe."""
    # input validation
    if not dataset_path.is_file:
        raise FileNotFoundError(f"Provided dataset file path is not a file: {dataset_path}")
    if not dataset_path.suffix == ".csv":
        raise ValueError(f"Provided dataset file is not a .csv file: {dataset_path}")
    # loading dataset as pandas dataframe
    df = pd.read_csv(dataset_path)
    # Replacing empty strings with np.nan
    df = df.replace("", np.nan)
    # Dropping rows with any missing values
    df = df.dropna()
    # Filtering the columns we only need for this dataset, discarding others
    required_cols = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height', 'max_stress_von_mises', 'max_deformation', 'mode_1_freq', 'mode_2_freq', 'mode_3_freq', 'mode_4_freq']
    # Only keep columns that exist in the dataframe
    df = df[required_cols]
    # Returning the cleaned and column filtered data
    return df

# scaling the dataset into normal form
def scale_normal(dataframe:pd.DataFrame, scale_dir:Path)->pd.DataFrame:
    """Scales the relevant features based on the directory structure using Min-Max scaling."""
    scaled_data = dataframe.copy()
    import json
    scale_content:json = {}
    for col in scaled_data.columns:
        if is_numeric_dtype(scaled_data[col]):
            min_val = scaled_data[col].min()
            max_val = scaled_data[col].max()
            
            if max_val != min_val:
                scaled_data[col] = (scaled_data[col] - min_val) / (max_val - min_val)
            else:
                # Handle the case where all values are the same (max_val == min_val)
                scaled_data[col] = 0.0
        else:
            # Log or handle non-numeric columns if necessary, here we leave them as is
            pass
        scale_content[f"{col}"] = {
            "max":max_val,
            "min":min_val,
        }
    # storing the scales as json file in scale_dir
    scale_json_file = scale_dir / "scales.json"
    with open(scale_json_file, 'w', encoding='utf-8') as file:
        json.dump(scale_content, file, indent=4)
    return scaled_data

# now we seperate the data into train, error map, val, test
def get_dataloaders(dataframe:pd.DataFrame|None=None,
                    seed:int=41,
                    device:str='cpu',
                    train:int=50, 
                    error_map:int=30, 
                    val:int=10, 
                    test:int=10,
                    build_from_scratch:bool=False)->str|dict:
    """This function creates dataloaders from cleaned and scaled dataframes"""
    # Input Validation
    if dataframe is None:
        return {
            "status":False,
            "content":None,
            "error":"The Dataframe provided is NoneType"
        }
    if not isinstance(dataframe, pd.DataFrame):
        return {
            "status":False,
            "content":None,
            "error":"The dataframe provided is not a dataframe",
        }
    
    # we shall check if the dataloaders indexes are present or not
    feature_columns = ['beam_width', 'beam_length',  'fillet_radius', 'beam_height']
    target_columns = ['max_stress_von_mises', 'max_deformation', 'mode_1_freq', 'mode_2_freq', 'mode_3_freq', 'mode_4_freq']
    
    train_df_path   =   data_folder /   f"train_df.csv"
    error_df_path   =   data_folder /   f"error_df.csv"
    val_df_path     =   data_folder /   f"val_df.csv"
    test_df_path    =   data_folder /   f"test_df.csv"
    
    train_df = error_df = val_df = test_df = None
    
    # now when the build_from_scratch=False(default) we look for stored files as csv
    # if any one of them are not present we proceed with building them up
    if not build_from_scratch:
        try:
            files = [train_df_path, error_df_path, val_df_path, test_df_path]
            if all(f.is_file() for f in files):
                try:
                    # when all files exist we load them in dataframes
                    train_df = pd.read_csv(train_df_path)
                    error_df = pd.read_csv(error_df_path)
                    val_df   = pd.read_csv(val_df_path)
                    test_df  = pd.read_csv(test_df_path)
                    print(f"All data loaded from csv")
                except Exception as e:
                    print(f"Couldn't read split csv ({e}) -> rebuilding")
                    train_df = error_df = val_df = test_df = None
            else:
                print(f"Split csvs are missing -> building from scratch")
        except Exception as e:
            print(f"ERROR occured: {e}")
            
    if train_df is None:
        print(f"Building from scratch")
        try:
            np.random.seed(seed)
            idx = np.random.permutation(len(dataframe))
            
            n           =   len(dataframe)
             
            train_n     =   int(round(n * train         / 100))
            error_n     =   int(round(n * error_map     / 100))
            val_n       =   int(round(n * val           / 100))
            
            train_idx   =   idx[:train_n]
            error_idx   =   idx[train_n:train_n+error_n]
            val_idx     =   idx[train_n+error_n:train_n+error_n+val_n]
            test_idx    =   idx[train_n+error_n+val_n:]
            
            # making splitted dataframes
            train_df    =   dataframe.iloc[train_idx].reset_index(drop=True)
            error_df    =   dataframe.iloc[error_idx].reset_index(drop=True)
            val_df      =   dataframe.iloc[val_idx].reset_index(drop=True)
            test_df     =   dataframe.iloc[test_idx].reset_index(drop=True)

            # now saving them for future reading
            train_df.to_csv(train_df_path, index=False)
            error_df.to_csv(error_df_path, index=False)
            val_df.to_csv(val_df_path, index=False)
            test_df.to_csv(test_df_path, index=False)
        except Exception as e:
            print(f"Error occured: {e}")
            
        # creating numpy arrays for feature and target set
    
    try:
        train_feature_df    =   train_df[feature_columns].values
        train_target_df     =   train_df[target_columns].values
        error_feature_df    =   error_df[feature_columns].values
        error_target_df     =   error_df[target_columns].values
        val_feature_df      =   val_df[feature_columns].values
        val_target_df       =   val_df[target_columns].values
        test_feature_df     =   test_df[feature_columns].values
        test_target_df      =   test_df[target_columns].values
                
        # Making the torch tensor combining the feature and target
        train_feature_t     =   torch.tensor(train_feature_df, dtype=torch.float32, device=device)
        train_target_t      =   torch.tensor(train_target_df, dtype=torch.float32, device=device)
        error_feature_t     =   torch.tensor(error_feature_df, dtype=torch.float32, device=device)
        error_target_t      =   torch.tensor(error_target_df, dtype=torch.float32, device=device)
        val_feature_t       =   torch.tensor(val_feature_df, dtype=torch.float32, device=device)
        val_target_t        =   torch.tensor(val_target_df, dtype=torch.float32, device=device)
        test_feature_t      =   torch.tensor(test_feature_df, dtype=torch.float32, device=device)
        test_target_t       =   torch.tensor(test_target_df, dtype=torch.float32, device=device)
                
            # combining feature and target dataset into a tensor dataset
        train_ds            =   TensorDataset(train_feature_t, train_target_t)
        error_ds            =   TensorDataset(error_feature_t, error_target_t)
        val_ds              =   TensorDataset(val_feature_t, val_target_t)
        test_ds             =   TensorDataset(test_feature_t, test_target_t)
                
        # Making dataloader from them
        train_dataloader    =   DataLoader(train_ds, batch_size=32, shuffle=True)
        error_dataloader    =   DataLoader(error_ds, batch_size=32, shuffle=True)
        val_dataloader      =   DataLoader(val_ds, batch_size=32, shuffle=True)
        test_dataloader     =   DataLoader(test_ds, batch_size=32, shuffle=True)
                
        return {
            "status":True,
            "content":{
                "train" :   train_dataloader,
                "error" :   error_dataloader,
                "val"   :   val_dataloader,
                "test"  :   test_dataloader,
                "train_size":   train_df.shape,
                "error_size":   error_df.shape,
                "val_size"  :   val_df.shape,
                "test_size" :   test_df.shape,
                },
            "error":None,
            }
    except Exception as e:
        return {
            "status":False,
            "content":None,
            "error":e,
        }


if __name__=="__main__":
    try:
        # 1. Load and clean
        loadded_dataframe = load_and_clean_dataset(dataset_path=dataset_raw_file)
        print(f"Data loaded and cleaned. Shape: {loadded_dataframe.shape}")

        # 2. Scaling
        scaled_df = scale_normal(loadded_dataframe, scale_dir=data_folder)
        print(f"\n\n# Data scaled. \n{scaled_df.shape}")

        # 3. Dataloader
        result = get_dataloaders(scaled_df)
        if result["status"]:
            print(f"Dataloader created succesfully")
        result = get_dataloaders(scaled_df, build_from_scratch=True)
        from rich.console import Console
        from rich.pretty  import Pretty
        con = Console()
        con.print(Pretty(result))
            
    except Exception as e:
        print(f"An error occurred during data loading pipeline: {e}")