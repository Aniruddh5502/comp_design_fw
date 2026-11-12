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
# [ ] 3. Scale the data, store scales
# [ ] 4. Seperate the data to train, error map, val, test 50:30:10:10
# [ ] 5. Convert to Numpy values -> Tensor -> DataLoader -> Return

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


def scale_normal(dataframe:pd.DataFrame, scale_dir:Path)->pd.DataFrame:
    """Scales the relevant features based on the directory structure using Min-Max scaling."""
    scaled_data = dataframe.copy()
    
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
            
    return scaled_data





if __name__=="__main__":
    try:
        # 1. Load and clean
        loadded_dataframe = load_and_clean_dataset(dataset_path=dataset_raw_file)
        print(f"Data loaded and cleaned. Shape: {loadded_dataframe.shape}")

        # 2. Scaling
        scaled_df = scale_normal(loadded_dataframe, scale_dir=data_folder)
        print(f"Data scaled. {scaled_df.describe()}")


    except Exception as e:
        print(f"An error occurred during data loading pipeline: {e}")