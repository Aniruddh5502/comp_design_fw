"""
Processing data. here we make first test dataset, 20%, and 80% training.
Now from this 80% training dataset which is 1897*80% = 1517[floor] taking 1500 as a training
size for the max model. We shall build 5 models. each with different scale of training data.

Training data size
1. model no 1 training size: 300
2. model no 2 training size: 600
3. model no 3 training size: 900
4. model no 4 training size: 1200
5. model no 5 training size: 1500
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import TypedDict
from dataclasses import dataclass, asdict
from rich.console import Console

con = Console()

from typing import TypedDict

class ColumnInfo(TypedDict):
    name: str
    dtype: str
    max: float | None
    min: float | None
    mean: float | None
    std: float | None
    missing: int

class Analytics(TypedDict):
    rows: int
    cols: int
    columns: list[ColumnInfo]     

## First read the raw data and process it.
def clean_csv(csv_filePath: Path, save_as_filename:str)->pd.DataFrame|str:
    """It saves the dataframe in the same folder with the updated name"""
    try:
        # =========================================================
        ## Validate the file if it exists
        # =========================================================
        if not csv_filePath.exists():
            con.print(f"Error Finding dataset at: {csv_filePath}")
            return "The requested file (dataset) doesn't exist"
        # =========================================================
        
        # =========================================================
        ## load the dataset and work on it.
        # =========================================================
        dataframe:pd.DataFrame
        dataframe = pd.read_csv(csv_filePath)
        # =========================================================

        # =========================================================
        # Now we clean the dataset and then save 
        # =========================================================
        # converting all the "" empty strings to pd.nan values
        dataframe_stage_1:pd.DataFrame
        dataframe_stage_1 = dataframe.replace(r'^\s*$', np.nan, regex=True)

        # now removing rows that have at least one empty cell
        dataframe_stage_1 = dataframe_stage_1.dropna(axis=0, how='any',inplace=False, ignore_index=False)
        # =========================================================


        # =========================================================
        # Writing the cleaned dataset into the desired location with desired name
        # =========================================================
        root_folder = csv_filePath.parent
        # create the target filepath for storing the cleaned dataset.
        target = root_folder / f"{save_as_filename}"
        if target.suffix != ".csv":
            save_as_filename += ".csv"
            target = root_folder / f"{save_as_filename}"
        
        dataframe_stage_1.to_csv(f'{target}', index=False, encoding='utf-8')
        # =========================================================
        return dataframe_stage_1
    except Exception as e:
        return f"ERROR: {e}"


def get_input_output_csv(cleaned_csvPath: Path, dataset_name:str)->pd.DataFrame|str:
    """
    This function takes the clean dataset csv file path and extracts the columns that
    are necessary for training & testing only. And discard metadata columns like design index
    status etc fields that aren't necessary for model training testing or any analysis.
    
    Args:
        cleaned_csvPath: The Path variable of cleaned dataset.
        dataset_name: The file where we intend to save our specific columns that will store training data
    """
    try:
        # Input File validation
        if not cleaned_csvPath.exists():
            return "The reference cleaned dataset does not exist."
        if cleaned_csvPath.suffix != ".csv":
            return "Wrong file has been used as a reference"
        
        # Loading the dataset
        dataframe:pd.DataFrame
        dataframe = pd.read_csv(cleaned_csvPath, encoding='utf-8')
        training_dataframe:pd.DataFrame
        training_dataframe = dataframe.iloc[:,[1,2,3,4,6,7,8,9,10,11]].copy()
        
        # Saving the file in the processed folder
        root = Path(__file__).parent.parent
        training_dataframe_path = root / "data" / "processed" / f"{dataset_name}"

        # check if the provided name contains .csv
        if training_dataframe_path.suffix != ".csv":
            dataset_name += ".csv"
            training_dataframe_path = root / "data" / "processed" / f"{dataset_name}"
        
        training_dataframe.to_csv(training_dataframe_path, encoding='utf-8', index=False)    
        return training_dataframe
    except Exception as e:
        return f"ERROR: {e}"


def dataset_analytics(full_dataset:Path, analytics_file_path:Path)->dict:
    """
    
    """
    try:
        pass
    except Exception as e:
        return {
            "status":"ERROR",
            "message":e,
        }


if __name__=="__main__":
    root = Path(__file__).parent.parent
    dataset = root / "data" / "raw" / "inputs.csv"
    dataframe = clean_csv(dataset, "cleaned.csv")
    
    cleaned_daatset_path = root / "data" / "raw" / "cleaned.csv"
    cleaned_dataset = get_input_output_csv(cleaned_daatset_path, "trainset")
    print(cleaned_dataset)
    