"""
This code will take the ErrorMapper model and run it over error dataset and make
data for error mapper training.

# Method
This code shall 
1. Get the Primary() class model from assisted_nn.core.models
2. Run them over the error dataset
3. get the error from the prediction of Primary class model and from actual output data from error dataset.
4. Then prepare that data as a csv to train the assisted_nn.core.models ErrorMapper() class using the assisted_nn.core.train_error

"""

# External Imports
from pathlib import Path
import pandas as pd
import numpy as np
from rich.console import Console

console = Console()

# local imports
from assisted_nn.core.models import ErrorMapper, Primary
from assisted_nn.utils.data_loader import get_dataloaders, load_and_clean_dataset, scale_normal
from assisted_nn.utils.config import dataset_raw_file, data_folder
from assisted_nn.utils.config import accent, success, another, focus

def generate_error_data_loader(raw_dataset_path:Path=dataset_raw_file)->dict:
    # this function will take the error dataset, create df, dataloader
    # Instantiate ErrorMapper() model
    # Run the dataset through the error dataloader get predictions
    # Compute error from actual data
    # create dataloader for training the ErrorMapper() model
    
    # Input Validation
    if raw_dataset_path.suffix == "csv":
        console.print(f"[{focus}]Provided dataset path is not a csv file[/]")
        return {
            "status":False,
            "content":None,
            "error":"Provided dataset path is not a csv file",
        }
        
    # Now we generate dataloader from csv
    # load and clean dataset
    dataset = load_and_clean_dataset(raw_dataset_path)
    dataset = scale_normal(dataframe=dataset, scale_dir=data_folder, filename="error_scales.json")
    data_loader = get_dataloaders(dataset)

    # dataloader provided all the dataloaders
    # now we need to extract the error data loader
    error_dataloader = data_loader['content']['error']
    return {
        "status":True,
        "content":{
            "error":error_dataloader
        },
        "error":None,
    }

# now this function will use the generate_error_data_loader to get the dataloader
# and do the predictions from primary model on those data, and record the predictions
# and create a df with input columns and generated error columns
def generate_error_mapper_models_training_dataset(model:Primary)->pd.DataFrame:
    pass


if __name__ == "__main__":
    error_model = ErrorMapper()
    result = generate_error_data_loader(raw_dataset_path=dataset_raw_file)
    print(result)