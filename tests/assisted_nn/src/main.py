"""
Now this is the final step to get the behaviior results
from the models that we have already trained.
We will create dataloader for the test set, and pass
the dataloader through the Primary Model. And get predictions.
Then use the target columns from the test data target columns
and compute error.
Now take the same feature columns from test data and run them through
the error mapper model to predict error for each target column.

Then at the end we take the predicted error per each column both
from computer error from Primary models prediction and error predicted
by ErrorMapper and compute Spearman Correlation and Pearson correlation.
"""

# imports
import numpy as np
import pandas as pd

import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


from assisted_nn.core.models import Primary, ErrorMapper
from assisted_nn.utils.config import model_folder, data_folder, accent, success, another, focus

# now variable declarations 
primary_model_weights_path = model_folder / "primary.pt"
error_mapper_model_weights_path = model_folder / "error_mapper.pt"

def create_dataloader_from_test(test_dataset_path:Path=data_folder / "test_df.csv")->DataLoader|dict:
    # input validation
    if test_dataset_path.suffix != ".csv":
        return {
            "status":False,
            "content":None,
            "error":"Dataset path provided is not a .csv file",
        }
    
    # Now we make dataloader from the csv(no shuffling as we want to keep them deterministic)
    test_df = pd.read_csv(test_dataset_path)
    features = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height']
    targets = ['max_stress_von_mises', 'max_deformation', 'mode_1_freq', 'mode_2_freq', 'mode_3_freq', 'mode_4_freq']
    
    # creating tensors
    feature_t = torch.tensor(test_df[features].values, dtype=torch.float32)
    target_t  = torch.tensor(test_df[targets].values,  dtype=torch.float32)
    
    # Tensor Dataset
    test_ds   = TensorDataset(feature_t, target_t)
    # dataloader
    test_dl   = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    return {
        "status":True,
        "content":test_dl,
        "error":None,
    }

def get_actual_error_df(test_dataloders:DataLoader, 
                        primary_model_path:Path=primary_model_weights_path)->pd.Dataframe:
    """
    Here we shall first load the primary model and 
    then use the test_dataloaders to predict the target columns
    then use the actual target columns to
    """
    # load the primary model first.
    device = torch.device('cpu')
    try:
        model = Primary()
        model.load_state_dict(torch.load(primary_model_path, map_location=device))
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Model loading failed. ERROR: {e}")
    
    targets = ['max_stress_von_mises', 'max_deformation', 'mode_1_freq', 'mode_2_freq', 'mode_3_freq', 'mode_4_freq']

    predictions = []
    y_true_all  = []
    # with no grad, run prediction
    with torch.no_grad():
        for x_batches, y_batches in test_dataloders:
            x_batches = x_batches.to(device)
            y_batches = y_batches.to(device)
            predictions.append(model(x_batches).cpu().numpy())
            y_true_all.append(y_batches.cpu().numpy())
    
    preds = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(y_true_all, axis=0)
    abs_error = np.abs(y_true-preds)
    abs_error_df = pd.DataFrame(abs_error, columns=[f"{t}_error" for t in targets])
                
    return abs_error_df

def get_predicted_error_df(test_dataloaders:DataLoader,
                           error_mapper_model_path:Path=error_mapper_model_weights_path):
    """
    Here we shall use the ErrorMapper class model to predict
    the error contents for the target columns from the input 
    columns. And then return them as df.
    """
    # input validation
    if not isinstance(test_dataloaders, DataLoader):
        return {
            "success":False,
            "content":None,
            "error":"Provided dataloader is not valid",
        }
    
    # After input validation we take the feature, targets
    # pass them through the error mapper get the errors df
    device = torch.device('cpu')
    model = ErrorMapper()
    model.load_state_dict(torch.load(error_mapper_model_path, map_location=device))
    model.to(device)
    model.eval()
    
    targets = ['max_stress_von_mises', 'max_deformation', 'mode_1_freq', 'mode_2_freq', 'mode_3_freq', 'mode_4_freq']
    
    predictions, y_true = [],[]
    with torch.no_grad():
        for x_batch, y_batch in test_dataloaders:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            predictions.append(model(x_batch).cpu().numpy())
            y_true.append(y_batch.cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    y_true_all = np.concatenate(y_true, axis=0)
    pred_df = pd.DataFrame(pred, columns=[f"{t}_error" for t in targets])
    return pred_df


def correlation_analysis(true_error:pd.DataFrame, 
                         pred_error:pd.DataFrame)->dict:
    """
    This function takes two error df prediction and comp
    """
    target_columns = []
    for column in true_error.columns:
        target_columns.append(column)
        pearson_correlation = true_error[f'{(column)}'].corr(pred_error[f"{column}"], method="pearson")
        spearman_correlation= true_error[f"{column}"].corr(pred_error[f"{column}"], method="spearman")
        console.print(f"\n[bold {accent}]Column: [/][bold]{column}[/]")
        console.print(f"1. Pearson Correlation  : {pearson_correlation}")
        console.print(f"2. Spearman Correlation : {spearman_correlation}")
    pass

if __name__=="__main__":
    from rich.console import Console
    console = Console()
    
    # Test Dataloader
    dataloader = create_dataloader_from_test(test_dataset_path=data_folder/"test_df.csv")
    if isinstance(dataloader["content"], DataLoader):
        console.print(f"[{accent}]Test 1:[/]    [{success}]Dataloader function test succesfull[/]")
    else:
        console.print(f"[{accent}]Test 1:[/]    [{focus}] Dataloader function test failed[/]")
    
    # getting error predictions
    actual_error_df = get_actual_error_df(dataloader["content"])
    if isinstance(actual_error_df, pd.DataFrame):
        console.print(f"[{accent}]Test 2:[/]    [{success}]Primary Model prediction test succesfull[/]")
    else:
        console.print(f"[{accent}]Test 2:[/]    [{focus}]Primary Model prediction failed")
    print(actual_error_df.describe())
    
    # predicting errors with error mapper
    pred_error_df = get_predicted_error_df(dataloader["content"])
    if isinstance(pred_error_df, pd.DataFrame):
        console.print(f"[{accent}]Test 3:[/]    [{success}]Error Mapper models prediction succesfull")
    else:
        console.print(f"[{accent}]Test 3:[/]    [{focus}]Error Mapper model prediction failed.")
        
    # now the correlation analysis
    correlation_result = correlation_analysis(actual_error_df, pred_error_df)
    