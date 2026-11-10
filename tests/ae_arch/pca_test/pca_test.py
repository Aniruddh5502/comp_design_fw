"""
Here we shall compute the PCA for the input data.
"""
from ae_arch.config.config import root, ae_test_root, dataset_raw_file
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


def get_dataset(dataset_path:Path)->pd.DataFrame:
    """reading and sending a cleaned dataset as pandas dataframe"""
    # validating
    if not dataset_path.is_file:
        return "The path provided is not a file"
    if not dataset_path.suffix == ".csv":
        return "The path provided is not a csv file"
    
    df = pd.read_csv(dataset_raw_file)
    df = df.replace("",np.nan)
    df = df.dropna()
    return df

def compute_pca(dataframe:pd.DataFrame, n_components:int|None=None, scale:bool=True)->dict:
    """
    here we take the dataset on which we want to compute pca and return a dict 
    with all the information
    """
    # input validation
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Dataframe must be a pandas dataframe")
    
    dataframe = dataframe.replace("",np.nan)
    dataframe = dataframe.dropna()
    
    # pca
    scaled_df   =   StandardScaler().fit_transform(dataframe)   # first scaling
    pca         =   PCA()
    x_pca       =   pca.fit_transform(scaled_df)
    return {
        "pca":pca,
        "x_pca":x_pca,
        "explained_variance_ratio":pca.explained_variance_ratio_,
        "cumulative":pca.explained_variance_ratio_.cumsum(),
    }
    
    

if __name__=="__main__":
    df = get_dataset(dataset_raw_file)
    # convert the inputs columns index into a list
    input_df = df[df.columns[1:5]]
    pca_analysis = compute_pca(input_df)
    
    # now we store the results into results/pca_test_result.md
    report = f"""# PCA analysis result for rach axis
{pca_analysis["explained_variance_ratio"]}
    
As all the 4 axis contains significant amount of variance data from the input variations, its impossible to use a autoencoder to squize low D latent representations from this.
    """
    # writing the file into the results folder
    result_dir = ae_test_root / "pca_test"/"result"
    result_dir.mkdir(exist_ok=True)
    result_file = result_dir / "results.md"
    with open(result_file, 'w', encoding='utf-8') as file:
        file.write(report)
    
