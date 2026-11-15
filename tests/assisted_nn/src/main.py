"""
This file is the orchastrator for all the work
1. dataloader creation
2. primary model training and eval tested on validation set to check overfitting
3. error map creation
4. error model training and eval tested on validation set to check overfitting
5. error prediction and absolute error correlation test data
"""

from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# local imports
# utils
from assisted_nn.utils.config import data_folder, plots_folder, dataset_raw_file
from assisted_nn.utils.data_loader import get_dataloaders, load_and_clean_dataset, scale_normal
# core
from assisted_nn.core.models import Primary, ErrorMapper
from assisted_nn.core.train import PrimaryTrainer

# 1. Train the Primary Model using PrimaryTrainer
def main()->dict:
    
    # 1. load data and train primary
    primary_trainer = PrimaryTrainer()
    cleaned_df = load_and_clean_dataset(dataset_path=dataset_raw_file)
    scaled_df  = scale_normal(dataframe=cleaned_df, scale_dir=data_folder)
    dataloader = get_dataloaders(dataframe=scaled_df)
    
    # getting dataloaders
    train_dataloader = dataloader["content"]["train"]
    val_dataloader   = dataloader["content"]["val"]
    error_dataloader = dataloader["content"]["error"]
    test_dataloader  = dataloader["content"]["test"]
    
    # training the primary model
    primary_trainer.train(train_loader=train_dataloader, val_loader=val_dataloader, plot=True)

    # 2. get error map from error    
    

if __name__=="__main__":
    main()