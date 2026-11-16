"""
Here we shall test if dataloading dataloader making and model predictions are
happening or not and for as a  practice for that
"""
import torch, time, os, json
import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data import TensorDataset, DataLoader

from assisted_nn.core.models import Primary
from assisted_nn.core.models import ErrorMapper
from assisted_nn.core.train import PrimaryTrainer
from assisted_nn.utils.config import model_folder, data_folder

features = ["beam_width", "beam_length", "fillet_radius", "beam_height"]
targets  = ["max_stress_von_mises", "max_deformation", "mode_1_freq", "mode_2_freq", "mode_3_freq", "mode_4_freq"]

# now we load the train data make dataloader from it
# do predictions and calculate accuracy and also
# load the val set to test accuracy on val set

train_csv   =   data_folder / "train_df.csv"
val_csv     =   data_folder / "val_df.csv"
error_csv   =   data_folder / "error_df.csv"
test_csv    =   data_folder / "test_df.csv"


# first read the csv and turn them into dataframes
train_df    =   pd.read_csv(train_csv)
val_df      =   pd.read_csv(val_csv)
error_df    =   pd.read_csv(error_csv)
test_df     =   pd.read_csv(test_csv)

# then convert them into numpy values with specific columns and make tensor from them
train_feature_t =   torch.tensor(train_df[features].values, dtype=torch.float32)
train_target_t  =   torch.tensor(train_df[targets].values, dtype=torch.float32)

val_feature_t   =   torch.tensor(val_df[features].values, dtype=torch.float32)
val_target_t    =   torch.tensor(val_df[targets].values, dtype=torch.float32)

error_feature_t =   torch.tensor(error_df[features].values, dtype=torch.float32)
error_target_t  =   torch.tensor(error_df[targets].values, dtype=torch.float32)

test_feature_t  =   torch.tensor(test_df[features].values, dtype=torch.float32)
test_target_t   =   torch.tensor(test_df[targets].values, dtype=torch.float32)


# and then combine the feature and target to make tensor dataset
train_ds    =   TensorDataset(train_feature_t, train_target_t)
val_ds      =   TensorDataset(val_feature_t, val_target_t)
error_ds    =   TensorDataset(error_feature_t, error_target_t)
test_ds     =   TensorDataset(test_feature_t, test_target_t)

# now we make them into dataloaders
train_dl    =   DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl      =   DataLoader(val_ds, batch_size=32, shuffle=True)
error_dl    =   DataLoader(error_ds, batch_size=32, shuffle=True)
test_dl     =   DataLoader(test_ds, batch_size=32, shuffle=True)

# now as all the dataloaders are prepared now we can load the models and 
# predict the outputs and compute the error in all the sets as percentage
# of the actual values, how many percent does the prediction deviate
primary_model = Primary()
primary_trainer = PrimaryTrainer()
primary_model.load_state_dict(torch.load(model_folder / 'primary.pt'))

primary_trainer.evaluate(loader=train_dl, 
                         target_names=targets,
                         split_name="TRAIN",
                         model=primary_model)

primary_trainer.evaluate(loader=test_dl, 
                         target_names=targets,
                         split_name="TEST",
                         model=primary_model)

primary_trainer.evaluate(loader=val_dl,
                         target_names=targets,
                         split_name="VALIDATION",
                         model=primary_model)