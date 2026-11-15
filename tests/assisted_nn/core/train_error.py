# External Imports
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from rich.console import Console

# Local Imports
from assisted_nn.core.models import ErrorMapper
from assisted_nn.utils.config import data_folder, model_folder, plots_folder
from assisted_nn.utils.config import accent, focus, success, another

console = Console()

# ================================================================================
# NEEDS WORK
# Make it the class that runs the primary model and sends back the error 
# map csv data. And maybe scale it to normal and save the scale and scaled data
# then train on it, and use the scaling limits to do error prediction/or maybe we don't
# need to scale back after all
# we can use the nromalized output max 1 and min 0 as an percentage estimate and see 
# correlation with absolute error if it exists at all.
# ================================================================================
class ErrorMap():
    def __init__(self,
                 device:str='cpu',
                 seed:int=41,
                 error_model_data_csv:Path=data_folder):
        self.device = device
        self.seed = seed
        self.error_data_path = error_model_data_csv

        self.model = ErrorMapper().to(device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss  = nn.MSELoss()
        self.train_loss:list=[]
        self.val_loss:list=[]
        self.best_val_loss=float("inf")
        self.console = Console()

    def get_error_map(self):
        pass
# ================================================================================
