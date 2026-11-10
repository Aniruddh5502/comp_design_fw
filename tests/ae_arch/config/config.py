"""
This file controls the saving and loading the files, paths, folders, settings
"""
from typing import TypedDict
from pathlib import Path

ae_test_root    =   Path(__file__).parent.parent
root            =   Path(__file__).parents[3]
dataset_raw_file=   root / "data" / "raw" / "inputs.csv"

class Config(TypedDict):
    arch:str
    dof:int
    input_dim:int
    output_dim:int

class ManageConfig():
    def init(self):
        self.config_file
