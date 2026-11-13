"""
Here we shall define the model classes.
1. Main Predictor Netework
2. Error Predictor Network
"""
import torch.nn as nn
from typing import List

class Primary(nn.Module):
    """Single vanilla NN for main predictor"""
    def __init__(self, input_dim:int=4, hidden_dim:List[int]|None=None, output_dim:int=6):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [32, 32]
        layers = []
        previous_dim = input_dim
        for dimension in hidden_dim:
            layers.append(nn.Linear(previous_dim, dimension))
            layers.append(nn.ReLU())
            previous_dim = dimension
        # Output layer: No activation
        layers.append(nn.Linear(previous_dim, output_dim))
        # inclusing the layers in sequential array
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ErrorMapper(nn.Module):
    """Single vanilla NN for error predictor"""
    def __init__(self, input_dim:int=4, hidden_dim:List[int]|None=None, output_dim:int=6):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [32, 32]
        layers = []
        previous_dimension = input_dim
        for dimension in hidden_dim:
            layers.append(nn.Linear(previous_dimension, dimension))
            layers.append(nn.ReLU())
            previous_dimension = dimension
        layers.append(nn.Linear(previous_dimension, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


if __name__=="__main__":
    # testing if the classes work
    try:
        primary =Primary()
        error_mapper = ErrorMapper()
        print(f"Class Initialization Succesfull")
    except Exception as e:
        print(f"Error Occured: {e}")