import numpy as np
import pandas as pd
from typing import Dict, List, Union
from scipy.stats import qmc

class Sampler:
    """
    Handles design of experiments (DoE) for mechanical design space.
    Supports Grid Sampling and Latin Hypercube Sampling (LHS).
    """
    
    def __init__(self, design_space: Dict):
        """
        Args:
            design_space: The 'parameters' dictionary from design_space.json
        """
        self.params = design_space.get('input_parameters', design_space.get('parameters'))
        self.param_names = list(self.params.keys())
        self.bounds = np.array([[p['min'], p['max']] for p in self.params.values()])

    def generate_grid(self, points_per_dim: int = 5) -> pd.DataFrame:
        """
        Generates a full factorial grid of design points.
        """
        grids = [np.linspace(p['min'], p['max'], points_per_dim) for p in self.params.values()]
        mesh = np.array(np.meshgrid(*grids)).T.reshape(-1, len(self.param_names))
        
        return self._to_dataframe(mesh)

    def generate_lhs(self, num_samples: int) -> pd.DataFrame:
        """
        Generates a space-filling Latin Hypercube Sample.
        """
        sampler = qmc.LatinHypercube(d=len(self.param_names))
        sample = sampler.random(n=num_samples)
        
        # Scale sample from [0, 1] to actual bounds
        scaled_sample = qmc.scale(sample, self.bounds[:, 0], self.bounds[:, 1])
        
        return self._to_dataframe(scaled_sample)

    def _to_dataframe(self, data: np.ndarray) -> pd.DataFrame:
        """
        Converts numpy array to a labeled pandas DataFrame with Design_ID.
        """
        df = pd.DataFrame(data, columns=self.param_names)
        df.insert(0, 'Design_ID', range(1, len(df) + 1))
        return df
