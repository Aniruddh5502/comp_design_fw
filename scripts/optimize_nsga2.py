import numpy as np
import pandas as pd
import json
import os
import sys

# Ensure root is in path for scripts import
sys.path.append('.')

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from scripts.predictor import SurrogatePredictor

class SurrogateOptimizationProblem(ElementwiseProblem):
    def __init__(self, predictor, design_space):
        self.predictor = predictor
        self.design_space = design_space
        
        params = design_space['input_parameters']
        self.xl = np.array([v['min'] for v in params.values()])
        self.xu = np.array([v['max'] for v in params.values()])
        
        super().__init__(
            n_var=len(self.xl),
            n_obj=2,
            n_constr=0,
            xl=self.xl,
            xu=self.xu
        )

    def _evaluate(self, x, out, *args, **kwargs):
        # Get prediction
        res = self.predictor.predict(x)
        mean = np.array(res["mean_preds"])
        
        # Indices based on scaling_bounds.json:
        # out_0: max_stress, out_1: max_def, out_2: mode_1_freq, out_3: mode_2_freq...
        # Actually, based on scaling_bounds: 
        # 0: stress, 1: def, 2: f1, 3: f2, 4: f3, 5: f4
        f1 = mean[2]
        f2 = mean[3]
        
        # Objective 1: Target f1 around 100Hz
        obj1 = abs(f1 - 100.0)
        
        # Objective 2: Maximize f2/f1 ratio (pymoo minimizes, so negate)
        # Avoid division by zero
        ratio = f2 / (f1 + 1e-6)
        obj2 = -ratio
        
        out["F"] = [obj1, obj2]

def main():
    with open('config/design_space.json', 'r') as f:
        design_space = json.load(f)
    
    predictor = SurrogatePredictor("models/ensemble_1500_42.pth")
    problem = SurrogateOptimizationProblem(predictor, design_space)
    
    algorithm = NSGA2(
        pop_size=100,
        sampling=LHS(),
        crossover=SBX(prob=0.9),
        mutation=PM(prob=0.1),
        eliminate_duplicates=True
    )
    
    print("Starting Natural NSGA-II Optimization...")
    print("Target: f1 ~ 100Hz, Maximize f2/f1 Ratio")
    
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 200),
        seed=42,
        verbose=True
    )
    
    candidates = res.X
    f_values = res.F
    
    all_means = []
    all_stds = []
    
    for x in candidates:
        res_pred = predictor.predict(x)
        all_means.append(res_pred["mean_preds"])
        all_stds.append(res_pred["std_preds"])
    
    param_names = list(design_space['input_parameters'].keys())
    df = pd.DataFrame(candidates, columns=param_names)
    
    df['target_f1_error'] = f_values[:, 0]
    df['f2_f1_ratio'] = -f_values[:, 1]
    
    for i in range(6):
        df[f'out_{i}_mean'] = [m[i] for m in all_means]
    
    os.makedirs('results', exist_ok=True)
    output_path = 'results/ood_candidates.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Optimization complete. {len(df)} candidates saved to {output_path}")

if __name__ == "__main__":
    main()
