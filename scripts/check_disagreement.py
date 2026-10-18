import pandas as pd
import numpy as np
from scripts.predictor import SurrogatePredictor

def main():
    predictor = SurrogatePredictor('models/ensemble_1500_42.pth')
    df = pd.read_csv('results/ood_candidates.csv')
    
    results = []
    # Using the exact columns from design_space.json
    input_cols = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height']
    
    for idx, row in df.iterrows():
        x = row[input_cols].values.astype(float)
        pred = predictor.predict(x)
        
        
        results.append({
            'idx': idx,
            'mean_f1': pred['mean_preds'][2],
            'std_f1': pred['std_preds'][2],
            'mean_f2': pred['mean_preds'][3],
            'std_f2': pred['std_preds'][3],
            'f2_f1_ratio': pred['mean_preds'][3] / (pred['mean_preds'][2] + 1e-6)
        })
    
    res_df = pd.DataFrame(results)
    
    print("=== F1 Distribution ===")
    print(res_df[['mean_f1', 'std_f1']].describe())
    
    print("\n=== F2 Distribution ===")
    print(res_df[['mean_f2', 'std_f2']].describe())
    
    print("\n=== Disagreement Analysis (Top 10 most 'extreme' ratios) ===")
    # These are the points the optimizer loves
    extreme = res_df.sort_values('f2_f1_ratio', ascending=False).head(10)
    print(extreme[['mean_f1', 'std_f1', 'mean_f2', 'std_f2', 'f2_f1_ratio']])
    
    # Check for "Hidden Failure": High ratio but LOW std_f1/std_f2
    # If std is low relative to the mean, the ensemble is confidently lying.
    res_df['cv_f1'] = res_df['std_f1'] / (res_df['mean_f1'].abs() + 1e-6)
    res_df['cv_f2'] = res_df['std_f2'] / (res_df['mean_f2'].abs() + 1e-6)
    
    print("\n=== Coefficient of Variation (Relative Uncertainty) ===")
    print(res_df[['cv_f1', 'cv_f2']].describe())

if __name__ == "__main__":
    main()
