import json
import os
import pandas as pd
from src.simulation.sampling import Sampler
from src.utils.logger import logger

def main():
    # 1. Load Configs
    with open('config/design_space.json', 'r') as f:
        design_space = json.load(f)
    
    with open('config/config.json', 'r') as f:
        config = json.load(f)

    data_gen_config = config['data_gen']
    
    # 2. Setup Sampler
    sampler = Sampler(design_space)
    
    # 3. Generate Points
    method = data_gen_config.get('sampling_method', 'LHS')
    if method == 'LHS':
        num_samples = data_gen_config.get('sample_count', 100)
        df = sampler.generate_lhs(num_samples)
    else:
        points_per_dim = 5 # Default grid size
        df = sampler.generate_grid(points_per_dim)

    # 4. Save Data
    output_path = 'data/raw/design_points.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated {len(df)} design points. Saved to {output_path}")

if __name__ == "__main__":
    main()
