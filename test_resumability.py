from scripts.user_input_sim import generate_input_dataset, config
import pandas as pd
import numpy as np
import os

def test_resumability():
    # 1. First run: Generate 10 samples
    config["data_gen"]["sampling_method"] = "LHS"
    config["data_gen"]["sample_count"] = 10
    config["data_gen"]["overwrite_raw"] = True
    config["data_gen"]["seed"] = 42
    config.save()
    
    print("Run 1: Generating initial dataset...")
    generate_input_dataset()
    
    path = "data/raw/inputs.csv"
    df1 = pd.read_csv(path)
    
    # Mock some results
    df1.loc[0, 'status'] = 'completed'
    df1.loc[0, 'max_stress_von_mises'] = 123.45
    df1.to_csv(path, index=False)
    print("Mocked one result as 'completed'.")

    # 2. Second run: Non-overwrite, same count
    config["data_gen"]["overwrite_raw"] = False
    config.save()
    print("Run 2: Generating with overwrite_raw=False...")
    generate_input_dataset()
    
    df2 = pd.read_csv(path)
    
    # Check if the mocked result is preserved
    if df2.loc[0, 'status'] == 'completed' and df2.loc[0, 'max_stress_von_mises'] == 123.45:
        print("Resumability Verified: Status and results preserved!")
    else:
        print("Resumability Failed: Data lost!")
        exit(1)

if __name__ == "__main__":
    test_resumability()
