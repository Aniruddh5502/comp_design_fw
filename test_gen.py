from scripts.user_input_sim import generate_input_dataset, config
import pandas as pd
import os

# Setup config for testing
config["data_gen"]["sampling_method"] = "LHS"
config["data_gen"]["sample_count"] = 10
config["data_gen"]["overwrite_raw"] = True
config["data_gen"]["seed"] = 42
config.save()

print("Running generate_input_dataset...")
res = generate_input_dataset()

if res["status"] == "success":
    path = res["path"]
    print(f"File created at {path}")
    df = pd.read_csv(path)
    print(f"DF shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if expected columns exist
    expected_cols = ["Design_ID", "status", "max_stress_von_mises", "max_deflection"]
    for col in expected_cols:
        if col not in df.columns:
            raise Exception(f"Missing column: {col}")
    
    print("Test Passed!")
else:
    print(f"Test Failed: {res}")
    exit(1)
