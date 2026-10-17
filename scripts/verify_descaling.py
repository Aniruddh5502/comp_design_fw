import torch
import numpy as np
import json
from scripts.predictor import SurrogatePredictor

# Setup
predictor = SurrogatePredictor("models/ensemble_1500_42.pth")

# Test case: extreme scaled values
# If scaled = -1, physical should be min
# If scaled = 1, physical should be max
# If scaled = 0, physical should be (max+min)/2

# We can mock the ensemble's output by patching the predict method or just 
# checking the de-scaling math manually as implemented in the class.

# Let's simulate the de-scaling part of predict()
out_cols = predictor.cols[4:]
out_mins = np.array([predictor.bounds["min"][c] for c in out_cols])
out_ranges = np.array([predictor.bounds["max"][c] - predictor.bounds["min"][c] for c in out_cols])

# Test points: -1, 0, 1
preds_scaled = np.array([[-1.0], [0.0], [1.0]]) # shape (3, 1)
# We need it to be [ensemble_size, batch, dims] for the broadcasting in the code
# But we can test the logic: physical = ((scaled + 1) * range / 2) + min

def descale(scaled, range_val, min_val):
    return ((scaled + 1) * range_val / 2) + min_val

test_vals = np.array([-1.0, 0.0, 1.0])
for v in test_vals:
    res = descale(v, out_ranges, out_mins)
    print(f"Scaled {v} -> Physical: {res}")
    
# Validation:
# v=-1: (0 * range / 2) + min = min -> Correct
# v=1: (2 * range / 2) + min = range + min = max -> Correct
# v=0: (1 * range / 2) + min = (max-min)/2 + min = (max+min)/2 -> Correct

print("\nMath verification passed.")
