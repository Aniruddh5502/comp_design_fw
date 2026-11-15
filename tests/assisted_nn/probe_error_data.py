"""Probe: generate error mapper training data from Primary model predictions."""
from pathlib import Path
import sys
sys.path.insert(0, '.')

from utils.error_data import generate_error_data_loader
from utils.config import dataset_raw_file
from core.models import Primary
import torch
import pandas as pd
import numpy as np

def main():
    # Step 1: Get error dataloader
    print("="*60)
    print("Step 1: Generate error dataloader")
    print("="*60)
    result = generate_error_data_loader(raw_dataset_path=dataset_raw_file)
    print("Status:", result.get('status'))
    
    if not result.get('status'):
        print("ERROR: Could not generate error dataloader")
        return
    
    error_loader = result['content']['error']
    print(f"Error dataloader obtained, length: {len(error_loader)}")
    
    # Step 2: Load Primary model
    print("\n" + "="*60)
    print("Step 2: Load Primary model")
    print("="*60)
    primary = Primary()
    primary.load_state_dict(torch.load('model/primary.pt', map_location='cpu'))
    primary.eval()
    print("Primary model loaded and set to eval mode")
    
    # Step 3: Run predictions and collect data
    print("\n" + "="*60)
    print("Step 3: Run Primary predictions on error data")
    print("="*60)
    
    all_features = []
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, (features, targets) in enumerate(error_loader):
            features = features.to('cpu')
            targets = targets.to('cpu')
            
            # Primary prediction
            predictions = primary(features)
            
            # Compute error = actual - predicted
            error = targets - predictions
            
            all_features.append(features.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            all_error.append(error.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(error_loader)}...")
    
    # Concatenate all batches
    X = np.concatenate(all_features, axis=0)  # shape: (N, 4)
    y_true = np.concatenate(all_targets, axis=0)  # shape: (N, 6)
    y_pred = np.concatenate(all_predictions, axis=0)  # shape: (N, 6)
    error = np.concatenate(all_error, axis=0)  # shape: (N, 6)
    
    print(f"\nCollected data shapes:")
    print(f"  X (inputs): {X.shape}")
    print(f"  y_true: {y_true.shape}")
    print(f"  y_pred: {y_pred.shape}")
    print(f"  error: {error.shape}")
    
    # Step 4: Create training DataFrame
    print("\n" + "="*60)
    print("Step 4: Create training DataFrame")
    print("="*60)
    
    # Create column names
    feature_cols = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height']
    target_cols = ['max_stress_von_mises', 'max_deformation', 'mode_1_freq', 'mode_2_freq', 'mode_3_freq', 'mode_4_freq']
    error_cols = [f'error_{c}' for c in target_cols]
    
    df = pd.DataFrame(X, columns=feature_cols)
    df[target_cols] = y_true
    df[error_cols] = error
    
    # Add predicted error as well (this is what ErrorMapper would predict)
    df['pred_error_0'] = y_pred[:, 0]  # placeholder
    
    print(f"Training DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
    
    # Step 5: Save CSV
    output_path = Path('data') / 'error_mapper_training_data.csv'
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # Step 6: Quick stats
    print("\n" + "="*60)
    print("Step 6: Quick correlation stats")
    print("="*60)
    for i, tcol in enumerate(target_cols):
        corr = np.corrcoef(y_true[:, i], error[:, i])[0, 1]
        print(f"  {tcol}: correlation(true_error) = {corr:.4f}")

if __name__ == "__main__":
    main()