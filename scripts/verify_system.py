import torch
import numpy as np
import matplotlib.pyplot as plt
from hybrid_arch.core.model import setup_model
from hybrid_arch.core.trainer import SNGPTrainer
from hybrid_arch.utils.data_cleaning import load_and_clean_data

def run_final_verification():
    # 1. Data
    train_loader, val_loader, test_loader = load_and_clean_data()
    
    # 2. Model
    model, criterion, optimizer, device = setup_model(num_samples=len(train_loader.dataset))
    trainer = SNGPTrainer(model, criterion, optimizer, device, patience=20)
    
    # 3. Train
    trainer.train(train_loader, val_loader, test_loader, epochs=50)
    model.fit_gp(train_loader, device)
    
    # 4. Calibration Audit
    model.eval()
    X_test, Y_test = next(iter(test_loader))
    X_test, Y_test = X_test.to(device), Y_test.to(device)
    
    with torch.no_grad():
        mean, var = model.predict(X_test, device)
        
    # MSE for each sample: [B, O]
    actual_sq_error = (mean - Y_test)**2
    
    # Calibration Check: Average Actual Error vs Average Predicted Variance
    # If calibrated, Mean(Actual Error) should be approx Mean(Predicted Variance)
    print("\n--- Calibration Audit ---")
    for i in range(model.output_dim):
        avg_actual = actual_sq_error[:, i].mean().item()
        avg_pred = var[:, i].mean().item()
        print(f"Output {i} | Actual MSE: {avg_actual:.5f} | Pred Var: {avg_pred:.5f} | Ratio: {avg_actual/avg_pred:.3f}")

if __name__ == "__main__":
    run_final_verification()
