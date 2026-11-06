import torch
from pathlib import Path
from comp_design_fw.tests.sngp_test.core.model import setup_model
from comp_design_fw.tests.sngp_test.core.trainer import SNGPTrainer
from comp_design_fw.tests.sngp_test.utils.data_cleaning import load_and_clean_data

def main():
    # Config
    INPUT_DIM = 4
    OUTPUT_DIM = 6
    EPOCHS = 3000
    
    root = Path(__file__).parent.parent.parent.parent.parent
    data_folder = root / "data"
    data_csv_raw_file = data_folder / "raw" / "inputs.csv"
    
    # Data Pipeline (Clean -> Scale -> Split -> Load)
    try:
        train_loader, val_loader, test_loader = load_and_clean_data(
            csv_path=data_csv_raw_file,
            batch_size=32
        )
        print("Data loaded and cleaned successfully.")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Model
    train_samples = len(train_loader.dataset)
    model, criterion, optimizer, device = setup_model(
        input_dim=INPUT_DIM, 
        output_dim=OUTPUT_DIM, 
        num_samples=train_samples
    )
    
    # Trainer
    trainer = SNGPTrainer(model, criterion, optimizer, device, patience=20)
    
    # Train
    history, best_val_loss = trainer.train(train_loader, val_loader, epochs=EPOCHS)
    
    # Final Test Evaluation (Unbiased)
    trainer.model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = trainer.model(X)
            test_loss += criterion(outputs, y).item() * X.size(0)
    
    final_test_loss = test_loss / len(test_loader.dataset)
    
    # Save Model
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(model_dir / "sngp_best.pth")
    
    # Sync GP uncertainty matrix
    model.fit_gp(train_loader, device)
    
    print("\nTraining Complete.")
    print(f"Best Val Loss: {best_val_loss:.5f}")
    print(f"Final Test Loss: {final_test_loss:.5f}")

    # plotting the train and val loss
    import numpy as np
    import matplotlib.pyplot as plt
    train_loss = np.array(history["train_loss"])
    val_loss   = np.array(history["val_loss"])
    
    plt.rcParams["font.family"] = "Anthropic Serif Web Text"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 16
    
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Training Loss", c="#CC79A7", linewidth=2, marker='*', markersize=8)
    ax.plot(epochs, val_loss, label="Validation Loss", c="#009E73", linewidth=2, marker='.', markersize=8)
    
    ax.set_title("Tracking Training and Validation loss")
    ax.set_xlabel("Epocs")
    ax.set_ylabel("Loss")
    
    
    fig_path = Path(__file__).parent.parent/"plots"
    fig_path.mkdir(exist_ok=True)
    fig_path_file = fig_path / "loss_vs_val_plot.svg"
    
    plt.grid(alpha=0.4)
    plt.legend()
    plt.savefig(fig_path_file, bbox_inches='tight', pad_inches=0.25)
    plt.show()
    
if __name__ == "__main__":
    main()
