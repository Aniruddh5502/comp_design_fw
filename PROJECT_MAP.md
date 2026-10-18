# Project Architecture & Technical Map

## 1. System Components

### Configuration & Environment
- `config/design_space.json`: The "Physical Truth" file. It defines the strict geometric boundaries (min/max) for the design parameters. Any point outside these bounds is physically impossible.
- `config/config.json`: The "Global State" file. It manages sampling methods (LHS vs Grid), sample counts, and random seeds to ensure experiment reproducibility.
- `data/processed/`: The "Data Warehouse." It contains the deterministic 80/20 training/test splits and the scaling bounds used to normalize inputs.

### Processing & Training
- `scripts/process.py`: The data pipeline. It cleans raw data, applies Min-Max scaling to transform inputs to a $[-1, 1]$ range, and generates the fixed splits used to prevent data leakage.
- `scripts/train_ensemble.py`: The model factory. It trains a 20-member MLP ensemble. It uses a `torch.Generator` for each member to avoid "Deterministic Collapse," where all members learn identical weights.
- `scripts/run_convergence.py`: The stability auditor. It runs the trainer across various dataset sizes (300 to 1500) to prove that the model's accuracy ($\bar{R}_{test}$) is stable and not a result of a lucky seed.

### Inference & Optimization
- `scripts/predictor.py`: The interface between the model and the world. The `SurrogatePredictor` class implements a "Hard-Range Gate" to block non-physical inputs and handles the de-scaling of outputs back to physical units.
- `scripts/optimize_nsga2.py`: The "Adversarial" optimizer. It uses NSGA-II to seek designs that hit specific frequency targets. It is intentionally kept "blind" to ensemble disagreement to see if the model will "lie" (predict success where it is actually unsure).
- `scripts/curate_verification_set.py`: The experimental filter. It analyzes the optimizer's output and splits candidates into a "Confident Group" (Low CV) and an "Unsure Group" (High CV) for targeted ANSYS verification.

### Verification & Simulation
- `scripts/verify_ood_sim.py`: The ANSYS bridge. It transforms curated candidates into the format required by ANSYS Workbench, manages the simulation loop via `ansys.workbench.core`, and merges the actual simulation results with the surrogate's predictions.

## 2. Operational Pipeline

The project operates as a sequential value chain:

1. **Setup**: `config` $\rightarrow$ `process.py` (Defines and prepares the design space).
2. **Learning**: `train_ensemble.py` $\rightarrow$ `run_convergence.py` (Creates a stable, validated ensemble).
3. **Exploitation**: `predictor.py` $\rightarrow$ `optimize_nsga2.py` (Uses the model to find theoretically optimal designs).
4. **Falsification**: `curate_verification_set.py` $\rightarrow$ `verify_ood_sim.py` $\rightarrow$ `ANSYS` (Tests if the model's confidence matches physical reality).

## 3. Critical Technical Dependencies

- **The Predictor Dependency**: Both the Optimizer and the Curator depend on `predictor.py`. If the scaling logic or the Hard-Range Gate changes here, it automatically updates how the optimizer searches and how the curator filters.
- **The Simulation Link**: `verify_ood_sim.py` acts as a wrapper. It depends on `scripts.script_datagen.run_ansys` to handle the low-level Workbench communication and `ansys.workbench.core` to manage the software instance.
- **The Boundary Link**: The Optimizer reads `design_space.json` directly to ensure it never proposes a design that is geometrically impossible.
