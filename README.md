# Computational Design Framework 

A professional framework for mechanical design optimization using ML surrogate models and FEA verification.

## Overview
This framework automates the pipeline from design space definition to multi-objective optimization. It leverages **PyAnsys** for ground-truth simulations and an **Ensemble MLP** for rapid surrogate modeling, allowing researchers to find optimal designs without the computational cost of thousands of FEA runs.

## Project Structure
### Directory Tree
```text
comp_design_fw/
├── config/                 # System & Design boundaries (JSON)
│   ├── config.json         # Global paths, ML hyperparameters, Logging settings
│   └── design_space.json    # Parameter ranges, Units, Sampling methods (Grid/LHS)
├── data/
│   ├── raw/                # Input parameter grids (CSV)
│   ├── ansys_projects/     # .wbpj / .db ANSYS project files (Versioned)
│   └── processed/          # Final FEA results datasets for training
├── models/                 # Trained model artifacts (.pkl), Scalers
├── plots/                  # Visualization outputs for demo/papers
├── src/                    # Core Engine (Importable Modules)
│   ├── main.py              # Primary Entry Point (Orchestrator)
│   ├── simulation/         # PyAnsys integration, Sweep logic, Project management
│   ├── core/               # ML Training, Inference, Uncertainty Estimation
│   ├── analysis/           # Optimization (NSGA-II), Manifold (UMAP), Sensitivity (Jacobian)
│   └── utils/              # Config loader, Logger, File system handlers
├── scripts/                # Execution Layer (CLI Tools)
│   └── user_input_sim.py    # Unified Simulation Workflow (Setup & Run)
│   └── script_datagen.py     # Script runner. Takes Ansys workbench and runs the script
├── tests/                  # Pytest suite for verification
├── tests/                  # Pytest suite for verification
├── demo/                   # One-click demo assets (Notebooks/Streamlit)
├── requirements.txt        # Dependency list
└── README.md               # Documentation
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Edit the following files in `config/` to match your local ANSYS environment and design goals:
- `config.json`: Paths and hyperparameters.
- `design_space.json`: Parameter bounds and sampling method.

### 3. Execution Pipeline
Run the scripts in sequence to build the project:

1. **Generate Data** (`python scripts/01_generate_data.py`):
   - Generates sampling grids (Grid or Latin Hypercube) based on `design_space.json`.
   - **State Awareness**: Prompts user to [Skip] or [Regenerate] if `design_points.csv` already exists.

2. **Run Simulations** (`python scripts/02_run_simulations.py`):
   - Executes PyAnsys simulations for the generated design points.
   - **State Awareness**: Implements robust **Resume Logic**. Checks for existing results and prompts to simulate only the missing `Design_IDs`.
   - **Checkpointing**: Results are appended to `simulation_results.csv` after every single successful run to prevent data loss.

3. **Train Model** (`python scripts/03_train_model.py`):
   - Trains an Ensemble MLP surrogate model using the simulation results.
   - Saves the trained model and scalers to `models/`.

4. **Optimize Design** (`python scripts/04_run_optimization.py`):
   - Performs multi-objective optimization (e.g., NSGA-II) using the surrogate model.
   - Generates Pareto fronts and identifies optimal design candidates.

## Demo Modes
The framework supports two primary usage patterns:
- **Production Mode**: Full sweep -> Full Train -> Optimization.
- **Verification Mode**: Run a small subset of simulations to compare ML predictions against ANSYS results side-by-side.

---
*Developed by Ani (Aniruddho Biswas Badhon)*
