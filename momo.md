# Project Analysis & Blueprint: Computational Design Framework (comp_design_fw)

## 1. Goal
Transition the mechanical design optimization project from a collection of research scripts to a **professional, one-click installable, and demoable framework**. The project focuses on using ML surrogate models to optimize mechanical designs, with the ability to verify ML predictions against ground-truth ANSYS FEA simulations.

## 2. System Architecture

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
│   ├── simulation/         # PyAnsys integration, Sweep logic, Project management
│   ├── core/               # ML Training, Inference, Uncertainty Estimation
│   ├── analysis/           # Optimization (NSGA-II), Manifold (UMAP), Sensitivity (Jacobian)
│   └── utils/              # Config loader, Logger, File system handlers
├── scripts/                # Execution Layer (CLI Tools)
│   ├── 01_generate_data.py # Param grid generation
│   ├── 02_run_simulations.py# ANSYS execution loop (with Resume capability)
│   ├── 03_train_model.py    # Ensemble training pipeline
│   └── 04_run_optimization.py# Design optimization execution
├── tests/                  # Pytest suite for verification
├── demo/                   # One-click demo assets (Notebooks/Streamlit)
├── requirements.txt        # Dependency list
└── README.md               # Documentation
```

---

## 3. Project Flow & Logic

### The Pipeline
`Config (JSON)` $\rightarrow$ `Data Generation` $\rightarrow$ `ANSYS Sweep (Ground Truth)` $\rightarrow$ `Data Processing` $\rightarrow$ `ML Model Training` $\rightarrow$ `Design Optimization` $\rightarrow$ `Verification`.

### Key Features
1. **Resume Capability**: The simulation runner checks `data/processed/` for existing entries. If a parameter set has a recorded result, it is skipped.
2. **Relative Pathing**: All paths are relative to the project root to ensure "one-click" portability across different machines.
3. **Dual-Mode Demo**:
    * **Production Mode**: Full pipeline execution for research.
    * **Verification Mode**: Targeted $N$-sample runs to verify the PyAnsys setup and ML-vs-ANSYS accuracy.
4. **Config-Driven**: No hard-coded parameters. Everything is controlled via `config.json` and `design_space.json`.

---

## 4. Implementation Roadmap

### Phase 1: Foundation
- [x] Setup directory structure.
- [x] Create `config.json` and `design_space.json` templates.
- [x] Implement `src.utils.config_loader` and `src.utils.logger`.
    - *Note: Verified via automated tests. `ConfigLoader` resolves project root automatically. `logger` uses `setup_logger()`.*

### Phase 2: Simulation Engine
- [ ] Refactor `ansys_runner.py` $\rightarrow$ `src/simulation/runner.py`.
- [ ] Refactor `sweep.py` $\rightarrow$ `src/simulation/sweep.py` (Add Resume logic).
- [ ] Implement `scripts/01_generate_data.py` (Grid & LHS).
- [ ] Implement `scripts/02_run_simulations.py`.

### Phase 3: ML & Analysis Core
- [ ] Refactor `model_build.py` $\rightarrow$ `src/core/trainer.py`.
- [ ] Refactor `predict.py` $\rightarrow$ `src/core/predictor.py`.
- [ ] Refactor `optimization.py` $\rightarrow$ `src/analysis/optimization.py`.
- [ ] Implement `scripts/03_train_model.py` and `scripts/04_run_optimization.py`.

### Phase 4: Demo & Verification
- [ ] Create a master `main.py` / `demo.py` for mode selection.
- [ ] Develop side-by-side ML vs ANSYS verification tool.
- [ ] Migrate verification tests to `tests/` using `pytest`.
- [ ] Finalize `README.md` and installation scripts.

# Tests writing

first read 
tests/test_foundation.py
tests/test_response.py
to understand test writing structure 

# Tools and hacks

## Lessons Learned (Momobot)
- **API Verification**: Never assume a function name (e.g., `get_logger` vs `setup_logger`). Always `read` the source before writing tests.
- **Constructor Logic**: Verify whether a class expects a file path or a root directory to avoid `FileNotFoundError` caused by path doubling (e.g., `ConfigLoader` expects project root, not the config file path).

## Powershell command to clear the __pycache__/

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

```cmd
for /d /r %d in (__pycache__) do @rd /s /q "%d"
```