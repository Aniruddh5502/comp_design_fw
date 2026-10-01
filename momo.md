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
│   ├── 01_generate_data.py # Param grid generation (with State Awareness)
│   ├── 02_run_simulations.py# ANSYS execution loop (with Resume & Checkpointing)
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
1. **State-Aware Data Generation**: `01_generate_data.py` prompts user to [Skip] or [Regenerate] if `design_points.csv` exists.
2. **Robust Simulation Resume**: `02_run_simulations.py` calculates missing `Design_IDs` and prompts to resume only the remaining simulations.
3. **Atomic Checkpointing**: Results are appended to `simulation_results.csv` immediately after every successful run to ensure zero data loss.
4. **Relative Pathing**: All paths are relative to the project root for "one-click" portability.
5. **Dual-Mode Demo**:
    * **Production Mode**: Full pipeline execution.
    * **Verification Mode**: Targeted $N$-sample runs to verify ML-vs-ANSYS accuracy.
6. **Config-Driven**: No hard-coded parameters. All controlled via `config.json` and `design_space.json`.

---

## 4. Implementation Roadmap

### Phase 1: Foundation
- [x] Setup directory structure.
- [x] Create `config.json` and `design_space.json` templates.
- [x] Implement `src.utils.config_loader` and `src.utils.logger`.

### Phase 2: Simulation Engine (Clean Rewrite)
- [x] Implement `src.simulation.sampling` (Grid & LHS).
- [ ] Implement `scripts/01_generate_data.py` (State-aware generation).
- [ ] Implement `src.simulation.runner` (PyAnsys interface).
- [ ] Implement `src.simulation.sweep` (Resume logic & atomic checkpointing).
- [ ] Implement `scripts/02_run_simulations.py` (CLI entry point).

### Phase 3: ML & Analysis Core
- [ ] Implement `src.core.trainer.py` (Ensemble MLP).
- [ ] Implement `src.core.predictor.py` (Inference engine).
- [ ] Implement `src.analysis.optimization.py` (NSGA-II).
- [ ] Implement `scripts/03_train_model.py` and `scripts/04_run_optimization.py`.

### Phase 4: Demo & Verification
- [ ] Create a master `main.py` / `demo.py` for mode selection.
- [ ] Develop side-by-side ML vs ANSYS verification tool.
- [ ] Migrate verification tests to `tests/` using `pytest`.
- [ ] Finalize `README.md` and installation scripts.

# Test Architecture Specification

The project uses a custom verification framework instead of raw pytest for structured reporting.

### 1. Test Structure
Every test file must follow this pattern:
- **Imports**: Import `TestResponse`, `TestResponses`, `create_test_response`, and `create_test_responses` from `tests.tests_response`.
- **Path Setup**: Add project root to `sys.path` to allow `src` imports.
- **The Suite Function**: A single function (e.g., `test_sampling()`) that:
    - Initializes a `tests_report: List[TestResponse] = []`.
    - Wraps each individual test case in a `try-except` block.
    - On Success: Appends `create_test_response(status=True, ...)`.
    - On Failure: Appends `create_test_response(status=False, error_type="...", error_message=f"ERROR: {e}")`.
- **Execution**: Returns `create_test_responses(...)`.
- **Main Block**: Uses `rich.console` to print the resulting JSON for readability.

### 2. Verification Flow
- Tests are executed as scripts: `python tests/test_module.py`.
- Pass/Fail is determined by the `status` boolean in the `test_report` list.

# Tools and hacks

## Lessons Learned (Momobot)
- **API Verification**: Never assume a function name. Always `read` the source before writing tests.
- **Constructor Logic**: Verify whether a class expects a file path or a root directory to avoid `FileNotFoundError`.

## Powershell command to clear the __pycache__/

```powershell
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

```cmd
for /d /r %d in (__pycache__) do @rd /s /q "%d"
```
