# Project Development & Implementation Guide

This document contains the technical implementation details, installation guides, and software architecture for the Computational Design Framework.

## Quick Start
### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Configuration
Edit the following files in `config/` to match your local ANSYS environment:
- `config.json`: Paths and hyperparameters.
- `design_space.json`: Parameter bounds and sampling method.

### 3. Execution Pipeline
The project features a terminal-based UI to guide the simulation and training process.
```bash
python -m src.main
```

**Functional Options in UI:**
- `setup_sim`: Configure sampling methods and generate the input dataset.
- `run_sim`: Execute the Ansys simulations and collect results.

## Project Structure
### Directory Tree
```text
comp_design_fw/
├── config/                 # System & Design boundaries
├── data/                   # Data storage (raw, processed, ansys_projects)
├── models/                 # Trained model artifacts (.pkl), Scalers
├── plots/                  # Visualization outputs
├── src/                    # Core Engine
│   ├── main.py             # Primary Entry Point
│   ├── simulation/         # PyAnsys integration, Sweep logic
│   ├── core/               # ML Training, Inference
│   ├── analysis/           # Optimization, Manifold, Sensitivity
│   ├── ui/                 # Terminal-based User Interface helpers
│   └── utils/              # Config loader, Logger
├── scripts/                # Execution Layer (CLI Tools)
├── tests/                  # Pytest suite
├── demo/                   # One-click demo assets
└── writings/               # Research notes and reports
```

## Current Functional State
- Parametric Geometry Making: Functional.
- Simulation setup and running: Functional.
- Data collection: 1897/2000 simulations successful.
- ML Core: Under development.
- Optimization: Under development.
