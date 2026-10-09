# Computational Design Framework 

A professional framework for mechanical design optimization using ML surrogate models and FEA verification.

## SPECS & SETTINGS

```text
Version                    :  2026R1 Student Edition
Geometry                   :  Ansys File data/ansys_projects.parametric_file.wbpj
Python Requirements        :  requirements.txt
Large Deflection           :  True
Static Structural Solver   :  Direct
Distributed Solver         :  False
Parameter Ranges           :  config.design_space.json
```

## Overview

This project aims to automate the simulation cycle and make input dataset generation easier with Grid/LHS sampling options. It allows users to run proposed workflows on their own machines and tweak parameters as perceived useful.

## Project Structure

### Directory Tree
```text
comp_design_fw/
├── config/                 # System & Design boundaries
│   ├── config.json         # Global paths, ML hyperparameters, Logging settings
│   ├── config.py           # Configuration management logic
│   ├── design_space.json   # Parameter ranges, Units, Sampling methods (Grid/LHS)
│   └── parameter_map.json  # Mapping between internal and Ansys parameters
├── data/                   # Data storage
│   ├── raw/                # Input parameter grids (CSV)
│   ├── ansys_projects/     # .wbpj / .db ANSYS project files (Versioned)
│   ├── processed/          # Final FEA results datasets for training
│   ├── backup/             # Simulation backups
│   └── params.json         # Runtime parameter snapshots
├── models/                 # Trained model artifacts (.pkl), Scalers
├── plots/                  # Visualization outputs for demo/papers
├── src/                    # Core Engine (Importable Modules)
│   ├── main.py             # Primary Entry Point (Orchestrator)
│   ├── simulation/         # PyAnsys integration, Sweep logic, Project management
│   ├── core/               # ML Training, Inference (Under Development)
│   ├── analysis/           # Optimization, Manifold, Sensitivity (Under Development)
│   ├── ui/                 # Terminal-based User Interface helpers
│   └── utils/              # Config loader, Logger, File system handlers
├── scripts/                # Execution Layer (CLI Tools)
│   ├── user_input_sim.py   # Unified Simulation Workflow (Setup & Run)
│   ├── script_datagen.py   # Script runner for Ansys Workbench
│   ├── map_parameters.py  # Parameter mapping utility
│   └── ansys_apis.md       # Documentation on Ansys API usage
├── tests/                  # Pytest suite for verification
├── demo/                   # One-click demo assets (Notebooks/Streamlit)
├── writings/               # Project documentation and research notes
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

The project features a terminal-based UI to guide the simulation and training process.

```bash
# Ensure you are in the project root
python -m src.main
```

**Current Functional Options in UI:**
- `setup_sim`: Configure sampling methods and generate the input dataset.
- `run_sim`: Execute the Ansys simulations and collect results.

**Note**: The `predict` and `optimization` options are currently under development and are not yet functional.

## Demo Modes
The framework is designed to support two primary usage patterns:
- **Production Mode**: Full sweep -> Full Train -> Optimization. (In development)
- **Verification Mode**: Run a small subset of simulations to compare ML predictions against ANSYS results side-by-side. (In development)

# RESEARCH
This work studies the computational optimization work on a Macro Scale Single Axis Flexure Geometry and its performance maping using surrogate models and an comparative analysis of different architectures use and their training data requirement for ranges of accuracy.

# The "SO WHAT"
The statement is, I have applied some known methods to this specific class of geometry and did analysis on how the training dataset size effects the performance compared to surrogate models with different architectures and which one configuration should get you where. I provide you the code for repurposing under a opensource lisence and the geometry and cad files along with the simulation files form ANSYS STUDENT version for academic use. I also provide the dataset on which the models were trained on so if needed these datapoints can be used to further the work.
---
*Developed by Ani (Aniruddho Biswas Badhon)*
