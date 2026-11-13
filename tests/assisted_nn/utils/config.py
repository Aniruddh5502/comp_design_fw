"""
Here we make all the paths, structures etc importable
Enverything will be imported from assisted_nn module
"""

from pathlib import Path

assisted_nn_root = Path(__file__).parent.parent

data_folder         =   assisted_nn_root / "data"
plots_folder        =   assisted_nn_root / "plots"
core_code_folder    =   assisted_nn_root / "core"
model_folder        =   assisted_nn_root / "model"

# files for dataset, scalers etc
dataset_raw_file    =   assisted_nn_root.parent.parent / "data" / "raw" / "inputs.csv"

# colors
accent  = "#D97757"
focus   = "#6A9BCC"
success = "#788C5D"
another = "#827DBD"
