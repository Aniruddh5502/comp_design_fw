import json
import numpy as np
from pathlib import Path
from ansys.workbench.core import workbench_client
from src.simulation.parameters import ParameterMapper

# Initialize mapper globally to avoid reloading file every simulation
mapper = ParameterMapper()

def _to_float(value_str: str) -> float:
    """'3.29 [MPa]' -> 3.29"""
    return float(value_str.split()[0])

def run_ansys(workbench: workbench_client, design_id: int, params: dict) -> dict:
    """
    Set input parameters, solve, and read output parameters using a dynamic mapping.
    """
    # Opening the params.json file and reading the parameter names form there
    params_file = Path(__file__).parent.parent / "data" / "params.json"
    with open(params_file, 'r', encoding='utf-8') as f:
        params_display_texts = json.load(f)
    
    p_bw = params_display_texts["beam_width"]["name"]
    p_bl = params_display_texts["beam_length"]["name"]
    p_bh = params_display_texts["beam_height"]["name"]
    p_fr = params_display_texts["fillet_radius"]["name"]
    """
    # Resolve logical names to project names (e.g., 'beam_width' -> 'P1')
    # Input mapping
    p_bw = mapper.get_project_name("beam_width")
    p_bl = mapper.get_project_name("beam_length")
    p_bh = mapper.get_project_name("beam_height")
    p_fr = mapper.get_project_name("fillet_radius")
    """
    # Output mapping
    output_logical_names = [
        "max_stress_von_mises", 
        "max_deformation", 
        "mode_1_freq", 
        "mode_2_freq", 
        "mode_3_freq", 
        "mode_4_freq"
    ]
    
    output_project_names = [mapper.get_project_name(name) for name in output_logical_names]

    # Get current values
    bw = params["beam_width"]
    bl = params["beam_length"]
    bh = params["beam_height"]
    fr = params["fillet_radius"]

    # Literal { } inside the Workbench script must be doubled as {{ }}.
    # We use a list of names for the loop in IronPython to keep it clean.
    output_names_str = ", ".join([f"'{n}'" for n in output_project_names])

    script = f"""
import json

Parameters.GetParameter(Name="{p_bw}").Expression = "{bw} [mm]"
Parameters.GetParameter(Name="{p_bl}").Expression = "{bl} [mm]"
Parameters.GetParameter(Name="{p_bh}").Expression = "{bh} [mm]"
Parameters.GetParameter(Name="{p_fr}").Expression = "{fr} [mm]"

# Updating means running the solver
Update()

output = {{}}

for name in [{output_names_str}]:
    p = Parameters.GetParameter(Name=name)
    output[name] = {{
        "value": str(p.Value),
        "expression": p.Expression
    }}
wb_script_result = json.dumps(output)
    """
    try:
        result = workbench.run_script_string(script)
    except Exception as e:
        print(f"ERROR OCCURED:     [{e}]")
        return None
    
    
    # Map the internal results back to logical names
    final_results = {}
    all_nan = True
    for logical, project_name in zip(output_logical_names, output_project_names):
        if project_name in result and result[project_name]["value"] != "NaN":
            val = _to_float(result[project_name]["value"])
            final_results[logical] = val
            all_nan = False
        else:
            final_results[logical] = np.nan

    if all_nan:
        raise RuntimeError("Simulation failed to produce results (all outputs are NaN). Check Ansys solve status.")

    return final_results
