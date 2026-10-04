from pathlib import Path
import json
from typing import Dict, Any
from ansys.workbench.core import workbench_client

class ParameterMapper:
    """
    Handles the mapping between logical parameter names used in Python 
    and the internal parameter names (e.g., P1, P2) used by ANSYS Workbench.
    """
    def __init__(self, map_file: str = "config/parameter_map.json"):
        self.map_path = Path(__file__).parent.parent.parent / map_file
        self.mapping = self._load_mapping()

    def _load_mapping(self) -> Dict[str, str]:
        if self.map_path.exists():
            try:
                with open(self.map_path, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_mapping(self, mapping: Dict[str, str]):
        """Saves the provided mapping to the JSON config file."""
        self.mapping = mapping
        with open(self.map_path, 'w') as f:
            json.dump(mapping, f, indent=4)

    def get_project_name(self, logical_name: str) -> str:
        """Returns the ANSYS parameter name (e.g., 'P1') for a given logical name."""
        if logical_name not in self.mapping:
            raise KeyError(f"Logical parameter '{logical_name}' not found in mapping file {self.map_path}")
        return self.mapping[logical_name]

    def get_all_mappings(self) -> Dict[str, str]:
        return self.mapping

def fetch_project_parameters(workbench: workbench_client) -> Dict[str, Dict[str, Any]]:
    """
    Interrogates the open ANSYS project to find all available parameters.
    Returns a dictionary where keys are DisplayText and values are parameter details.
    """
    script = """
params = {}
try:
    # Method 1: Global Parameters helper
    all_params = Parameters.GetAllParameters()
except:
    # Method 2: Fallback to Project.Parameters
    try:
        all_params = Project.Parameters.GetAllParameters()
    except:
        all_params = []

for p in all_params:
    params[p.DisplayText] = {
        "name": p.Name,
        "value": str(p.Value),
        "expression": p.Expression
    }
params
    """
    return workbench.run_script_string(script)
