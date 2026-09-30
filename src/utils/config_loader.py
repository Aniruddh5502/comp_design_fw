import json
from pathlib import Path

class ConfigLoader:
    """Utility to load and manage project configurations."""
    
    def __init__(self, root_dir=None):
        if root_dir is None:
            # Find the project root (where the 'config' folder lives) relative to this file
            current_file = Path(__file__).resolve()
            # Path: .../comp_design_fw/src/utils/config_loader.py
            # Root should be .../comp_design_fw/
            self.root_dir = current_file.parent.parent.parent
        else:
            self.root_dir = Path(root_dir).resolve()
            
        self.config_path = self.root_dir / "config" / "config.json"
        self.design_space_path = self.root_dir / "config" / "design_space.json"
        self._config = self._load_json(self.config_path)
        self._design_space = self._load_json(self.design_space_path)

    def _load_json(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {path}")
        except json.JSONDecodeError:
            raise json.JSONDecodeError("Failed to decode JSON config", path.name, 0)

    def get_config(self):
        """Returns the global config dictionary."""
        return self._config

    def get_design_space(self):
        """Returns the design space dictionary."""
        return self._design_space

    def update_pipeline_state(self, step, status):
        """Updates the state of a specific pipeline step."""
        self._config["pipeline_state"][step] = status
        self._save_config()

    def _save_config(self):
        """Saves the current state of config.json back to disk."""
        with open(self.config_path, 'w') as f:
            json.dump(self._config, f, indent=2)

# Singleton instance for easy import
config_manager = ConfigLoader()
