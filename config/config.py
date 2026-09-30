import json
from pathlib import Path
from typing import Any, Optional
from rich.console import Console

console = Console()

class ConfigManager:
    """
    Manages configuration variables from JSON files.
    Allows reading, updating, and storing config variables.
    """
    def __init__(self, default_file: Optional[str] = None):
        self._data = {}
        self._current_file = Path(default_file) if default_file else None
        if self._current_file:
            self.load(self._current_file)

    def load(self, file_path: Any) -> bool:
        """Loads configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            console.print(f"[red]Error:[/red] Config file not found: {path}")
            return False
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._data = json.load(f)
            self._current_file = path
            return True
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to load config from {path}: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieves a value from the loaded configuration."""
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        """Updates a value in the loaded configuration (in-memory)."""
        self._data[key] = value

    def save(self, file_path: Optional[Any] = None) -> bool:
        """Saves the current configuration to a JSON file."""
        path = Path(file_path) if file_path else self._current_file
        if path is None:
            console.print("[red]Error:[/red] No file path provided to save configuration.")
            return False
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4)
            self._current_file = path
            return True
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to save config to {path}: {e}")
            return False

    def update_file(self, file_path: Any, key: str, value: Any) -> bool:
        """Updates a specific key in a specific file without affecting current manager state."""
        path = Path(file_path)
        data = {}
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to read file for update {path}: {e}")
                return False
        
        data[key] = value
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to update file {path}: {e}")
            return False

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.set(key, value)

    def __repr__(self):
        return f"ConfigManager(file={self._current_file}, data={self._data})"

# Backward compatibility: create a global config instance
_default_config_path = Path(__file__).parent / "config.json"
config = ConfigManager(default_file=_default_config_path)

if __name__ == "__main__":
    # --- Example Usage ---
    console.print("[bold cyan]Running ConfigManager Examples...[/bold cyan]\n")

    # 1. Using the global config instance (Backward Compatibility)
    console.print("1. [yellow]Global Config Access:[/yellow]")
    # Reading a value
    val = config.get("system", "Default Value")
    console.print(f"   - Value for 'system': {val}")
    
    # Updating a value and saving
    config.set("last_run", "2023-10-27")
    config.save()
    console.print("   - Updated 'last_run' and saved to config.json")

    # 2. Creating a custom ConfigManager instance
    console.print("\n2. [yellow]Custom ConfigManager instance:[/yellow]")
    custom_file = Path("example_settings.json")
    cm = ConfigManager(default_file=custom_file)
    
    # Setting values (in-memory)
    cm["api_key"] = "XYZ-123-ABC"
    cm["timeout"] = 30
    
    # Saving to file
    cm.save()
    console.print(f"   - Created {custom_file} with api_key and timeout")

    # 3. Direct file update (without loading full manager)
    console.print("\n3. [yellow]Direct File Update:[/yellow]")
    cm.update_file("example_settings.json", "version", "1.0.0")
    console.print("   - Directly updated 'version' in example_settings.json")

    # 4. Reading back and validating
    cm.load("example_settings.json")
    console.print(f"   - Loaded value for 'version': {cm['version']}")
    
    # Cleanup example file
    if custom_file.exists():
        custom_file.unlink()
    
    console.print("\n[bold green]Examples completed successfully![/bold green]")
