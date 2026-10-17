import json
from pathlib import Path

bounds_path = Path("data/processed/scaling_bounds.json")
with open(bounds_path, "r") as f:
    bounds = json.load(f)

print(json.dumps(bounds, indent=2))
