import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from rich.console import Console
import json

con = Console()

def run_command(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        con.print(f"[red]Command failed: {cmd}[/red]\n{result.stderr}")
        return None
    return result.stdout

def main():
    root = Path(__file__).parent.parent
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ensure data is split
    con.print("[yellow]Preparing data splits...[/yellow]")
    run_command("python scripts/process.py --mode split")
    
    # Create models directory
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Define convergence loop
    training_sizes = [300, 600, 900, 1200, 1500]
    seeds = [42, 123, 999] # 3 seeds to calculate sigma_R
    
    all_results = []
    
    for size in training_sizes:
        con.print(f"\n[bold blue]Evaluating Training Size: {size}[/bold blue]")
        size_rs = []
        
        for s in seeds:
            con.print(f"  Running seed {s}...")
            metrics_file = f"results/metrics_{size}_{s}.json"
            model_file = f"models/ensemble_{size}_{s}.pth"
            
            # Pass the seed and the save_model path
            cmd = f"python scripts/train_ensemble.py --seed {s} --train_file data/processed/train_{size}.csv --test_file data/processed/test_set.csv --save_metrics {metrics_file} --save_model {model_file}"
            run_command(cmd)
            
            with open(root / metrics_file, 'r') as f:
                data = json.load(f)
                size_rs.append(data['r_test'])
        
        mean_r = np.mean(size_rs)
        std_r = np.std(size_rs)
        all_results.append({"size": size, "mean_r": mean_r, "std_r": std_r})
        con.print(f"  [green]Size {size} -> Mean R: {mean_r:.4f}, Std: {std_r:.4f}[/green]")
    
    # 3. Save final convergence table
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / "convergence.csv", index=False)
    con.print(f"\n[bold green]Convergence Study Complete. Results saved to {results_dir / 'convergence.csv'}[/bold green]")

if __name__ == "__main__":
    main()
