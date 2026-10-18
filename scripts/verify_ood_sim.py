import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

# Ensure root is in path for scripts import
sys.path.append('.')

from ansys.workbench.core import launch_workbench
from scripts.script_datagen import run_ansys

console = Console()

def main():
    # 1. Paths
    root = Path(__file__).parent.parent
    verify_list_path = root / "results" / "ansys_verification_list.csv"
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    verify_inputs_path = raw_dir / "verify_inputs.csv"
    
    if not verify_list_path.exists():
        console.print(f"[red]Error: {verify_list_path} not found. Run curation first.[/]")
        return

    # 2. Adaptation: Convert Verification List to Workbench Format
    console.print("[yellow]Adapting verification list to Workbench format...[/]")
    df_verify = pd.read_csv(verify_list_path)
    
    # The actual parameters for ANSYS
    input_cols = ['beam_width', 'beam_length', 'fillet_radius', 'beam_height']
    
    # Create a new dataframe that mimics inputs.csv
    # We use Design_ID as the anchor for the simulation loop
    df_sim = pd.DataFrame()
    df_sim['Design_ID'] = range(1, len(df_verify) + 1)
    
    for col in input_cols:
        df_sim[col] = df_verify[col]
    
    # Result columns expected by the results merge logic later
    result_cols = [
        "max_stress_von_mises",
        "max_deformation",
        "mode_1_freq",
        "mode_2_freq",
        "mode_3_freq",
        "mode_4_freq"
    ]
    
    for col in result_cols:
        df_sim[col] = np.nan
    
    df_sim['status'] = 'pending'
    
    # Save as the active input file for the simulation loop
    df_sim.to_csv(verify_inputs_path, index=False)
    console.print(f"[green]Adapter complete. Input file created at: {verify_inputs_path}[/]")

    # 3. Simulation Execution
    pending_df = df_sim[df_sim['status'] == 'pending']
    total_pending = len(pending_df)
    
    if total_pending == 0:
        console.print("[green]No pending simulations found.[/]")
        return

    console.print(f"\n[bold blue]Starting ANSYS Verification Loop for {total_pending} points...[/]")
    
    # Launch Workbench
    workbench = launch_workbench()
    project_file = root / "data" / "ansys_projects" / "parametric_file.wbpj"
    try:
        workbench.run_script_string(f"Open(FilePath=\"{project_file.as_posix()}\")")
        console.print(f"[green]Project opened: {project_file}[/]")
    except Exception as e:
        console.print(f"[red]Critical Error opening project: {e}[/]")
        return

    with Progress() as progress:
        task = progress.add_task("[cyan]Simulating...", total=total_pending)
        
        for index, row in df_sim.iterrows():
            if row['status'] == 'completed':
                progress.update(task, advance=1)
                continue
                
            # Extract params (only input columns)
            params = row[input_cols].to_dict()
            
            try:
                # Call the core simulation function from script_datagen
                results = run_ansys(workbench=workbench, design_id=int(row['Design_ID']), params=params)
                
                if results:
                    for key, value in results.items():
                        if key in df_sim.columns:
                            df_sim.at[index, key] = value
                
            except Exception as e:
                console.print(f"[red]Design {row['Design_ID']} failed: {e}[/]")
            
            df_sim.at[index, 'status'] = 'completed'
            # Atomic save for resumability
            df_sim.to_csv(verify_inputs_path, index=False)
            progress.update(task, advance=1)

    # 4. Final Merge
    # Merge the simulated results back into the verification list (with the Group/Predicted info)
    df_final = df_verify.merge(df_sim[['Design_ID'] + result_cols], left_index=True, right_index=True)
    df_final.to_csv(root / "results" / "ansys_verification_results.csv", index=False)
    
    console.print(f"\n[bold green]Verification complete! Results saved to results/ansys_verification_results.csv[/]")

if __name__ == "__main__":
    main()
