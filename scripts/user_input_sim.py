import json
import os, sys
import pandas as pd
import numpy as np
import time
from pathlib                    import Path
from typing                     import TypedDict, Literal
from rich.console               import Console
from rich.pretty                import Pretty
from rich.markdown              import Markdown
from rich.progress              import Progress
from src.simulation.sampling    import Sampler
from src.utils.logger           import logger
from ansys.workbench.core       import workbench_client, launch_workbench
from config.config              import theme_char, focus, error, book_cloth
from scripts.script_datagen     import run_ansys

# from scripts.script_datagen     import run_ansys
# Use a safe character if theme_char causes encoding issues in Windows terminal
# We use a standard ASCII character to avoid UnicodeEncodeError on Windows cp1252 consoles
SAFE_THEME_CHAR = theme_char

console = Console()

class Sampling_settings(TypedDict):
    sampling_method :   Literal["latin_hypercube", "grid"]|None   
    sample_counts   :   int|None                                    
    divisions       :   int|None                                 
    overwrite_raw   :   bool|None                                  

from config.config import ConfigManager

# the design space config file
design_space_config_path    =   Path(__file__).parent.parent / "config" / "design_space.json"
design_space_config         =   ConfigManager(default_file=design_space_config_path)
sampling_config_path        =   Path(__file__).parent.parent / "config" / "config.json"
config                      =   ConfigManager(default_file=sampling_config_path)

def mock_ansys_run(design_id: int, params: dict) -> dict:
    """
    Mocks an Ansys simulation run.
    In reality, this would write a .py or .wbjn script, call Ansys, and read results.
    """
    # Simulate computation time
    time.sleep(0.1) 
    
    # Generate deterministic-ish random values based on params to simulate physics
    # We use sum of params as a seed for the specific design
    seed = int(np.sum(list(params.values())) * 100) if params else 42
    np.random.seed(seed)
    
    return {
        "max_stress_von_mises": np.random.uniform(100, 500),
        "max_deflection": np.random.uniform(0.01, 0.5),
        "modal_freq_1": np.random.uniform(10, 100),
        "modal_freq_2": np.random.uniform(100, 500),
        "modal_freq_3": np.random.uniform(500, 1000),
        "modal_freq_4": np.random.uniform(1000, 2000),
        "status": "completed"
    }

def get_user_input()->dict:
    # 1. Sampling Method
    while True:
        info = f"\n\n{SAFE_THEME_CHAR} [bold {book_cloth}]Enter Sampling Method: [/]\n1 For Grid Method\n2 For Latin Hypercube\n0 To go back:\n"
        console.print(info)
        sampling_method  = input("> ")
        
        if sampling_method == '1':
            config["data_gen"]["sampling_method"] = "GRID"
            console.print(f"{SAFE_THEME_CHAR} Sampling method set to [green]Grid[/]\n")
            break
        elif sampling_method == '2':
            config["data_gen"]["sampling_method"] = "LHS"
            console.print(f"{SAFE_THEME_CHAR} Sampling method set to [green]Latin Hypercube Sampling[/]\n")
            break
        elif sampling_method == '0':
            console.print(f"{SAFE_THEME_CHAR} Exiting System")
            return
        else:
            console.print("[red]Invalid choice. Please enter 0, 1, or 2.[/]")

    # 2. Sample Count (Only relevant for LHS)
    if config["data_gen"]["sampling_method"] == "LHS":
        while True:
            try:
                count = int(input(f"{SAFE_THEME_CHAR} Enter Sample Count (e.g. 1000): "))
                config["data_gen"]["sample_count"] = count
                console.print(f"{SAFE_THEME_CHAR} Sample count set to [green]{count}[/]\n")
                break
            except ValueError:
                console.print("[red]Invalid input. Please enter an integer.[/]")

    # 3. Division (Only relevant for GRID)
    if config["data_gen"]["sampling_method"] == "GRID":
        while True:
            try:
                div = int(input(f"{SAFE_THEME_CHAR} Enter Number of Divisions per dimension: "))
                config["data_gen"]["division"] = div
                console.print(f"{SAFE_THEME_CHAR} Division set to [green]{div}[/]\n")
                break
            except ValueError:
                console.print("[red]Invalid input. Please enter an integer.[/]")

    # 4. Overwrite Raw Data
    while True:
        overwrite = input(f"{SAFE_THEME_CHAR} Overwrite existing raw data? (y/n): ").lower()
        if overwrite == 'y':
            config["data_gen"]["overwrite_raw"] = True
            console.print(f"{SAFE_THEME_CHAR} Overwrite set to [green]True[/]\n")
            break
        elif overwrite == 'n':
            config["data_gen"]["overwrite_raw"] = False
            console.print(f"{SAFE_THEME_CHAR} Overwrite set to [green]False[/]\n")
            break
        else:
            console.print("[red]Invalid choice. Please enter 'y' or 'n'.[/]")

    # 5. Random Seed
    while True:
        try:
            seed = int(input(f"{SAFE_THEME_CHAR} Enter Random Seed (default 42): ") or "42")
            config["data_gen"]["seed"] = seed
            console.print(f"{SAFE_THEME_CHAR} Seed set to [green]{seed}[/]\n")
            break
        except ValueError:
            console.print("[red]Invalid input. Please enter an integer.[/]")
    
    # Save the config to file
    config.save()
    console.print(f"\n{theme_char} [bold green]Configuration saved successfully![/]")
    return config

def generate_input_dataset()->dict:
    config.load(sampling_config_path)
    
    # Load design space
    design_space = design_space_config._data
    sampler = Sampler(design_space)
    
    method = config["data_gen"]["sampling_method"]
    seed = config["data_gen"].get("seed", 42)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    if method == "LHS":
        count = config["data_gen"]["sample_count"]
        df = sampler.generate_lhs(num_samples=count)
    elif method == "GRID":
        div = config["data_gen"]["division"]
        df = sampler.generate_grid(points_per_dim=div)
    else:
        console.print("[red]Unsupported sampling method.[/]")
        return {"status": "error", "message": "Unsupported sampling method"}

    # Define output columns for simulation results
    result_cols = [
        "max_stress_von_mises",
        "max_deformation",
        "mode_1_freq",
        "mode_2_freq",
        "mode_3_freq",
        "mode_4_freq"
    ]
    
    # Initialize status and results
    df["status"] = "pending"
    for col in result_cols:
        df[col] = np.nan

    # Resumability logic
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_file = raw_dir / "inputs.csv"
    
    if not config["data_gen"]["overwrite_raw"] and input_file.exists():
        console.print(f"{SAFE_THEME_CHAR} Existing input dataset found. Merging status...")
        existing_df = pd.read_csv(input_file)
        
        # Set index to Design_ID for alignment
        df.set_index('Design_ID', inplace=True)
        existing_df.set_index('Design_ID', inplace=True)
        
        # For results (which are numeric), we can use combine_first
        for col in result_cols:
            if col in existing_df.columns:
                df[col] = existing_df[col].combine_first(df[col])
        
        # For status, we prioritize existing_df
        df['status'] = existing_df['status'].fillna(df['status'])
        
        df.reset_index(inplace=True)
    
    # Save the dataset
    df.to_csv(input_file, index=False)
    
    console.print(f"\n\n[dim {book_cloth}]Input dataset generated/updated at: {input_file}[/]")
    console.print(f"{SAFE_THEME_CHAR} Total samples: [green]{len(df)}[/]")
    console.print(f"{SAFE_THEME_CHAR} Pending samples: [yellow]{len(df[df['status'] == 'pending'])}[/]")
    
    return {"status": "success", "path": str(input_file)}

def run_sim()->dict:
    # 1. Path check
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    input_file = raw_dir / "inputs.csv"
    
    if not input_file.exists():
        console.print(f"\n{SAFE_THEME_CHAR} [bold red]Input dataset not found![/]")
        console.print(f"{SAFE_THEME_CHAR} Please run [bold green]'setup_sim'[/] first to generate the design space.")
        return {"status": "error", "message": "No dataset found"}

    # 2. Load and Filter
    df = pd.read_csv(input_file)
    pending_df = df[df['status'] == 'pending']
    
    if pending_df.empty:
        console.print(f"\n{SAFE_THEME_CHAR} [bold green]All simulations are already complete![/]")
        
        # Ask for reset
        while True:
            choice = input(f"{SAFE_THEME_CHAR} Do you want to re-initialize the dataset and sampling settings? (y/n): ").lower()
            if choice == 'y':
                console.print(f"{SAFE_THEME_CHAR} [bold {error}]WARNING[/] This will overwrite your current dataset. Type [bold {focus}]'RESET'[/] to confirm: ")
                confirm = input(f"{SAFE_THEME_CHAR} : ")
                if confirm == "RESET":
                    console.print(f"{SAFE_THEME_CHAR} Triggering setup workflow...")
                    
                    # This handles settings and dataset generation
                    setup_sim() 
                    
                    console.print(f"{SAFE_THEME_CHAR} [green]Dataset re-initialized. You can now run 'run_sim' again.[/]")
                    
                    return {"status": "reset"}
                else:
                    console.print(f"{SAFE_THEME_CHAR} Confirmation failed. Reset aborted.")
                    break
            elif choice == 'n':
                break
        return {"status": "complete"}


    
    # 3. Execution Loop
    total_pending = len(pending_df)
    console.print(f"\n{SAFE_THEME_CHAR} Found [bold yellow]{total_pending}[/] pending simulations. Starting execution...\n")
    
    # Creating workbench instance and opening the file and passing the workbench instance
    workbench       = launch_workbench()
    project_file    = Path(__file__).parent.parent / "data" / "ansys_projects" / "parametric_file.wbpj"
    path_str        = project_file.as_posix()
    try:
        workbench.run_script_string(f"""Open(FilePath="{path_str}")""")
        console.print(f"{theme_char} File found and opened \n{path_str}")
    except Exception as e:
        console.print(f"{theme_char} [red]ERROR[/]: {e}")
        

    with Progress() as progress:
        task = progress.add_task("[cyan]Running Simulations...", total=total_pending)
        
        for index, row in pending_df.iterrows():
            # Extract parameters for this design (everything except status and result cols)
            result_cols = ["max_stress_von_mises", "max_deformation", "mode_1_freq", "mode_2_freq", "mode_3_freq", "mode_4_freq", "status", "Design_ID"]
            params = row[~row.index.isin(result_cols)].to_dict()

            # results = mock_ansys_run(int(row['Design_ID']), params)
            results = run_ansys(workbench=workbench, design_id=int(row['Design_ID']), params=params)
            
            # Update the main dataframe at the specific index
            for key, value in results.items():
                df.at[index, key] = value
            
            df.at[index, 'status'] = 'completed'
            
            # ATOMIC SAVE: Save the whole CSV after every design to ensure resumability
            df.to_csv(input_file, index=False)
            
            progress.update(task, advance=1)
            
    console.print(f"\n{SAFE_THEME_CHAR} [bold green]Simulations complete![/]\n")
    return {"status": "success"}

def setup_sim()->dict:
    get_user_input()
    generate_input_dataset()
    return {
        "status":"success"
    }

if __name__ == "__main__":
    # For testing purposes, we can bypass get_user_input() or run it.
    # In production, setup_sim() handles it.
    setup_sim()
