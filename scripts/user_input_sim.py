import json
import os, sys
import pandas as pd
import numpy as np

from pathlib                    import Path
from typing                     import TypedDict, Literal
from rich.console               import Console
from rich.pretty                import Pretty
from rich.markdown              import Markdown
from src.simulation.sampling    import Sampler
from src.utils.logger           import logger
from config.config              import theme_char, focus, error, book_cloth

# Use a safe character if theme_char causes encoding issues in Windows terminal
SAFE_THEME_CHAR =  theme_char

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
            break
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
        "max_deflection",
        "modal_freq_1",
        "modal_freq_2",
        "modal_freq_3",
        "modal_freq_4"
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
