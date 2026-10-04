import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from ansys.workbench.core import launch_workbench
from src.simulation.parameters import fetch_project_parameters, ParameterMapper

console = Console()

def map_parameters():
    # 1. Setup Paths
    project_file = Path(__file__).parent.parent / "data" / "ansys_projects" / "parametric_file.wbpj"
    mapper = ParameterMapper()
    
    # 2. Launch and Open Project
    console.print("[bold blue]Launching ANSYS Workbench...[/]")
    workbench = launch_workbench()
    try:
        path_str = project_file.as_posix()
        workbench.run_script_string(f'Open(FilePath="{path_str}")')
        console.print(f"[green]Project opened successfully: {path_str}[/]\n")
    except Exception as e:
        console.print(f"[bold red]Failed to open project: {e}[/]")
        return

    # 3. Fetch all parameters from project
    console.print("[yellow]Interrogating project parameters...[/]")
    
    # Check if project is actually loaded
    project_status = workbench.run_script_string("Project.Name")
    console.print(f"Project Name: {project_status}")
    
    project_params = fetch_project_parameters(workbench)
    
    if not project_params:
        console.print("[bold red]No parameters found in the project![/]")
        console.print("Please ensure that your .wbpj file has defined parameters in the 'Parameter Set'.")
        return
    
    # Create a list of display names for the user to choose from
    display_names = list(project_params.keys())
    
    # 4. Define the logical names we need in our code
    required_logical_names = [
        "beam_width", 
        "beam_length", 
        "beam_height", 
        "fillet_radius",
        "max_stress_von_mises", 
        "max_deflection", 
        "modal_freq_1", 
        "modal_freq_2", 
        "modal_freq_3", 
        "modal_freq_4"
    ]
    
    final_mapping = {}
    
    # Table for visualization of available parameters
    table = Table(title="Available ANSYS Parameters")
    table.add_column("Index", justify="right", style="cyan")
    table.add_column("Display Text", style="magenta")
    table.add_column("Internal Name", style="green")
    
    for i, name in enumerate(display_names):
        table.add_row(str(i), name, project_params[name]["name"])
    
    console.print(table)
    console.print("\n[bold]Please map the logical names to the available parameters by entering the Index number.[/]\n")

    for logical in required_logical_names:
        while True:
            try:
                choice = input(f"Map [bold cyan]{logical}[/] to which index? (or 's' to skip): ")
                if choice.lower() == 's':
                    console.print(f"[yellow]Skipping {logical}...[/]")
                    break
                
                idx = int(choice)
                if 0 <= idx < len(display_names):
                    project_display_name = display_names[idx]
                    project_internal_name = project_params[project_display_name]["name"]
                    final_mapping[logical] = project_internal_name
                    console.print(f"[green]Mapped {logical} -> {project_internal_name} ({project_display_name})[/]\n")
                    break
                else:
                    console.print("[red]Index out of range. Try again.[/]")
            except ValueError:
                console.print("[red]Invalid input. Please enter a number or 's'.[/]")

    # 5. Save the mapping
    mapper.save_mapping(final_mapping)
    console.print(f"\n[bold green]Mapping saved successfully to {mapper.map_path}[/]")

if __name__ == "__main__":
    map_parameters()
