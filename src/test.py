from ansys.tools.path import find_ansys
from pathlib import Path
from rich.console import Console
import re

console = Console()
theme_char = "✽ "


#======================================================================================
#   Launcing workbench, open the file and start the server                            |
#======================================================================================
from ansys.workbench.core import launch_workbench
try:
    workbench = launch_workbench()
    console.print(f"Workbench launched  :   {workbench}")
    console.print(f"Server Port         :   {workbench._server_port}")
    console.print(f"Server Version      :   {workbench._server_version}")
except Exception as e:
    print(f"ERROR: {e}")
    
try:
    returned = workbench.run_script_string(r"""
import json
wb_script_result = json.dumps(
    GetTemplate(TemplateName="Static Structural (ANSYS)").CreateSystem().Name
)
""")
    console.print(f"Script execution succesfully ran.")
    console.print(f"Returned: {returned}")
except Exception as e:
    print(f"Script execution failed.")

try:
    # try opening the project file
    root = Path(__file__).parent.parent
    project_file = root / "data" / "ansys_projects" / "parametric_file.wbpj"
    
    console.print(f"{theme_char}[bold]Opening the {project_file}[/]")
    console.print(f"Running file opening script...")
    
    workbench.run_script_string(f"""
Open(FilePath=r"{project_file}")
""")
    console.print(f"{theme_char}[bold]Script ran succesfully[/] [green]DONE[/]")
except Exception as e:
    console.print()