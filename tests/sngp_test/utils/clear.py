import shutil
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown

con = Console()

def clean_folder(root_path:Path)->dict:
    if root_path == None:
        return {
            "status":False,
            "content":"Root Path not provided."
        }
    cache_folders = root.rglob("__pycache__")
    for cache_folder in cache_folders:
        shutil.rmtree(cache_folder)
        con.print(f"[dim]Removed: {cache_folder}[/]\n")
    return {
        "status":True,
        "content":"removed all the __pycache__/ folders."
    }
    
if __name__=="__main__":
    root = Path(__file__).parent.parent.parent.parent.parent
    clean_folder(root_path=root)