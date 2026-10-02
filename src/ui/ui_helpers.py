import sys
from rich.console import Console
from rich.markdown import Markdown
from scripts.user_input_sim import setup_sim

con = Console()

def parse_user_input(user_input:str)->dict:
    if user_input in ["exit", "x", "c"]:
        sys.exit()
    
    elif user_input == "setup_sim":
        setup_sim()
    
    elif user_input == "run_sim":
        con.print("This one is a demo right now")
        con.print("This command should run/resume the simulations")
        input_ = input("> Enter 'k' to go back : ")
        if input_ == "k":
            pass
    else:
        con.print(f"[dim]Input is Case sensitive. \nYou might have entered something wrong[/]")
        