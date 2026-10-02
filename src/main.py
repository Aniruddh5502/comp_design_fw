import shutil
from rich.console import Console
from rich.markdown import Markdown

from config.config import theme_char, book_cloth, focus, error
from src.ui.ui_helpers import parse_user_input

con = Console()
state = ""

"""
This is the main orcastrator that will guide the professor in this project in which direction he wants to go. With like a terminal ui serial prints.
The flow view
1. Enter python -m src.main to run the program
2. It prompts with options 
    -   Dataset Generation (Generate dataset and run Ansys simulations)
    -   ML model Training  (Clean the dataset, and train the model on those settings)
    -   Run Forward Prediction (Run normal prediction adnd validate with Ansys)
    -   Run Optimizer and Validate the selected design point with Ansys
"""

def run():
    while True:
        # Getting terminal size
        size = shutil.get_terminal_size()
        columns = size.columns
        lines   = size.lines

        # If the user enters exit then exit the loop otherwise work within
        global state
        state = f"""
{"="*columns}
[bold green]OPTIONS[/]  [dim]Enter 'exit' or 'x' or 'c' to exit[/]\n
{theme_char}    setup_sim
{theme_char}    run_sim
{theme_char}    predict
{theme_char}    optimization
{"="*columns}
        """
    
    
        con.print(state)
        user_input = input("> ")
    
        parse_user_input(user_input)

if __name__ == "__main__":
    run()