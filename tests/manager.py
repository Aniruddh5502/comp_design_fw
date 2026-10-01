import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List
from rich.console       import Console
from rich.markdown      import Markdown
from .animation         import ThinkingAnimation

theme_char  = "✽ "
console     = Console()
anim        = ThinkingAnimation()

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from tests.tests_response import TestExecutionReport

def run_tests():
    """
    Manager to run all test suites and compile a final report.
    """
    print("Starting Global Test Execution...\n")
    
    # List of test functions to run
    # Format: (module_path, function_name)
    test_suites = [
        ("tests.test_foundation",   "test_foundation"),
        ("tests.test_ansys",        "test_ansys"),
        ("tests.test_config",       "test_config_manager"),
        ("tests.test_sampling",     "test_sampling"),

        # ("module.file_name", "function_name")
        # Add future tests here:
        # ("tests.test_simulation", "test_simulation"),
        # ("tests.test_ml_core", "test_ml_core"),
    ]
    
    all_responses: List = []
    
    console.print(f"{theme_char} [bold green]Tests Starting.[/bold green]")
    
    for module_path, func_name in test_suites:
        anim.start()
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=[func_name])
            test_func = getattr(module, func_name)
            
            # Execute test
            response = test_func()
            all_responses.append(response)
            
            anim.stop()
            console.print(f"{theme_char} Module     :   {module_path:<30} Running [green][Done][/green]")
            
        except Exception as e:
            
            anim.stop()
            console.print(f"{theme_char} Module     :   {module_path:<30} Running [red][Failed][/red]")
                        
            all_responses.append({
                "test_file": module_path,
                "test_name": func_name,
                "test_report": [{
                    "task_name": "Module Execution",
                    "status": False,
                    "error_type": "ImportError/ExecutionError",
                    "error_message": str(e)
                }]
            })

    # Compile final report
    report: TestExecutionReport = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_details": all_responses
    }

    # Save to tests.json
    report_path = Path(__file__).parent / "tests.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nAll tests completed. Global report saved to: {report_path}")
    return report

if __name__ == "__main__":
    run_tests()
