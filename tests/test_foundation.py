import sys
from pathlib import Path
from typing import List
from tests.tests_response import TestResponse, TestResponses, create_test_response, create_test_responses

# Add the project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))

def test_foundation() -> TestResponses:
    """Runs the foundation verification suite."""
    tests_report: List[TestResponse] = []
    test_name = "Foundation Verification"
    test_file = "test_foundation.py"

    # Test 1: Config and Design space loading
    try:
        from src.utils.config_loader import config_manager
        from src.utils.logger import logger
        
        config = config_manager.get_config()
        design_space = config_manager.get_design_space()
        
        assert "pipeline_state" in config
        assert "parameters" in design_space
        
        tests_report.append(create_test_response(
            task_name="Config and Design space loading",
            status=True,
            error_type=None,
            error_message=None
        ))
    except Exception as e:
        tests_report.append(create_test_response(
            task_name="Config and Design space loading",
            status=False,
            error_type="ConfigError",
            error_message=f"ERROR: {e}"
        ))
    
    # Test 2: Pipeline state update
    try:
        from src.utils.config_loader import config_manager
        config_manager.update_pipeline_state("data_generation", "in_progress")
        new_config = config_manager.get_config()
        assert new_config["pipeline_state"]["data_generation"] == "in_progress"
        
        tests_report.append(create_test_response(
            task_name="Pipeline state update",
            status=True,
            error_type=None,
            error_message=None
        ))
    except Exception as e:
        tests_report.append(create_test_response(
            task_name="Pipeline state update",
            status=False,
            error_type="StateError",
            error_message=f"ERROR: {e}"
        ))

    # Test 3: Logger check
    try:
        from src.utils.logger import logger
        # logger.info("Testing logger in foundation suite")
        
        tests_report.append(create_test_response(
            task_name="Logger initialization",
            status=True,
            error_type=None,
            error_message=None
        ))
    except Exception as e:
        tests_report.append(create_test_response(
            task_name="Logger initialization",
            status=False,
            error_type="LoggerError",
            error_message=f"ERROR: {e}"
        ))

    return create_test_responses(
        test_file=test_file,
        test_name=test_name,
        test_report=tests_report
    )

if __name__ == "__main__":
    import json
    from rich.console import Console
    console = Console()
    
    res = test_foundation()
    console.print((json.dumps(res, indent=2)))
