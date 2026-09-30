import sys
import json
from pathlib import Path
from typing import List
from tests.tests_response import TestResponse, TestResponses, create_test_response, create_test_responses

# Add the project root to sys.path to allow imports from src and config
sys.path.append(str(Path(__file__).parent.parent))

def test_config_manager() -> TestResponses:
    """Runs the ConfigManager verification suite."""
    tests_report: List[TestResponse] = []
    test_name = "Config Manager Verification"
    test_file = "test_config.py"

    # Setup test files
    test_file_path = Path("tests/test_config_temp.json")
    test_data = {"test_key": "test_value", "nested": {"a": 1}}
    with open(test_file_path, 'w') as f:
        json.dump(test_data, f)

    try:
        from config.config import ConfigManager, config
        
        # Test 1: Read existing file
        try:
            cm = ConfigManager(default_file=test_file_path)
            assert cm.get("test_key") == "test_value"
            assert cm["nested"]["a"] == 1
            tests_report.append(create_test_response(
                task_name="Read existing file",
                status=True, error_type=None, error_message=None
            ))
        except Exception as e:
            tests_report.append(create_test_response(
                task_name="Read existing file",
                status=False, error_type="ReadError", error_message=f"ERROR: {e}"
            ))

        # Test 2: Set and Save
        try:
            cm = ConfigManager(default_file=test_file_path)
            cm.set("new_key", "new_value")
            cm.save()
            
            with open(test_file_path, 'r') as f:
                disk_data = json.load(f)
            assert disk_data["new_key"] == "new_value"
            tests_report.append(create_test_response(
                task_name="Set and Save",
                status=True, error_type=None, error_message=None
            ))
        except Exception as e:
            tests_report.append(create_test_response(
                task_name="Set and Save",
                status=False, error_type="SaveError", error_message=f"ERROR: {e}"
            ))

        # Test 3: Update file direct
        try:
            cm = ConfigManager()
            cm.update_file(test_file_path, "direct_key", "direct_value")
            with open(test_file_path, 'r') as f:
                disk_data = json.load(f)
            assert disk_data["direct_key"] == "direct_value"
            tests_report.append(create_test_response(
                task_name="Update file direct",
                status=True, error_type=None, error_message=None
            ))
        except Exception as e:
            tests_report.append(create_test_response(
                task_name="Update file direct",
                status=False, error_type="UpdateError", error_message=f"ERROR: {e}"
            ))

        # Test 4: Global config access (Backward Compatibility)
        try:
            assert config.get("system") is not None
            assert config["system"]["project_root"] == "."
            tests_report.append(create_test_response(
                task_name="Global config access",
                status=True, error_type=None, error_message=None
            ))
        except Exception as e:
            tests_report.append(create_test_response(
                task_name="Global config access",
                status=False, error_type="CompatError", error_message=f"ERROR: {e}"
            ))

        # Test 5: Load different file
        try:
            cm = ConfigManager()
            ds_path = Path("config/design_space.json")
            assert cm.load(ds_path) is True
            assert cm._data is not None
            tests_report.append(create_test_response(
                task_name="Load different file",
                status=True, error_type=None, error_message=None
            ))
        except Exception as e:
            tests_report.append(create_test_response(
                task_name="Load different file",
                status=False, error_type="LoadError", error_message=f"ERROR: {e}"
            ))

        # Test 6: Create new file
        try:
            new_file = Path("tests/new_config_temp.json")
            cm = ConfigManager()
            cm.set("created", True)
            cm.save(new_file)
            assert new_file.exists()
            with open(new_file, 'r') as f:
                data = json.load(f)
            assert data["created"] is True
            new_file.unlink()
            tests_report.append(create_test_response(
                task_name="Create new file",
                status=True, error_type=None, error_message=None
            ))
        except Exception as e:
            tests_report.append(create_test_response(
                task_name="Create new file",
                status=False, error_type="CreateError", error_message=f"ERROR: {e}"
            ))

    except Exception as e:
        tests_report.append(create_test_response(
            task_name="Imports and Setup",
            status=False, error_type="ImportError", error_message=f"ERROR: {e}"
        ))

    # Cleanup
    if test_file_path.exists():
        test_file_path.unlink()

    return create_test_responses(
        test_file=test_file,
        test_name=test_name,
        test_report=tests_report
    )

if __name__ == "__main__":
    import json
    from rich.console import Console
    console = Console()
    
    res = test_config_manager()
    console.print((json.dumps(res, indent=2)))
