from tests.tests_response   import create_test_response, create_test_responses, TestResponse, TestResponses
from ansys.tools.path       import find_ansys
from ansys.workbench.core   import launch_workbench
from tests.animation        import ThinkingAnimation
from rich.console           import Console
from pathlib                import Path
from config.config          import config

anim        = ThinkingAnimation()
console     = Console()
theme_char  = "✽ "

def test_ansys()->TestResponses:
    # initializing the total TestResponses veriable
    test_responses : list[TestResponse] = []
    test_name : str = "Test Ansys"
    test_file : str = "test_ansys.py"
    
    
    #================================================================================
    # Ansys Launching Test
    #================================================================================
    try:
        workbench = launch_workbench()
        
        server_port = workbench._server_port
        
        # Update the config variable for server port
        config.set("ansys_server_port", server_port)
        config.save()
        
        test_result_1 = create_test_response(
            task_name = "Ansys_Launch_Test",
            status=True,
            error_type=None,
            error_message=None
        )
        # append this test result in the test_resonses list
        test_responses.append(test_result_1)
    except Exception as e:
        test_result_1 = create_test_response(
            task_name="Ansys_Launch_Test",
            status=False,
            error_type="RunTimeError",
            error_message=f"Workbench launching failed. {e}"
        )
        test_responses.append(test_result_1)
    
    
    #================================================================================
    # Project File Availability Check
    #================================================================================
    try:
        # Adjusted root path logic to be more robust
        root = Path(__file__).parent.parent
        project_file = root / "data" / "ansys_projects" / "parametric_file.wbpj"
        
        if not project_file.exists():
            console.print(f"{theme_char} Project File Not Found: {project_file}")
            test_result_2 = create_test_response(
                task_name="File check",
                status=False,
                error_type="File Error",
                error_message=f"File not found: {project_file}"
            )
            test_responses.append(test_result_2)
        else:
            test_result_2 = create_test_response(
                task_name="File check",
                status=True,
                error_type=None,
                error_message=None
            )
            test_responses.append(test_result_2)    
    except Exception as e:
        test_result_2 = create_test_response(
            task_name="File Check",
            status=False,
            error_type="Unknown",
            error_message=f"Error occured in project file checking. {e}"
        )
        test_responses.append(test_result_2)
    
    #================================================================================
    # Ansys Script Running Test
    #================================================================================
    try:
        workbench.run_script_string(f"""
Open(FilePath=r"{project_file}")
""")
        test_result_3 = create_test_response(
            task_name="Script Running Test",
            status=True,
            error_type=None,
            error_message=None
        )
        test_responses.append(test_result_3)
    except Exception as e:
        test_result_3 = create_test_response(
            task_name="Script Running Test",
            status=False,
            error_type="Script Running Error",
            error_message=f"ERROR: {e}"
        )
        test_responses.append(test_result_3)
        
    return create_test_responses(
        test_name=test_name,
        test_file=test_file,
        test_report=test_responses,
    )

if __name__ == "__main__":
    # Simple runner to see results when executing as a script
    results = test_ansys()
    print(results)
