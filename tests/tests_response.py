from typing import List, TypedDict

# This TestResponse holds individual tasks execution details
class TestResponse(TypedDict):
    task_name: str              # what was this task checking
    status:bool                 # True = passed, False = Failed
    error_type:str|None         # If Error has occured then what type of error
    error_message:str|None      # If there are any messages that needs to be  

# This TestRespones holds a test files all tests data. what testing was the file doing
class TestResponses(TypedDict):
    test_file:str|None                       # Name of the test file
    test_name: str|None                      # What type of tests are we running
    test_report: List[TestResponse]     # List of execution report of all individual tests

class TestExecutionReport(TypedDict):
    time:str
    execution_details:List[TestResponses]    


def create_test_response(task_name:str, status:bool, error_type:str|None, error_message:str|None)->TestResponse:
    """This function creates and returns a single test response. following the class structure"""
    return {
        "task_name":task_name,
        "status":status,
        "error_type":error_type,
        "error_message":error_message,
    }

def create_test_responses(test_file:str, test_name:str, test_report:List[TestResponse])->TestResponses:
    """This function creates and returns total files test response and execution details"""
    return {
        "test_file":test_file,
        "test_name":test_name,
        "test_report":test_report,
    }