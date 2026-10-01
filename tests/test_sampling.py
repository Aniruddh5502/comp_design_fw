import sys
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
from .tests_response import TestResponse, TestResponses, create_test_response, create_test_responses

# Add the project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent))

def test_sampling() -> TestResponses:
    """Runs the sampling verification suite."""
    tests_report: List[TestResponse] = []
    test_name = "Sampling Verification"
    test_file = "test_sampling.py"

    mock_design_space = {
        "parameters": {
            "thickness": {"min": 0.1, "max": 2.0},
            "width": {"min": 10.0, "max": 50.0},
            "length": {"min": 100.0, "max": 500.0}
        }
    }

    # Test 1: Grid Sampling Accuracy
    try:
        from src.simulation.sampling import Sampler
        sampler = Sampler(mock_design_space)
        points = 5
        df = sampler.generate_grid(points_per_dim=points)
        
        # Expected shape: 5^3 = 125 rows, 4 columns (ID + 3 params)
        assert df.shape == (125, 4), f"Expected shape (125, 4), got {df.shape}"
        assert 'Design_ID' in df.columns
        assert df['thickness'].min() == 0.1
        assert df['thickness'].max() == 2.0
        
        tests_report.append(create_test_response(
            task_name="Grid Sampling Accuracy",
            status=True,
            error_type=None,
            error_message=None
        ))
    except Exception as e:
        tests_report.append(create_test_response(
            task_name="Grid Sampling Accuracy",
            status=False,
            error_type="SamplingError",
            error_message=f"ERROR: {e}"
        ))

    # Test 2: LHS Sampling Bounds and Uniqueness
    try:
        from src.simulation.sampling import Sampler
        sampler = Sampler(mock_design_space)
        num_samples = 10
        df = sampler.generate_lhs(num_samples=num_samples)
        
        assert df.shape == (10, 4), f"Expected shape (10, 4), got {df.shape}"
        assert df['Design_ID'].nunique() == 10, "Design IDs must be unique"
        
        for col in ['thickness', 'width', 'length']:
            assert df[col].min() >= mock_design_space['parameters'][col]['min']
            assert df[col].max() <= mock_design_space['parameters'][col]['max']
            
        tests_report.append(create_test_response(
            task_name="LHS Sampling Bounds and Uniqueness",
            status=True,
            error_type=None,
            error_message=None
        ))
    except Exception as e:
        tests_report.append(create_test_response(
            task_name="LHS Sampling Bounds and Uniqueness",
            status=False,
            error_type="SamplingError",
            error_message=f"ERROR: {e}"
        ))

    # Test 3: Sequential Design ID Generation
    try:
        from src.simulation.sampling import Sampler
        sampler = Sampler(mock_design_space)
        df = sampler.generate_lhs(5)
        assert list(df['Design_ID']) == [1, 2, 3, 4, 5], f"Expected [1,2,3,4,5], got {list(df['Design_ID'])}"
        
        tests_report.append(create_test_response(
            task_name="Sequential Design ID Generation",
            status=True,
            error_type=None,
            error_message=None
        ))
    except Exception as e:
        tests_report.append(create_test_response(
            task_name="Sequential Design ID Generation",
            status=False,
            error_type="IDError",
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
    
    res = test_sampling()
    console.print((json.dumps(res, indent=2)))
