import pandas as pd
from pathlib import Path

def generate_report(csv_path, report_path):
    df = pd.read_csv(csv_path)
    
    # Calculate basic stats
    stats = df.describe().to_markdown()
    
    # Identify key metrics
    # Based on the column names:
    # Constraint: lipschitz
    # Performance: rmse, spearman, ratio (both mean and val)
    
    report_content = f"""# Data Analysis Report: {Path(csv_path).name}

## 1. Dataset Overview
- **Total Rows**: {len(df)}
- **Total Columns**: {len(df.columns)}

## 2. Column Summary & Ranges
{stats}

## 3. Key Metrics for Visualization
To analyze the relationship between architectural constraints and performance, the following variables are identified:

### Architectural Constraint (Independent Variable)
- `lipschitz`: This is the primary constraint being swept (Range: {df['lipschitz'].min()} to {df['lipschitz'].max()}).

### Performance Metrics (Dependent Variables)
The following metrics should be plotted against the Lipschitz constant to evaluate model performance and generalization:

1. **Error Metrics**: 
   - `mean_rmse` (Training Root Mean Squared Error)
   - `val_mean_rmse` (Validation Root Mean Squared Error)
   - *Visualization Goal*: Observe if increasing the Lipschitz constant reduces error or leads to overfitting.

2. **Correlation Metrics**:
   - `mean_spearman` (Training Spearman Correlation)
   - `val_mean_spearman` (Validation Spearman Correlation)
   - *Visualization Goal*: Assess the monotonicity/trend preservation as the constraint relaxes.

3. **Ratio Metrics**:
   - `mean_ratio`
   - `val_mean_ratio`
   - *Visualization Goal*: Evaluate the stability or scale of the predictions relative to the target.

## 4. Data Integrity Audit
- **Null Values**: {df.isnull().sum().sum()}
- **Data Loss**: 0% (No records dropped during analysis).
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)

if __name__ == "__main__":
    csv_file = "tests/sngp_test/sweep_results.csv"
    report_file = "analysis/reports/sweep_analysis_report.md"
    generate_report(csv_file, report_file)
    print(f"Report generated at {report_file}")
