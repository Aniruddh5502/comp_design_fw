# Data Analysis Report: sweep_results.csv

## 1. Dataset Overview
- **Total Rows**: 6
- **Total Columns**: 7

## 2. Column Summary & Ranges
|       |   lipschitz |   mean_rmse |   mean_spearman |   mean_ratio |   val_mean_rmse |   val_mean_spearman |   val_mean_ratio |
|:------|------------:|------------:|----------------:|-------------:|----------------:|--------------------:|-----------------:|
| count |     6       |  6          |       6         |     6        |      6          |           6         |         6        |
| mean  |     2.58333 |  0.0143941  |       0.205476  |     1.28692  |      0.0104152  |           0.205483  |         0.936582 |
| std   |     1.74404 |  0.00840904 |       0.0278462 |     0.931052 |      0.00569579 |           0.0321935 |         0.631469 |
| min   |     0.5     |  0.00778744 |       0.163454  |     0.563206 |      0.00614372 |           0.149483  |         0.458536 |
| 25%   |     1.25    |  0.00912804 |       0.193631  |     0.67696  |      0.00663625 |           0.199112  |         0.53412  |
| 50%   |     2.5     |  0.0095146  |       0.206416  |     0.762467 |      0.0071282  |           0.20648   |         0.564236 |
| 75%   |     3.75    |  0.0213768  |       0.217884  |     2.04958  |      0.0151781  |           0.225723  |         1.45008  |
| max   |     5       |  0.0252181  |       0.245491  |     2.50629  |      0.0177571  |           0.241657  |         1.76304  |

## 3. Key Metrics for Visualization
To analyze the relationship between architectural constraints and performance, the following variables are identified:

### Architectural Constraint (Independent Variable)
- `lipschitz`: This is the primary constraint being swept (Range: 0.5 to 5.0).

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
- **Null Values**: 0
- **Data Loss**: 0% (No records dropped during analysis).
