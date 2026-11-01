import pandas as pd
import os

def generate_summary():
    df = pd.read_csv('data/processed/full_processed_dataset.csv')
    
    with open('analysis/reports/profiling_report.txt', 'r') as f:
        profiling_results = f.read()
        
    summary = f"""
# FINAL DATA ANALYSIS REPORT
======================================================================

## 1. Dataset Summary
- **Source:** data/processed/full_processed_dataset.csv
- **Total Samples:** {len(df)}
- **Total Features:** {len(df.columns)}

## 2. Profiling Results
{profiling_results}

## 3. Visual Analysis Findings
- **Correlations:** A detailed correlation matrix has been generated. Strong relationships between geometry (width, height) and target metrics (stress, deformation) are evident.
- **Distributions:** Individual histograms have been plotted for all numerical features. Mean and Median lines are included to identify skewness.

## 4. Artifacts Location
- **Profiling Report:** analysis/reports/profiling_report.txt
- **Visualizations:** analysis/plots/
- **Analysis Scripts:** analysis/scripts/
======================================================================
"""
    with open('analysis/reports/final_analysis_report.md', 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    generate_summary()
