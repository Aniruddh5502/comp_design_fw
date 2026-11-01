import pandas as pd
import numpy as np
import os

def profile_csv(file_path):
    df = pd.read_csv(file_path)
    
    # Basic Stats
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Nulls
    null_counts = df.isnull().sum()
    null_pct = (null_counts / total_rows) * 100
    
    # Numerical Analysis
    num_df = df.select_dtypes(include=[np.number])
    stats = num_df.describe().T
    stats['null_count'] = null_counts[num_df.columns]
    stats['null_pct'] = null_pct[num_df.columns]
    
    # Categorical Analysis
    cat_df = df.select_dtypes(exclude=[np.number])
    cat_stats = []
    for col in cat_df.columns:
        cat_stats.append({
            'column': col,
            'unique_values': df[col].nunique(),
            'top_value': df[col].mode()[0] if not df[col].mode().empty else np.nan,
            'null_count': null_counts[col],
            'null_pct': null_pct[col]
        })
    cat_stats_df = pd.DataFrame(cat_stats)

    return total_rows, total_cols, stats, cat_stats_df

def generate_markdown_report(total_rows, total_cols, stats, cat_stats_df, output_path):
    with open(output_path, 'w') as f:
        f.write("# Dataset Profiling Report\n\n")
        f.write(f"## Overview\n- **Total Rows:** {total_rows}\n- **Total Columns:** {total_cols}\n\n")
        
        f.write("## Numerical Columns\n")
        f.write(stats.to_markdown() + "\n\n")
        
        f.write("## Categorical Columns\n")
        if not cat_stats_df.empty:
            f.write(cat_stats_df.to_markdown(index=False) + "\n\n")
        else:
            f.write("No categorical columns found.\n\n")

if __name__ == "__main__":
    input_csv = 'data/processed/full_processed_dataset.csv'
    output_report = 'analysis/reports/profiling_report.md'
    
    try:
        rows, cols, num_stats, cat_stats = profile_csv(input_csv)
        generate_markdown_report(rows, cols, num_stats, cat_stats, output_report)
        print(f"Report successfully saved to {output_report}")
    except Exception as e:
        print(f"Error profiling CSV: {e}")
