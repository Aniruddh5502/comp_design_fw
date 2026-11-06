import pandas as pd
import os

def generate_report(csv_path, output_path):
    df = pd.read_csv(csv_path)
    
    # The user wants unique lipschitz and corresponding values.
    # Since the columns are already 'mean_...', we assume one row per lipschitz.
    # We sort by lipschitz to ensure the table is organized.
    df_sorted = df.sort_values('lipschitz')
    
    cols_to_keep = ['lipschitz', 'mean_rmse', 'mean_spearman', 'mean_ratio', 'val_mean_rmse', 'val_mean_spearman', 'val_mean_ratio']
    result = df_sorted[cols_to_keep]
    
    # Convert to markdown table
    markdown_table = result.to_markdown(index=False, floatfmt=".18g")
    
    report_content = f"## Sweep Results Analysis\n\n{markdown_table}\n\n**Data Loss Audit:**\n- Initial Row Count: {len(df)}\n- Final Row Count: {len(result)}\n- Percentage Lost: 0%\n- Justification: No rows dropped."
    
    with open(output_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report generated at {output_path}")

if __name__ == "__main__":
    csv_file = "tests/sngp_test/sweep_results.csv"
    output_file = "analysis/reports/sweep_analysis.md"
    generate_report(csv_file, output_file)
