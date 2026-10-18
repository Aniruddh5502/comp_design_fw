import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('results/ood_candidates.csv')
    
    # We focus on f1 (out_2) as the primary anchor for 'confidence'
    # CV = std / abs(mean)
    df['cv_f1'] = df['out_2_std'] / (df['out_2_mean'].abs() + 1e-6)
    
    # Sort by CV to split into confident and unsure
    df_sorted = df.sort_values('cv_f1')
    
    # Group A: 50 most confident (Lowest CV)
    group_a = df_sorted.head(50).copy()
    group_a['group'] = 'confident'
    
    # Group B: 50 most unsure (Highest CV)
    group_b = df_sorted.tail(50).copy()
    group_b['group'] = 'unsure'
    
    # Combine into final verification list
    verification_df = pd.concat([group_a, group_b])
    
    # Keep only necessary columns for ANSYS and comparison
    cols_to_keep = [
        'beam_width', 'beam_length', 'fillet_radius', 'beam_height',
        'out_2_mean', 'out_2_std', 'out_3_mean', 'out_3_std',
        'group'
    ]
    
    final_df = verification_df[cols_to_keep]
    final_df.to_csv('results/ansys_verification_list.csv', index=False)
    
    print(f"Verification list created with {len(final_df)} points.")
    print(f"Confident range CV: {group_a['cv_f1'].min():.4f} to {group_a['cv_f1'].max():.4f}")
    print(f"Unsure range CV: {group_b['cv_f1'].min():.4f} to {group_b['cv_f1'].max():.4f}")

if __name__ == "__main__":
    main()
