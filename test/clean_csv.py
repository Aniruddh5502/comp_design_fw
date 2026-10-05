import pandas as pd
import sys

def clean_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        initial_completed = (df['status'] == 'completed').sum()
        
        # Identify rows that have at least one empty value AND status is 'completed'
        # isnull().any(axis=1) finds rows with at least one NaN
        mask = df.isnull().any(axis=1) & (df['status'] == 'completed')
        
        # Update those specific rows
        df.loc[mask, 'status'] = 'pending'
        
        final_completed = (df['status'] == 'completed').sum()
        updated_count = initial_completed - final_completed
        
        df.to_csv(file_path, index=False)
        print(f"Processed {file_path}: {updated_count} rows updated from 'completed' to 'pending'.")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_csv.py <file_path>")
        sys.exit(1)
    
    clean_csv(sys.argv[1])
