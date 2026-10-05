import pandas as pd
import os

def inspect_csv(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"--- File Information: {file_path} ---")
    print(f"File Size: {os.path.getsize(file_path)} bytes")
    
    # Read only the header and a few rows to get metadata without loading the whole file
    # chunksize=1 allows us to get the header and first row efficiently
    reader = pd.read_csv(file_path, chunksize=1)
    for chunk in reader:
        df_head = chunk
        headers = df_head.columns.tolist()
        
        print("\n--- Headlines (Columns) ---")
        for i, col in enumerate(headers, 1):
            print(f"{i}. {col}")
        
        # To get the total number of rows without reading everything into memory, 
        # we can count the lines in the file.
        with open(file_path, 'r') as f:
            row_count = sum(1 for line in f)
        
        print(f"\nTotal Rows (including header): {row_count}")
        print(f"Total Columns: {len(headers)}")
        
        # Get data types of columns from the first chunk
        print("\n--- Column Data Types (Inferred from first row) ---")
        print(df_head.dtypes)
        
        break # We only need the first chunk for metadata

if __name__ == "__main__":
    inspect_csv('inputs.csv')
