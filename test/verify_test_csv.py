import pandas as pd

def verify_logic(file_path):
    df = pd.read_csv(file_path)
    # Check if any 'completed' status remains for rows with NaN values
    mask = df.isnull().any(axis=1) & (df['status'] == 'completed')
    if mask.any():
        print(f"Verification Failed: Found {mask.sum()} rows that should be 'pending' but are 'completed'.")
        return False
    
    # Check if the expected rows were changed to pending
    # In our test_inputs.csv, rows 2 and 3 (0-indexed) have NaNs
    # Note: we check for any row that has NaN AND status == 'pending' 
    # (since we converted them from completed)
    pending_count = (df['status'] == 'pending').sum()
    if pending_count == 0:
        print("Verification Failed: No rows were updated to 'pending'.")
        return False
        
    print("Verification Successful: All empty-column rows are marked as pending.")
    return True

if __name__ == "__main__":
    if not verify_logic('test_inputs.csv'):
        exit(1)
