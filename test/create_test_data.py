import pandas as pd

# Create a test CSV with some missing values
data = {
    'Design_ID': [1, 2, 3, 4],
    'beam_width': [0.9, 1.6, None, 1.6], # Row 3 empty
    'beam_length': [17, 20, 23, None],    # Row 4 empty
    'status': ['completed', 'completed', 'completed', 'completed']
}
df = pd.DataFrame(data)
df.to_csv('test_inputs.csv', index=False)
print("Test CSV created successfully.")
