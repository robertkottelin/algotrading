import os
import pandas as pd

input_dir = 'data/macrotechnicalfearandgreed/'
output_dir = 'data/macrotechnicalfearandgreedprepped/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_file(file_path):
    # Load the dataframe
    df = pd.read_csv(file_path)

    # Create a new column based on the comparison of 'Close' prices between rows
    # The new column will be '1' if the next row's 'Close' is higher, else '0'
    df['Next_Higher'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Add the last 5 days' close prices as new columns
    for i in range(1, 6):
        df[f'Close_{i}d_ago'] = df['Close'].shift(i)

    # Drop the rows where any of the new 'Close_Xd_ago' columns have NaN values
    df.dropna(subset=[f'Close_{i}d_ago' for i in range(1, 6)], inplace=True)

    return df

for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        print(f"Processing {filename}...")

        # Full path for input and output files
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        # Process the file
        new_df = process_file(input_file_path)

        # Save the updated dataframe to the new location
        new_df.to_csv(output_file_path, index=False)
        print(f"Processed {filename} and saved to {output_file_path}")

print("All files have been processed.")