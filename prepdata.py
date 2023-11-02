import os
import pandas as pd

# Set the directories for NYSE and NASDAQ stocks
directories_to_update = ['data/nysestocks', 'data/nasdaqstocks']
output_dir = 'data/preppeddata/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_file(file_path):
    # Load the dataframe
    df = pd.read_csv(file_path)

    # Drop the 'Date' column if present
    if 'Date' in df.columns:
        df = df.drop(['Date'], axis=1)

    # Create a new column based on the comparison of 'Close' prices between rows
    # The new column will be '1' if the next row's 'Close' is higher, else '0'
    df['Next_Higher'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop the last row as it will have a NaN value in 'Next_Higher'
    df = df[:-1]

    return df

# Process all CSV files in the specified directories
for directory in directories_to_update:
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            print(f"Processing {filename}...")
            
            # Full path for input and output files
            input_file_path = os.path.join(directory, filename)
            output_file_path = os.path.join(output_dir, filename)

            # Process the file
            new_df = process_file(input_file_path)

            # Save the updated dataframe to the new location
            new_df.to_csv(output_file_path, index=False)
            print(f"Processed {filename} and saved to {output_file_path}")

print("All files have been processed.")
