import os
import pandas as pd

# Set the directories
input_dir = 'data/macrotechnicalfearandgreed/'
output_dir = 'data/preppeddata/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process_file(file_path):
    # Load the dataframe
    df = pd.read_csv(file_path)

    # Drop the 'Date' column
    if 'Date' in df.columns:
        df = df.drop(['Date'], axis=1)

    # Create a new column based on the comparison of 'Close' prices between rows
    # The new column will be '1' if the next row's 'Close' is higher, else '0'
    df['Next_Higher'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # Drop the last row as it will have a NaN value in 'Next_Higher'
    df = df[:-1]

    return df

# Process all CSV files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        # Create the full file paths
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        # Process the file
        new_df = process_file(input_file_path)

        # Save to new CSV file in the output directory
        new_df.to_csv(output_file_path, index=False)
        print(f"Processed {filename} and saved to {output_file_path}")