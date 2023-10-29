import os
import pandas as pd

# Directories
input_directory = 'data/macrotechnical/'
fear_and_greed_file = 'data/fearandgreed/fearandgreed.csv'
output_directory = 'data/macrotechnicalfearandgreed/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load and process the fear and greed index data
fear_greed_data = pd.read_csv(fear_and_greed_file)

# Ensure 'Date' column in fear_greed_data is in datetime format
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])

# Process percentage columns
columns_to_update = ['Bullish', 'Neutral', 'Bearish']
for col in columns_to_update:
    if fear_greed_data[col].dtype == object:  # Check if the column is object type (possibly string)
        fear_greed_data[col] = fear_greed_data[col].str.rstrip('%')  # Remove '%' for strings
    # fear_greed_data[col] = pd.to_numeric(fear_greed_data[col], errors='coerce') / 100.0  # Convert to float and handle non-numeric gracefully

# Flag to indicate if any file was processed
any_file_processed = False

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        print(f"Processing {filename}...")  # Indicate the file being processed

        try:
            # File paths
            file_path = os.path.join(input_directory, filename)

            # Load macroeconomic data
            macro_data = pd.read_csv(file_path)

            # Check if 'Date' column exists in the macro_data
            if 'Date' not in macro_data.columns:
                print(f"Skipping {filename}: 'Date' column not found.")
                continue  # Skip this file

            # Convert 'Date' column to datetime
            macro_data['Date'] = pd.to_datetime(macro_data['Date'])

            # Check for date intersections between the two datasets
            common_dates = set(macro_data['Date']).intersection(set(fear_greed_data['Date']))
            
            if not common_dates:
                print(f"Skipping {filename}: No matching dates found.")
                continue  # Skip this file if there are no matching dates

            # Merging datasets on 'Date'
            merged_data = pd.merge(fear_greed_data, macro_data, on='Date', how='inner')

            # If the CSV has an index column not needed, remove it
            if 'Unnamed: 0' in merged_data.columns:
                merged_data.drop(columns=['Unnamed: 0'], inplace=True)

            # Saving to new CSV
            output_filename = os.path.splitext(filename)[0] + '_fearandgreed.csv'
            output_file_path = os.path.join(output_directory, output_filename)
            merged_data.to_csv(output_file_path, index=False)

            print(f"Merged and saved data for {filename}")
            any_file_processed = True  # Updating the flag

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Summary of the process
if any_file_processed:
    print("Processing completed. Some files were merged and saved.")
else:
    print("Processing completed. No files were merged due to lack of matching dates or other issues.")

