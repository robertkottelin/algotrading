import os
import pandas as pd

# Load and process the fear and greed index data
fear_and_greed_file = 'data/fearandgreed/fearandgreed.csv'
fear_greed_data = pd.read_csv(fear_and_greed_file)
fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])

# Process percentage columns if necessary (uncomment if required)
# columns_to_update = ['Bullish', 'Neutral', 'Bearish']
# for col in columns_to_update:
#     if fear_greed_data[col].dtype == object:
#         fear_greed_data[col] = fear_greed_data[col].str.rstrip('%').astype(float) / 100

# Directories to update
input_directory = 'data/macrotechnical/'
directory_to_save_to = 'data/macrotechnicalfearandgreed/'

for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):  # Only process CSV files
        print(f"Processing {filename}...")
        file_path = os.path.join(input_directory, filename)

        try:
            # Load the stock data
            stock_data = pd.read_csv(file_path)
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])

            # Merge the fear and greed index data with the stock data
            merged_data = pd.merge(stock_data, fear_greed_data, on='Date', how='left')

            # Define the path to save the updated file
            save_path = os.path.join(directory_to_save_to, filename)

            # Save the updated data back to the new CSV file
            merged_data.to_csv(save_path, index=False)
            print(f"{filename} updated with fear and greed index data.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Script execution completed.")

