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
directories_to_update = ['data/nysestocks', 'data/nasdaqstocks']

for directory in directories_to_update:
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Only process CSV files
            print(f"Processing {filename}...")
            file_path = os.path.join(directory, filename)

            try:
                # Load the stock data
                stock_data = pd.read_csv(file_path)
                stock_data['Date'] = pd.to_datetime(stock_data['Date'])

                # Merge the fear and greed index data with the stock data
                merged_data = pd.merge(stock_data, fear_greed_data, on='Date', how='left')

                # Save the updated data back to the original CSV file
                merged_data.to_csv(file_path, index=False)
                print(f"{filename} updated with fear and greed index data.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

print("Script execution completed.")
