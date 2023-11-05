import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Directory containing your CSV files
input_dir = 'data/macrotechnicalfearandgreedprepped/'
output_dir = 'data/preppedconcatdata/'

# Check if output directory exists, if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

import numpy as np

def load_and_combine_data(data_directory, scaler):
    combined_data = pd.DataFrame()

    for filename in os.listdir(data_directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(data_directory, filename)
            df = pd.read_csv(file_path)

            # Optionally, handle very large values that might be considered as infinities.
            # You can clip / limit the max and min values of the dataset here if necessary
            # For example:
            max_val = df.quantile(0.99999, numeric_only=True)
            min_val = df.quantile(0.00001, numeric_only=True)
            df = df.clip(lower=min_val, upper=max_val, axis=1)

            # print(f"Shape of the DataFrame from {filename} before dropping columns: {df.shape}")
            df.drop('Date', axis=1, inplace=True, errors='ignore')
            # print(f"Shape of the DataFrame from {filename} after dropping columns: {df.shape}")

            # Drop rows with NaN or empty values
            df.dropna(inplace=True)

            # Extract X without the target column 'Next_Higher'
            X = df.drop('Next_Higher', axis=1)
            
            if X.empty:
                print(f"No data to scale in file {filename}.")
                continue

            # Normalize the data, do a try, catch statement to handle any potential errors
            try:
                X_scaled = scaler.fit_transform(X)
            except ValueError as e:
                print(f"Error scaling data from file {filename}.")
                print(e)
                continue
        
            # Merge the scaled data back with the target column
            df_scaled = pd.concat([pd.DataFrame(X_scaled, columns=X.columns), df['Next_Higher'].reset_index(drop=True)], axis=1)
            
            combined_data = pd.concat([combined_data, df_scaled], ignore_index=True)

    return combined_data


def main():
    scaler = StandardScaler()

    print("Loading and combining data...")
    # Load and combine data
    combined_data = load_and_combine_data(input_dir, scaler)

    # Save the scaler
    scaler_filepath = "models/scaler.save"
    dump(scaler, scaler_filepath)

    # Save the combined and normalized data to a new CSV file
    combined_data.to_csv(os.path.join(output_dir, 'combined_data.csv'), index=False)

if __name__ == "__main__":
    main()
