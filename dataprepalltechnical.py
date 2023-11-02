import os
import pandas as pd
from pafunctions import (
    calculate_rsi, calculate_macd, calculate_vwap, calculate_momentum, 
    calculate_bollinger_bands, calculate_pivot_points
)

# Define directories
nyse_directory = 'data/nysestocks'
nasdaq_directory = 'data/nasdaqstocks'

def add_technical_indicators(df):
    # Make sure 'Date' and other relevant columns are in the correct data type
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculations
    window = 14  # for RSI
    df['RSI'] = calculate_rsi(df, window)

    slow, fast, signal = 26, 12, 9  # for MACD
    df['MACD'], df['MACD_Signal'] = calculate_macd(df, slow, fast, signal)

    df['VWAP'] = calculate_vwap(df)

    n = 10  # for Momentum
    df['Momentum'] = calculate_momentum(df, n)

    bb_window = 20  # for Bollinger Bands
    num_of_std = 2
    df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df, bb_window, num_of_std)

    pivot_points = calculate_pivot_points(df)
    for key, value in pivot_points.items():
        df[key] = value

    # Drop the MACD column, only interested in the MACD_Signal column
    # df.drop(columns=['MACD'], inplace=True)

    # Drop rows with NaN values that may have been introduced by technical indicator calculations
    df.dropna(inplace=True)
    return df

def process_files_in_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Check whether it's a CSV file
            file_path = os.path.join(directory, filename)
            print(f"Attempting to process {filename}...")
            
            try:
                df = pd.read_csv(file_path)

                if 'Date' not in df.columns:
                    print(f"File {filename} does not contain 'Date' column. Skipping...")
                    continue

                print(f"'Date' column entries in {filename}:")
                print(df['Date'].head())

                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)

                # Debug line to confirm the index
                print(f"Index after setting 'Date':\n{df.index}")

                df = add_technical_indicators(df)

                df.to_csv(file_path)
                print(f"Processed and updated {filename} with technical indicators.")

            except Exception as e:
                print(f"An error occurred while processing {filename}: {e}")



# Process CSV files in both directories
process_files_in_directory(nyse_directory)
process_files_in_directory(nasdaq_directory)

print("All files have been processed with technical indicators.")
