import os
import pandas as pd
from pafunctions import (
    calculate_rsi, calculate_macd, calculate_vwap, calculate_momentum, 
    calculate_bollinger_bands, calculate_pivot_points
)

# Define directories
input_directory = 'data/SNP'
output_directory = 'data/SNP'

# Ensure output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def add_technical_indicators(df):
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

    # Drop rows with NaN values that may have been introduced by technical indicator calculations
    df.dropna(inplace=True)

    return df

# def process_files_in_directory(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith('.csv'):  # Check whether it's a CSV file
#             file_path = os.path.join(directory, filename)
#             output_path = os.path.join(output_directory, filename)
#             print(f"Attempting to process {filename}...")
            
#             try:
#                 df = pd.read_csv(file_path)
#                 df['Date'] = pd.to_datetime(df['Date'])
#                 df.set_index('Date', inplace=True)

#                 df = add_technical_indicators(df)

#                 df.reset_index(inplace=True)
#                 df.to_csv('data/SNP/SNPMacroTechnical.csv', index=False)
#                 print(f"Processed and updated {filename} with technical indicators.")

#             except Exception as e:
#                 print(f"An error occurred while processing {filename}: {e}")

# # Process CSV files in the input directory
# process_files_in_directory(input_directory)

df = pd.read_csv('data/SNP/SNPMacro.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df = add_technical_indicators(df)

df.reset_index(inplace=True)
df.to_csv('data/SNP/SNPMacroTechnical.csv', index=False)
print(f"Processed and updated with technical indicators.")

print("All files have been processed with technical indicators.")
