import os
import pandas as pd
from pafunctions import (
    calculate_rsi, calculate_macd, calculate_vwap, calculate_momentum, 
    calculate_bollinger_bands, calculate_fibonacci_retracements, calculate_pivot_points
)

# Directory paths
input_directory = 'data/macrodata/'
output_directory = 'data/macrotechnical/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):  # Check whether it's a CSV file
        try:
            # Construct the full (absolute) file path
            file_path = os.path.join(input_directory, filename)
            
            # Load and process the data
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])  # Ensure the Date is in datetime format

            # Calculations
            window = 14  # for RSI
            df['RSI'] = calculate_rsi(df, window)

            slow, fast, signal = 26, 12, 9  # for MACD
            df['MACD'], df['MACD_Signal'] = calculate_macd(df, slow, fast, signal)
            df.drop(columns=['MACD'], inplace=True)  # if you want to drop the MACD

            df['VWAP'] = calculate_vwap(df)

            n = 10  # for Momentum
            df['Momentum'] = calculate_momentum(df, n)

            bb_window = 20  # for Bollinger Bands
            num_of_std = 2
            df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df, bb_window, num_of_std)

            pivot_points = calculate_pivot_points(df)
            for key, value in pivot_points.items():
                df[key] = value

            # Assuming you want to calculate Fibonacci levels based on the overall high and low
            # overall_high = df['High'].max()
            # overall_low = df['Low'].min()
            # fibonacci_retracements = calculate_fibonacci_retracements(overall_high, overall_low)
            # for key, value in fibonacci_retracements.items():
            #     df[key] = value

            # Drop rows with NaN values
            df.dropna(inplace=True)

            # Save to a new CSV in the output directory
            output_filename = os.path.splitext(filename)[0] + '_technical.csv'
            output_file_path = os.path.join(output_directory, output_filename)
            df.to_csv(output_file_path)

            print(f"Processed and saved technical data for {filename}")

        except Exception as e:
            print(f"An error occurred while processing {filename}: {e}")

print("All files have been processed.")
