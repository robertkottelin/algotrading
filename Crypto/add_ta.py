import pandas as pd
import numpy as np

# Function to calculate RSI
def compute_rsi(data, window):
    delta = data.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    roll_up = up.rolling(window).mean()
    roll_down = down.abs().rolling(window).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

# Function to calculate MACD
def compute_macd(data, span1, span2, signal_span):
    ema1 = data.ewm(span=span1, adjust=False).mean()
    ema2 = data.ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal

# Load the CSV file
def process_file(file_path, output_file_path):
    df = pd.read_csv(file_path)

    # Assuming 'c' is the 'Close' price for RSI and MACD calculation
    df['RSI'] = compute_rsi(df['c'], 14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['c'], 12, 26, 9)
    
    # Drop rows with NaN values
    df = df.dropna()

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

# File paths
input_file_path = 'Crypto/coinglass.csv'  # Replace with your input file path
output_file_path = 'Crypto/coinglass_ta.csv' # Replace with your desired output file path

# Process the file
process_file(input_file_path, output_file_path)
