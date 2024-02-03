import pandas as pd
import numpy as np
import os

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

# Function to process each file
def process_file(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)

    # Assuming 'BTC price' is the column name for the 'Close' price for RSI and MACD calculation
    df['RSI'] = compute_rsi(df['BTC price'], 14)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['BTC price'], 12, 26, 9)
    
    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

def process_all_files():
    input_dir = 'Crypto/data'
    output_dir = 'Crypto/ta_data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)
        process_file(input_file_path, output_file_path)
        print(f'Processed and saved: {output_file_path}')

process_all_files()
