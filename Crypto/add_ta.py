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

# Function to process each file
def process_file(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)

    # Assuming 'c' is the 'Close' price for RSI and MACD calculation
    df['RSI'] = compute_rsi(df['c'], 25)  # Optimized, Lower threshold 17, higher 76
    df['MACD'], df['MACD_Signal'] = compute_macd(df['c'], 15, 23, 6)  # Optimized
    
    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False)

# List of symbols to process
symbols = ['AAVE' , 'ADA', 'AVAX', 'BNB', 'BTC', 'DOT', 'ETH', 'ETC',  'LINK', 'LTC', 'MATIC', 'SOL', 'XRP']  # Add or remove symbols as needed

# Loop through the symbols and process each file
for symbol in symbols:
    input_file_path = f'Crypto/data/coinglass_{symbol}.csv'
    output_file_path = f'Crypto/data/coinglass_{symbol}_ta.csv'
    process_file(input_file_path, output_file_path)
