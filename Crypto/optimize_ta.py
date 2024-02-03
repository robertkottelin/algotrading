import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count

# Function to calculate RSI
def compute_rsi(data, window=14):
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
def compute_macd(data, span1=12, span2=26, signal_span=9):
    ema1 = data.ewm(span=span1, adjust=False).mean()
    ema2 = data.ewm(span=span2, adjust=False).mean()
    macd = ema1 - ema2
    signal = macd.ewm(span=signal_span, adjust=False).mean()
    return macd, signal

# Function to define trade signals
def define_trade_signals(df, rsi_lower_threshold, rsi_upper_threshold):
    df['Buy_Signal'] = ((df['RSI'] > rsi_lower_threshold) & (df['RSI'].shift(1) <= rsi_lower_threshold)) | ((df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)))
    df['Sell_Signal'] = ((df['RSI'] < rsi_upper_threshold) & (df['RSI'].shift(1) >= rsi_upper_threshold)) | ((df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)))
    return df

# Function to simulate trading and calculate profitability
def simulate_trading(df):
    in_position = False
    buy_price = 0
    profit = 0

    for index, row in df.iterrows():
        if row['Buy_Signal'] and not in_position:
            buy_price = row['BTC price']
            in_position = True
        elif row['Sell_Signal'] and in_position:
            sell_price = row['BTC price']
            profit += (sell_price - buy_price) / buy_price
            in_position = False

    return profit


# Function to process a single set of parameters (slightly modified for multiprocessing)
def process_parameters(params):
    df = pd.read_csv('Crypto/data/coinglass_BTC.csv')  # Load the data in each process
    rsi_window, lower_threshold, upper_threshold, macd_span1, macd_span2, macd_signal_span = params

    df['RSI'] = compute_rsi(df['BTC price'], rsi_window)
    df['MACD'], df['MACD_Signal'] = compute_macd(df['BTC price'], macd_span1, macd_span2, macd_signal_span)
    df = define_trade_signals(df, lower_threshold, upper_threshold)
    profit = simulate_trading(df)

    return profit, params

# Function to optimize parameters using multiprocessing
def optimize_parameters(file_path):
    parameter_space = list(itertools.product(
            range(8, 32, 1), range(14, 30, 1), range(70, 86, 1), # RSI parameters: window, lower threshold, upper threshold
            range(8, 24, 1), range(20, 38, 1), range(6, 26, 1))) # MACD parameters: span1, span2, signal_span
    
    total_iterations = len(parameter_space)
    processed_iterations = 0

    best_profit = -np.inf
    best_parameters = None

    with Pool(cpu_count()) as pool:
        for profit, params in pool.imap_unordered(process_parameters, parameter_space):
            processed_iterations += 1
            if profit > best_profit:
                best_profit = profit
                best_parameters = params
                print(f"New Best Profit: {best_profit} with Parameters: {best_parameters}")
                # Write parameters to file
                with open('Crypto/data/parameters.txt', 'w') as f:
                    f.write(f"Final Best Parameters: {best_parameters}\n")
                    f.write(f"Final Best Profit: {best_profit}\n")
            if processed_iterations % 1000 == 0:
                print(f"Progress: {processed_iterations}/{total_iterations} scenarios completed.")

    return best_parameters, best_profit

# File path
input_file_path = 'Crypto/data/coinglass_BTC.csv' 

# Optimize parameters
best_parameters, best_profit = optimize_parameters(input_file_path)

# Print final best parameters and profit
print(f"Final Best Parameters: {best_parameters}")
print(f"Final Best Profit: {best_profit}")

# Write parameters to file
with open('Crypto/data/parameters.txt', 'w') as f:
    f.write(f"Final Best Parameters: {best_parameters}\n")
    f.write(f"Final Best Profit: {best_profit}\n")