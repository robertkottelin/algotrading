import pandas as pd
from pafunctions import calculate_rsi, calculate_macd, calculate_vwap, calculate_momentum, calculate_bollinger_bands, calculate_fibonacci_retracements, calculate_pivot_points

# Load your data
df = pd.read_csv('macrodata.csv')
df['Date'] = pd.to_datetime(df['Date'])  # Make sure the Date is in datetime format

# Calculate RSI and MACD
window = 14  # for RSI
slow, fast, signal = 26, 12, 9  # for MACD
df['RSI'] = calculate_rsi(df, window)
df['MACD'], df['MACD_Signal'] = calculate_macd(df, slow, fast, signal)
# drop the macd
df.drop(columns=['MACD'], inplace=True)

# Calculate VWAP, Momentum
df['VWAP'] = calculate_vwap(df)
n = 10  # number of days for Momentum calculation, can be adjusted
df['Momentum'] = calculate_momentum(df, n)

# Bollinger Bands parameters
bb_window = 20  # Choose moving average periods
num_of_std = 2  # Number of standard deviations for the bands
df['Bollinger_Upper'], df['Bollinger_Lower'] = calculate_bollinger_bands(df, bb_window, num_of_std)

# Calculate Pivot Points
pivot_points = calculate_pivot_points(df)
for key, value in pivot_points.items():
    df[key] = value

# Assuming you want to calculate Fibonacci levels based on the overall high and low
overall_high = df['High'].max()
overall_low = df['Low'].min()

fibonacci_retracements = calculate_fibonacci_retracements(overall_high, overall_low)
for key, value in fibonacci_retracements.items():
    df[key] = value  # This will create a new column for each Fibonacci level

# drop rows with NaN
df.dropna(inplace=True)
# Print the head of the dataframe to verify calculations
print(df.head())
