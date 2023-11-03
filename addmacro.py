import pandas as pd
import os

# Function to clean the date by removing the time and timezone
def clean_date(date_str):
    return pd.to_datetime(date_str).date()

# Load the macroeconomic data
macro_df = pd.read_csv('data/macrodata.csv')
# Convert the Date to datetime and format it to date only
macro_df['Date'] = pd.to_datetime(macro_df['Date']).dt.date

# Find the start and end dates
start_date = macro_df['Date'].min()
end_date = macro_df['Date'].max()

# Create a date range that includes every day between the start and end dates
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Reindex the dataframe with the date range, forward-filling missing values
macro_df = macro_df.set_index('Date').reindex(date_range).fillna(method='ffill').reset_index()
macro_df.rename(columns={'index': 'Date'}, inplace=True)
macro_df['Date'] = macro_df['Date'].dt.date

# Get the list of stock data files
stock_files = os.listdir('data/stocks')

# Make sure the macrodata directory exists
os.makedirs('data/macrodata', exist_ok=True)

for file in stock_files:
    stock_path = os.path.join('data/stocks', file)
    
    # Load stock data
    stock_df = pd.read_csv(stock_path)

    # Remove the time and timezone part from the 'Date' column
    stock_df['Date'] = stock_df['Date'].str.replace(r' \d{2}:\d{2}:\d{2}[-+]\d{2}:\d{2}', '', regex=True)
    # Then convert the Date to datetime and format it to date only
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.date
    
    # Merge the data on the Date column
    merged_df = pd.merge(stock_df, macro_df, on='Date', how='left')
    
    # Drop rows where the Date does not have macroeconomic data
    merged_df.dropna(subset=['GDP', 'Unemployment', 'CPI', 'Interest_Rate', 'M2_Money_Supply'], inplace=True)
    
    # Save the merged DataFrame
    merged_path = os.path.join('data/macrodata', file)
    merged_df.to_csv(merged_path, index=False)
    
print("All files have been processed and saved in the data/macrodata directory.")
