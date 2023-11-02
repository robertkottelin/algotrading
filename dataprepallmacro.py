import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from yahoo_fin import stock_info
from dotenv import load_dotenv

# Load .env environment variables
load_dotenv()

# Define the starting point for all data
start_date = '1980-01-01'

# Get the FRED API key from environment variables
fred_api_key = os.getenv('FRED_API_KEY')

# Initialize the Fred client with your API key
fred = Fred(api_key=fred_api_key)

# Fetch economic data series from FRED
gdp = fred.get_series('GDP', observation_start=start_date)
unemployment = fred.get_series('UNRATE', observation_start=start_date)
cpi = fred.get_series('CPIAUCSL', observation_start=start_date)
interest_rate = fred.get_series('FEDFUNDS', observation_start=start_date)
m2_money_supply = fred.get_series('M2NS', observation_start=start_date)

# Function to prepare economic dataframes
def prepare_economic_data(series, name):
    df = series.reset_index()
    df.rename(columns={'index': 'Date', 0: name}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')

# Prepare the economic dataframes
gdp_df = prepare_economic_data(gdp, 'GDP')
unemployment_df = prepare_economic_data(unemployment, 'Unemployment')
cpi_df = prepare_economic_data(cpi, 'CPI')
interest_rate_df = prepare_economic_data(interest_rate, 'Interest_Rate')
m2_money_supply_df = prepare_economic_data(m2_money_supply, 'M2_Money_Supply')


# Make sure the function can handle a DataFrame directly (not much change needed)
def prepare_market_data(df, suffix=''):
    columns_to_keep = {
        'High': f"High{suffix}",
        'Low': f"Low{suffix}",
        'Close': f"Close{suffix}",
        'Volume': f"Volume{suffix}"
    }
    df = df[['High', 'Low', 'Close', 'Volume']].rename(columns=columns_to_keep)
    return df

# New function to update existing CSV files with macroeconomic data
def update_csv_with_macrodata(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                print(f"Processing {file}")
                try:
                    file_path = os.path.join(root, file)
                    stock_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                    stock_df = prepare_market_data(stock_df)

                    # Merge stock data with macroeconomic data
                    full_df = pd.concat([
                        stock_df, 
                        gdp_df, 
                        unemployment_df, 
                        cpi_df, 
                        interest_rate_df, 
                        m2_money_supply_df
                    ], axis=1)

                    # Forward fill missing values, if any
                    full_df.fillna(method='ffill', inplace=True)

                    # Remove any rows that still have missing values after forward filling
                    full_df.dropna(inplace=True)

                    # Save to CSV
                    full_df.to_csv(file_path)
                    print(f"Data for {file} updated and saved.")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

# Directories containing stock data CSVs
nyse_directory = 'data/nysestocks'
nasdaq_directory = 'data/nasdaqstocks'

# Update CSV files with macroeconomic data
update_csv_with_macrodata(nyse_directory)
update_csv_with_macrodata(nasdaq_directory)

print("Script execution completed.")