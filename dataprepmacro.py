import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()

# Set the starting point for all data
start_date = '1990-01-01'

# Get the API key from environment variables
fred_api_key = os.getenv('FRED_API_KEY')

# Initialize the Fred instance with your API key
fred = Fred(api_key=fred_api_key)

# Fetch economic data with the specified start date
gdp = fred.get_series('GDP', observation_start=start_date)
unemployment = fred.get_series('UNRATE', observation_start=start_date)
cpi = fred.get_series('CPIAUCSL', observation_start=start_date)
# Fetching the Federal Funds Rate
interest_rate = fred.get_series('FEDFUNDS', observation_start=start_date)

# Standardizing data retrieval process and structure for economic indicators
def prepare_economic_data(series, name):
    df = series.reset_index()
    df.rename(columns={'index': 'Date', 0: name}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure the 'Date' column is in datetime format
    return df.set_index('Date')  # Set 'Date' as the index for convenient time-series operations

# Prepare economic data
gdp = prepare_economic_data(gdp, 'GDP')
unemployment = prepare_economic_data(unemployment, 'Unemployment')
cpi = prepare_economic_data(cpi, 'CPI')
interest_rate = prepare_economic_data(interest_rate, 'Interest_Rate')  # Preparing interest rate data

# Fetch S&P 500 and VIX data
tickerSymbol = '^GSPC'
vixSymbol = '^VIX'

# Using yfinance to fetch data for S&P 500 and VIX
sp500_data = yf.download(tickerSymbol, start=start_date)
vix_data = yf.download(vixSymbol, start=start_date)

# Standardizing the structure for financial market data
def prepare_market_data(df, suffix=''):
    # Define the columns you want to keep. Include 'High' and 'Low'.
    columns_to_keep = {
        'High': f"High{suffix}",
        'Low': f"Low{suffix}",
        'Close': f"Close{suffix}",
        'Volume': f"Volume{suffix}"  # Keep the 'Volume' and rename to avoid confusion in the merged data.
    }

    # Select the 'High', 'Low', 'Close', and 'Volume' columns and rename them.
    df = df[['High', 'Low', 'Close', 'Volume']].rename(columns=columns_to_keep)
    return df

# Prepare market data
sp500_data = prepare_market_data(sp500_data)
vix_data = prepare_market_data(vix_data, '_VIX') 
# drop volume_vix
vix_data.drop(columns=['Volume_VIX', 'High_VIX', 'Low_VIX'], inplace=True)
vix_data.rename(columns={'Close_VIX': 'VIX'}, inplace=True)

# Combining all dataframes into a single dataframe
merged_df = pd.concat([sp500_data, vix_data, gdp, unemployment, cpi, interest_rate], axis=1)

# Fill missing values by propagating the last valid observation forward to next valid
merged_df.fillna(method='ffill', inplace=True)

# Drop NaN rows
merged_df.dropna(inplace=True)

# Display the merged DataFrame
print(merged_df.head())

# Save the merged DataFrame to a CSV file
merged_df.to_csv('data/macrodata.csv')
