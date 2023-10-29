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
m2_money_supply = fred.get_series('M2NS', observation_start=start_date)  # 'M2NS' is the FRED code for seasonally adjusted M2

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
m2_money_supply = prepare_economic_data(m2_money_supply, 'M2_Money_Supply') 

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

def add_lagging_close_prices(df, n_days):
    """
    This function adds columns to the DataFrame for n_days of lagging close prices.

    :param df: DataFrame containing market data, including a 'Close' column for S&P 500 prices.
    :param n_days: The number of days of lagging prices to include.
    :return: DataFrame with new columns for lagging close prices.
    """
    # Ensure that the data is sorted by date
    df = df.sort_index()

    # Generate columns for lagging close prices
    for i in range(1, n_days + 1):
        # The new column name
        col_name = f'Close_Lag_{i}'
        # Shift the data and save it in the new column
        df[col_name] = df['Close'].shift(i)

    return df


# Prepare market data
sp500_data = prepare_market_data(sp500_data)
vix_data = prepare_market_data(vix_data, '_VIX') 
# drop volume_vix
vix_data.drop(columns=['Volume_VIX', 'High_VIX', 'Low_VIX'], inplace=True)
vix_data.rename(columns={'Close_VIX': 'VIX'}, inplace=True)

# Combining all dataframes into a single dataframe
merged_df = pd.concat([sp500_data, vix_data, gdp, unemployment, cpi, interest_rate, m2_money_supply], axis=1)
# Fill missing values by propagating the last valid observation forward to next valid
merged_df.fillna(method='ffill', inplace=True)

# Add 5 previous days' closing prices
merged_df = add_lagging_close_prices(merged_df, 5)

merged_df.fillna(method='ffill', inplace=True)

# Drop NaN rows
merged_df.dropna(inplace=True)


# Specify the new order of the columns explicitly
new_order_columns = [
    'High', 'Low', 'Close', 
    'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5', 
    'Volume', 'VIX', 
    'GDP', 'Unemployment', 'CPI', 'Interest_Rate', 'M2_Money_Supply'
    # Add any other columns here as per your DataFrame
]

# Ensure that all columns are present (especially important if some columns might have been optional or conditional)
assert set(new_order_columns) == set(merged_df.columns), "Column sets do not match"

# Reindex the DataFrame with the new column order
merged_df = merged_df[new_order_columns]

# Display the DataFrame with the new column order
print(merged_df.head())

# Save the reordered DataFrame to a CSV file
merged_df.to_csv('data/macrodata.csv')
