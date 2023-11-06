import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import datetime

# Load .env environment variables
load_dotenv()

# Define the starting point for all data
start_date = '1980-01-01'

# Get the FRED API key from environment variables
fred_api_key = os.getenv('FRED_API_KEY')

# Check if FRED API key is not found
if not fred_api_key:
    raise ValueError("FRED API Key not found in the environment variables.")

# Initialize the Fred client with your API key
fred = Fred(api_key=fred_api_key)

# Function to fetch and prepare economic dataframes
def prepare_economic_data(fred, series_id, start_date, name):
    series = fred.get_series(series_id, observation_start=start_date)
    df = series.reset_index()
    df.rename(columns={'index': 'Date', 0: name}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df.set_index('Date')

# Prepare the economic dataframes
gdp_df = prepare_economic_data(fred, 'GDP', start_date, 'GDP')
unemployment_df = prepare_economic_data(fred, 'UNRATE', start_date, 'Unemployment')
cpi_df = prepare_economic_data(fred, 'CPIAUCSL', start_date, 'CPI')
interest_rate_df = prepare_economic_data(fred, 'FEDFUNDS', start_date, 'Interest_Rate')
m2_money_supply_df = prepare_economic_data(fred, 'M2NS', start_date, 'M2_Money_Supply')

# Merge all economic dataframes
all_data_df = pd.concat([gdp_df, unemployment_df, cpi_df, interest_rate_df, m2_money_supply_df], axis=1)
all_data_df.fillna(method='ffill', inplace=True)

# Ensure the DataFrame covers all dates up to today
all_dates = pd.date_range(start=all_data_df.index.min(), end=datetime.datetime.today(), freq='D')
all_data_df = all_data_df.reindex(all_dates).fillna(method='ffill')

# Now, export the merged DataFrame to a CSV file
all_data_df.to_csv('data/macrodata.csv')

print(all_data_df)


print("Script execution completed.")
