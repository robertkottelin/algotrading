import os
import requests
import json
from dotenv import load_dotenv
import pandas as pd


# Load API key and secret from .env file
load_dotenv()
coinglass_api_key = os.getenv("COINGLASS_API_KEY")
# Define headers
headers = {
    "accept": "application/json",
    "coinglassSecret": coinglass_api_key,
    "CG-API-KEY": coinglass_api_key

}

# Info
urlSupportedCoins = "https://open-api-v3.coinglass.com/api/futures/supported-exchange-pairs"


response = requests.get(urlSupportedCoins, headers=headers)

# Parse the JSON string
data = response.json()

# Extract the 'data' part which contains the exchanges
exchanges_data = data['data']

# Initialize an empty DataFrame
df = pd.DataFrame()

# Loop through each exchange and append to the DataFrame
for exchange, instruments in exchanges_data.items():
    # Convert current exchange's instruments to a DataFrame
    temp_df = pd.json_normalize(instruments)
    # Add a column for the exchange name
    temp_df['exchange'] = exchange
    # Append to the main DataFrame
    df = pd.concat([df, temp_df], ignore_index=True)

# Save the DataFrame to a CSV file
df.to_csv('Crypto/data/exchanges.csv', index=False)