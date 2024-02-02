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
    "coinglassSecret": coinglass_api_key
}

urlOHLC = "https://open-api.coinglass.com/public/v2/indicator/price_ohlc?ex=Binance&pair=ADAUSDT&interval=h4&limit=4500"
urlOI = "https://open-api.coinglass.com/public/v2/indicator/open_interest_ohlc?ex=Binance&pair=ADAUSDT&interval=h4&limit=4500"
urlFunding = "https://open-api.coinglass.com/public/v2/indicator/funding_avg?symbol=ADA&interval=h4&limit=4500"
urlLiquidation = "https://open-api.coinglass.com/public/v2/indicator/liquidation_pair?ex=Binance&pair=ADAUSDT&interval=h4&limit=4500"
urlLongShort = "https://open-api.coinglass.com/public/v2/indicator/top_long_short_position_ratio?ex=Binance&pair=ADAUSDT&interval=h4&limit=4500"
urlPuell = "https://open-api.coinglass.com/public/v2/index/puell_multiple"
urlPi = "https://open-api.coinglass.com/public/v2/index/pi"
urlGoldenRatio = "https://open-api.coinglass.com/public/v2/index/golden_ratio_multiplier"
urlStockFlow = "https://open-api.coinglass.com/public/v2/index/stock_flow"

ohlc = requests.get(urlOHLC, headers=headers)
ohlc_data = ohlc.json()['data']
# Convert to DataFrame
ohlc_data = pd.DataFrame(ohlc_data, columns=['t', 'o', 'h', 'l', 'c', 'v'])
# Convert createTime from Unix time (seconds) to datetime
ohlc_data['t'] = pd.to_datetime(ohlc_data['t'], unit='s')
# drop o, h, l
# ohlc_data = ohlc_data.drop(columns=['o', 'h', 'l'])
# Display the first few rows
print("PRICE:", ohlc_data.head())

# Make requests, saved as json
oi = requests.get(urlOI, headers=headers)
oi_data = oi.json()['data']
# print(oi_data)
# Convert to DataFrame
oi_data = pd.DataFrame(oi_data, columns=['t', 'o', 'c', 'h', 'l'])
# Convert createTime from Unix time (seconds) to datetime
oi_data['t'] = pd.to_datetime(oi_data['t'], unit='ms')
# drop column o, h, l
oi_data = oi_data.drop(columns=['o', 'h', 'l'])
oi_data = oi_data.rename(columns={'c': 'c_oi'})
# Display the first few rows
print("OI:", oi_data.head())

funding = requests.get(urlFunding, headers=headers)
funding_data = funding.json()['data']
# print(funding_data)
# Convert to DataFrame
funding_data = pd.DataFrame(funding_data, columns=['fundingRate', 'createTime'])
# Convert createTime from Unix time (seconds) to datetime
funding_data['createTime'] = pd.to_datetime(funding_data['createTime'], unit='ms')
# rename createTime to t
funding_data = funding_data.rename(columns={'createTime': 't'})
# Display the first few rows
print("FUNDING:", funding_data.head())

liquidation = requests.get(urlLiquidation, headers=headers)
liquidation_data = liquidation.json()['data']
# print(funding_data)
# Convert to DataFrame
liquidation_data = pd.DataFrame(liquidation_data, columns=['turnoverNumber', 'buyTurnoverNumber', 'sellTurnoverNumber', 'sellQty', 'buyQty', 'buyVolUsd', 'sellVolUsd', 'volUsd', 'createTime', 't'])
# Convert createTime from Unix time (seconds) to datetime
liquidation_data['createTime'] = pd.to_datetime(liquidation_data['createTime'], unit='ms')
# drop column t, sellQty, buyQty
liquidation_data = liquidation_data.drop(columns=['t', 'sellQty', 'buyQty'])
# rename createTime to t
liquidation_data = liquidation_data.rename(columns={'createTime': 't'})
# Fill NaN values with previous values
liquidation_data = liquidation_data.fillna(method='ffill')
# Display the first few rows
print("Liquidation:", liquidation_data.head())

long_short = requests.get(urlLongShort, headers=headers)
long_short_data = long_short.json()['data']
# print(funding_data)
# Convert to DataFrame
long_short_data = pd.DataFrame(long_short_data, columns=['longRatio', 'shortRatio', 'longShortRatio', 'createTime'])
# Convert createTime from Unix time (seconds) to datetime
long_short_data['createTime'] = pd.to_datetime(long_short_data['createTime'], unit='ms')
# rename createTime to t
long_short_data = long_short_data.rename(columns={'createTime': 't'})
# Display the first few rows
print("LONG/SHORT:", long_short_data.head())

puell = requests.get(urlPuell, headers=headers)
puell_data = puell.json()['data']
# print(funding_data)
# Convert to DataFrame
puell_data = pd.DataFrame(puell_data, columns=['buyQty', 'createTime', 'price', 'puellMultiple', 'sellQty'])
# Convert createTime from Unix time (seconds) to datetime
puell_data['createTime'] = pd.to_datetime(puell_data['createTime'], unit='ms')
# drop buyQty, sellQty, price
puell_data = puell_data.drop(columns=['buyQty', 'sellQty', 'price'])
# rename createTime to t
puell_data = puell_data.rename(columns={'createTime': 't'})
puell_data['t'] = pd.to_datetime(puell_data['t'])
# Display the first few rows
print("PUELL:", puell_data.head())

pi = requests.get(urlPi, headers=headers)
pi_data = pi.json()['data']
# Convert to DataFrame
pi_data = pd.DataFrame(pi_data, columns=['ma110', 'createTime', 'ma350Mu2', 'price'])
# Convert createTime from Unix time (seconds) to datetime
pi_data['createTime'] = pd.to_datetime(pi_data['createTime'], unit='ms')
pi_data = pi_data.drop(columns=['price'])
# rename createTime to t
pi_data = pi_data.rename(columns={'createTime': 't'})
pi_data['t'] = pd.to_datetime(pi_data['t'])
# Display the first few rows
print("PI:", pi_data.head())

golden_ratio = requests.get(urlGoldenRatio, headers=headers)
golden_ratio_data = golden_ratio.json()['data']
# print(funding_data)
# Convert to DataFrame
golden_ratio_data = pd.DataFrame(golden_ratio_data, columns=['3LowBullHigh', 'x8', '2LowBullHigh', 'createTime', 'price', 'ma350', '1.6AccumulationHigh', 'x21', 'x13', 'x5'])
# Convert createTime from Unix time (seconds) to datetime
golden_ratio_data['createTime'] = pd.to_datetime(golden_ratio_data['createTime'], unit='ms')
# drop price
golden_ratio_data = golden_ratio_data.drop(columns=['price', 'x8', 'x21', 'x13', 'x5'])
# rename createTime to t
golden_ratio_data = golden_ratio_data.rename(columns={'createTime': 't'})
golden_ratio_data['t'] = pd.to_datetime(golden_ratio_data['t'])
# Display the first few rows
print("GOLDEN RATIO:", golden_ratio_data.head())

stock_flow = requests.get(urlStockFlow, headers=headers)
stock_flow_data = stock_flow.json()['data']
# print(funding_data)
# Convert to DataFrame
stock_flow_data = pd.DataFrame(stock_flow_data, columns=['modelVariance', 'createTime', 'price', 'nextHalving', 'stockFlow365dAverage'])
# drop nan rows
stock_flow_data = stock_flow_data.dropna()
stock_flow_data = stock_flow_data.reset_index(drop=True)
# Convert createTime from Unix time (seconds) to datetime
stock_flow_data['createTime'] = pd.to_datetime(stock_flow_data['createTime']).dt.date
# drop price
stock_flow_data = stock_flow_data.drop(columns=['price'])
# rename createTime to t
stock_flow_data = stock_flow_data.rename(columns={'createTime': 't'})
stock_flow_data['t'] = pd.to_datetime(stock_flow_data['t'])
# Display the first few rows
print("STOCK FLOW:", stock_flow_data.head())

# merge all dataframes
merged_df = pd.merge_asof(ohlc_data, oi_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, funding_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, liquidation_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, puell_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, pi_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, golden_ratio_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, stock_flow_data, on='t', direction='nearest')

# Fill NaN values with previous values
merged_df = merged_df.fillna(method='ffill')

import pandas as pd
import os

# Check if the file exists
file_path = 'Crypto/data/coinglass_ADA.csv'
if os.path.exists(file_path):
    # Load the existing data
    existing_df = pd.read_csv(file_path)

    # Assuming 'merged_df' is already defined and contains the new data
    # And assuming 't' is the timestamp or unique identifier column
    new_rows = merged_df[~merged_df['t'].isin(existing_df['t'])]

    # Append new rows to the CSV file, if there are any
    if not new_rows.empty:
        new_rows.to_csv(file_path, mode='a', header=False, index=False)
        print("New rows appended.")
    else:
        print("No new rows to append.")
else:
    # If the file does not exist, you might want to save the new data as the file
    merged_df.to_csv(file_path, index=False)
    print(f"File {file_path} created with new data.")

# Optionally, you can print the head of the merged_df to confirm its structure or for debugging purposes
print(merged_df.head())
