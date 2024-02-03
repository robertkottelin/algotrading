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
urlSupportedCoins = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urlSupportedExchanges = "https://open-api-v3.coinglass.com/api/futures/supported-exchange-pairs"

# Open Interest
urlOIOHLC = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urlOIHistory = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-aggregated-history?symbol=ETC&interval=1d&limit=4500&startTime=1652342400"
# urlExchangeListOIETC = "https://open-api-v3.coinglass.com/api/futures/openInterest/exchange-list?symbol=ETC" no time column

# Funding
urlFunding = "https://open-api-v3.coinglass.com/api/futures/fundingRate/ohlc-history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urlFundingHistoryOIWeight = "https://open-api-v3.coinglass.com/api/futures/fundingRate/oi-weight-ohlc-history?symbol=ETC&interval=1d&limit=4500"
urlFundingHistoryVolWeight = "https://open-api-v3.coinglass.com/api/futures/fundingRate/vol-weight-ohlc-history?symbol=ETC&interval=1d&limit=4500"

# Liquidation
urlLiquidation = "https://open-api-v3.coinglass.com/api/futures/liquidation/history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urlLiquidationHistory = "https://open-api-v3.coinglass.com/api/futures/liquidation/aggregated-history?symbol=ETC&interval=1d&limit=4500"
# urlLiquidationCoinList = "https://open-api-v3.coinglass.com/api/futures/liquidation/coin-list?symbol=ETC&range=1d"
# urlLiquidationExchangeList = "https://open-api-v3.coinglass.com/api/futures/liquidation/exchange-list?symbol=ETC&range=1d"

# Long/Short
urlGlobalLongShort = "https://open-api-v3.coinglass.com/api/futures/globalLongShortAccountRatio/history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urlAccountLongShort = "https://open-api-v3.coinglass.com/api/futures/topLongShortAccountRatio/history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urltopLongShortPositionRatio = "https://open-api-v3.coinglass.com/api/futures/topLongShortPositionRatio/history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urlaggregatedTakerBuySellVolumeRatio = "https://open-api-v3.coinglass.com/api/futures/aggregatedTakerBuySellVolumeRatio/history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
urltakerBuySellVolume = "https://open-api-v3.coinglass.com/api/futures/takerBuySellVolume/history?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"
# urltakerBuySellVolume = "https://open-api-v3.coinglass.com/api/futures/takerBuySellVolume/exchange-list?exchange=Binance&symbol=ETCUSDT&interval=1d&limit=4500"

# Indicator
urlBubbleIndex = "https://open-api-v3.coinglass.com/api/index/bitcoin-bubble-index"
urlahr999 = "https://open-api-v3.coinglass.com/api/index/ahr999"
urlPuell = "https://open-api-v3.coinglass.com/api/index/puell-multiple"
urlStockFlow = "https://open-api-v3.coinglass.com/api/index/stock-flow"
urlPi = "https://open-api-v3.coinglass.com/api/index/pi"
urlGoldenRatio = "https://open-api-v3.coinglass.com/api/index/golden-ratio-multiplier"
urlETCProfitable = "https://open-api-v3.coinglass.com/api/index/bitcoin-profitable-days"
urlRainbow = "https://open-api-v3.coinglass.com/api/index/bitcoin-rainbow-chart"
urlFearGreed = "https://open-api-v3.coinglass.com/api/index/fear-greed-history"


# Option
urlOption = "https://open-api-v3.coinglass.com/api/option/info?symbol=ETC"


oi_ohlc = requests.get(urlOIOHLC, headers=headers)
oi_ohlc_data = oi_ohlc.json()['data']
# Convert to DataFrame
oi_ohlc_data = pd.DataFrame(oi_ohlc_data, columns=['o', 'h', 'l', 'c', 't'])
# Convert createTime from Unix time (seconds) to datetime
oi_ohlc_data['t'] = pd.to_datetime(oi_ohlc_data['t'], unit='s')
# drop nan
oi_ohlc_data = oi_ohlc_data.dropna()
print("openInterest/ohlc-history:", oi_ohlc_data.head())

# oi_history = requests.get(urlOIHistory, headers=headers)
# oi_history_data = oi_history.json()['data']
# # print(oi_data)
# # Convert to DataFrame
# oi_history_data = pd.DataFrame(oi_history_data, columns=['o', 'h', 'l', 'c', 't'])
# # Convert createTime from Unix time (seconds) to datetime
# oi_history_data['t'] = pd.to_datetime(oi_history_data['t'], unit='s')
# # drop column o, h, l
# # oi_data = oi_data.drop(columns=['o', 'h', 'l'])
# # oi_data = oi_data.rename(columns={'c': 'c_oi'})
# # Display the first few rows
# oi_history_data = oi_history_data.dropna()
# print("openInterest/ohlc-aggregated-history:", oi_history_data.head())

funding = requests.get(urlFunding, headers=headers)
funding_data = funding.json()['data']
# print(funding_data)
# Convert to DataFrame
funding_data = pd.DataFrame(funding_data, columns=['o', 'h', 'l', 'c', 't'])
# Convert createTime from Unix time (seconds) to datetime
funding_data['t'] = pd.to_datetime(funding_data['t'], unit='s')
# rename createTime to t
# Display the first few rows
funding_data = funding_data.dropna()
print("fundingRate/ohlc-history:", funding_data.head())

funding_oi = requests.get(urlFundingHistoryOIWeight, headers=headers)
# Extract the JSON content
funding_oi_json = funding_oi.json()
# Assuming the relevant data is in the 'data' key
funding_oi_data = funding_oi_json['data']
# Convert to DataFrame
funding_oi_df = pd.DataFrame(funding_oi_data)
# Convert 't' from Unix time (seconds) to datetime
funding_oi_df['t'] = pd.to_datetime(funding_oi_df['t'], unit='s')
# Convert 'o', 'h', 'l', 'c' from strings to numeric values
funding_oi_df['o'] = pd.to_numeric(funding_oi_df['o'])
funding_oi_df['h'] = pd.to_numeric(funding_oi_df['h'])
funding_oi_df['l'] = pd.to_numeric(funding_oi_df['l'])
funding_oi_df['c'] = pd.to_numeric(funding_oi_df['c'])
funding_oi_df = funding_oi_df.dropna()
# Display the first few rows
print("fundingRate/oi-weight-ohlc-history:", funding_oi_df.head())

# funding_oi = requests.get(urlFundingHistoryVolWeight, headers=headers)
# # Extract the JSON content
# funding_oi_json = funding_oi.json()
# # Assuming the relevant data is in the 'data' key
# funding_oi_data = funding_oi_json['data']
# # Convert to DataFrame
# funding_vol_df = pd.DataFrame(funding_oi_data)
# # Convert 't' from Unix time (seconds) to datetime
# funding_vol_df['t'] = pd.to_datetime(funding_vol_df['t'], unit='s')
# # Convert 'o', 'h', 'l', 'c' from strings to numeric values
# funding_vol_df['o'] = pd.to_numeric(funding_vol_df['o'])
# funding_vol_df['h'] = pd.to_numeric(funding_vol_df['h'])
# funding_vol_df['l'] = pd.to_numeric(funding_vol_df['l'])
# funding_vol_df['c'] = pd.to_numeric(funding_vol_df['c'])
# # drop nan rows
# funding_vol_df = funding_vol_df.dropna()
# # Display the first few rows
# print("fundingRate/vol-weight-ohlc-history:", funding_vol_df.head())

liquidation_history = requests.get(urlLiquidation, headers=headers)
liquidation_history_data = liquidation_history.json()['data']
# print(funding_data)
# Convert to DataFrame
liquidation_history_data = pd.DataFrame(liquidation_history_data, columns=['longLiquidationUsd', 'shortLiquidationUsd', 't'])
# Convert createTime from Unix time (seconds) to datetime
liquidation_history_data['t'] = pd.to_datetime(liquidation_history_data['t'], unit='s')
# Display the first few rows
liquidation_history_data = liquidation_history_data.dropna()
print("liquidation/history:", liquidation_history_data)

liquidation_history_aggr = requests.get(urlLiquidationHistory, headers=headers)
liquidation_history_aggr_data = liquidation_history_aggr.json()['data']
# print(funding_data)
# Convert to DataFrame
liquidation_history_aggr_data = pd.DataFrame(liquidation_history_aggr_data, columns=['longLiquidationUsd', 'shortLiquidationUsd', 't'])
# Convert createTime from Unix time (seconds) to datetime
liquidation_history_aggr_data['t'] = pd.to_datetime(liquidation_history_aggr_data['t'], unit='s')
liquidation_history_aggr_data = liquidation_history_aggr_data.dropna()
# Display the first few rows
print("liquidation/aggregated-history:", liquidation_history_aggr_data)


global_long_short = requests.get(urlGlobalLongShort, headers=headers)
global_long_short_data = global_long_short.json()['data']
# print(funding_data)
# Convert to DataFrame
global_long_short_data = pd.DataFrame(global_long_short_data, columns=['time', 'longAccount', 'shortAccount', 'longShortRadio'])
# Convert createTime from Unix time (seconds) to datetime
global_long_short_data['time'] = pd.to_datetime(global_long_short_data['time'], unit='s')
# rename createTime to t
global_long_short_data = global_long_short_data.rename(columns={'time': 't'})
global_long_short_data = global_long_short_data.dropna()
# Display the first few rows
print("globalLongShortAccountRatio/history:", global_long_short_data.head())

top_long_short = requests.get(urlAccountLongShort, headers=headers)
top_long_short_data = top_long_short.json()['data']
# print(funding_data)
# Convert to DataFrame
top_long_short_data = pd.DataFrame(top_long_short_data, columns=['time', 'longAccount', 'shortAccount', 'longShortRadio'])
# Convert createTime from Unix time (seconds) to datetime
top_long_short_data['time'] = pd.to_datetime(top_long_short_data['time'], unit='s')
# rename createTime to t
top_long_short_data = top_long_short_data.rename(columns={'time': 't'})
top_long_short_data = top_long_short_data.dropna()
# Display the first few rows
print("topLongShortAccountRatio/history:", top_long_short_data.head())

top_long_short_position = requests.get(urltopLongShortPositionRatio, headers=headers)
top_long_short_position_data = top_long_short_position.json()['data']
# print(funding_data)
# Convert to DataFrame
top_long_short_position_data = pd.DataFrame(top_long_short_position_data, columns=['time', 'longAccount', 'shortAccount', 'longShortRadio'])
# Convert createTime from Unix time (seconds) to datetime
top_long_short_position_data['time'] = pd.to_datetime(top_long_short_position_data['time'], unit='s')
# rename createTime to t
top_long_short_position_data = top_long_short_position_data.rename(columns={'time': 't'})
top_long_short_position_data = top_long_short_position_data.dropna()
# Display the first few rows
print("topLongShortAccountRatio/history:", top_long_short_position_data.head())

aggr_taker_vol = requests.get(urlaggregatedTakerBuySellVolumeRatio, headers=headers)
aggr_taker_vol_data = aggr_taker_vol.json()['data']
# print(funding_data)
# Convert to DataFrame
aggr_taker_vol_data = pd.DataFrame(aggr_taker_vol_data, columns=['time', 'longShortRadio'])
# Convert createTime from Unix time (seconds) to datetime
aggr_taker_vol_data['time'] = pd.to_datetime(aggr_taker_vol_data['time'], unit='s')
# rename createTime to t
aggr_taker_vol_data = aggr_taker_vol_data.rename(columns={'time': 't'})
aggr_taker_vol_data = aggr_taker_vol_data.dropna()
# Display the first few rows
print("aggregatedTakerBuySellVolumeRatio/history:", aggr_taker_vol_data.head())

taker_history = requests.get(urltakerBuySellVolume, headers=headers)
taker_history_data = taker_history.json()['data']
# print(funding_data)
# Convert to DataFrame
taker_history_data = pd.DataFrame(taker_history_data, columns=['time', 'buy', 'sell'])
# Convert createTime from Unix time (seconds) to datetime
taker_history_data['time'] = pd.to_datetime(taker_history_data['time'], unit='s')
# rename createTime to t
taker_history_data = taker_history_data.rename(columns={'time': 't'})
taker_history_data = taker_history_data.dropna()
# Display the first few rows
print("aggregatedTakerBuySellVolumeRatio/history:", taker_history_data.head())

bubble = requests.get(urlBubbleIndex, headers=headers)
# print(bubble.json())
bubble_data = bubble.json()['data']
# print(funding_data)
# Convert to DataFrame
bubble_data = pd.DataFrame(bubble_data, columns=['price', 'index', 'googleTrend', 'difficulty', 'transcations', 'sentByAddress', 'tweets', 'date'])
# drop nan rows
# bubble_data = bubble_data.dropna()
bubble_data = bubble_data.reset_index(drop=True)
# Convert createTime from Unix time (seconds) to datetime
bubble_data['date'] = pd.to_datetime(bubble_data['date']).dt.date
# drop price
# bubble_data = bubble_data.drop(columns=['price'])
# rename createTime to t
bubble_data = bubble_data.rename(columns={'date': 't'})
bubble_data['t'] = pd.to_datetime(bubble_data['t'])
bubble_data = bubble_data.dropna()
# Display the first few rows
print("BUBBLE:", bubble_data.head())

ahr999 = requests.get(urlahr999, headers=headers)
# print(bubble.json())
ahr999_data = ahr999.json()['data']
# print(funding_data)
# Convert to DataFrame
ahr999_data = pd.DataFrame(ahr999_data, columns=['date', 'avg', 'ahr999', 'value'])
# drop nan rows
# ahr999_data = ahr999_data.dropna()
ahr999_data = ahr999_data.reset_index(drop=True)
# Convert createTime from Unix time (seconds) to datetime
ahr999_data['date'] = pd.to_datetime(ahr999_data['date']).dt.date
# drop price
# bubble_data = bubble_data.drop(columns=['price'])
# rename createTime to t
ahr999_data = ahr999_data.rename(columns={'date': 't'})
ahr999_data['t'] = pd.to_datetime(ahr999_data['t'])
ahr999_data = ahr999_data.dropna()
# Display the first few rows
print("ahr999:", ahr999_data.head())

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
puell_data = puell_data.dropna()
# Display the first few rows
print("PUELL:", puell_data.head())

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
# drop 
# golden_ratio_data = golden_ratio_data.drop(columns=['x5', 'x13', 'x21', 'x8'])
# fill down
golden_ratio_data = golden_ratio_data.fillna(method='ffill')
# rename createTime to t
golden_ratio_data = golden_ratio_data.rename(columns={'createTime': 't'})
golden_ratio_data['t'] = pd.to_datetime(golden_ratio_data['t'])
# Display the first few rows
print("GOLDEN RATIO:", golden_ratio_data.head())

rainbow = requests.get(urlRainbow, headers=headers)
rainbow_data = rainbow.json()['data']
# print(funding_data)
# Convert to DataFrame
rainbow_data = pd.DataFrame(rainbow_data, columns=['BTC price', 'Model price', 'Fire sale', 'Buy', 'Accumulate', 'Still cheap', 'Hold', 'Is this a bubble?', 'FOMO intensifies', 'Sell. Seriously, sell', 'Maximum bubble territory', 'createTime'])
# Convert createTime from Unix time (seconds) to datetime
rainbow_data['createTime'] = pd.to_datetime(rainbow_data['createTime'], unit='ms')
# rename createTime to t
rainbow_data = rainbow_data.rename(columns={'createTime': 't'})
rainbow_data['t'] = pd.to_datetime(rainbow_data['t'])
# Display the first few rows
print("Rainbow:", rainbow_data.head())

FearGreed = requests.get(urlFearGreed, headers=headers)
FearGreed_data = FearGreed.json()['data']
# print(funding_data)
# Convert to DataFrame
FearGreed_data = pd.DataFrame(FearGreed_data, columns=['values', 'prices', 'dates'])
# Convert createTime from Unix time (seconds) to datetime
FearGreed_data['dates'] = pd.to_datetime(FearGreed_data['dates'], unit='ms')
# rename createTime to t
FearGreed_data = FearGreed_data.rename(columns={'dates': 't'})
FearGreed_data['t'] = pd.to_datetime(FearGreed_data['t'])
print("FearGreed:", FearGreed_data.head())

# Before merging, ensure 't' columns are in datetime format and sort the dataframes
dataframes = [oi_ohlc_data, funding_data, funding_oi_df, 
              liquidation_history_data, liquidation_history_aggr_data, global_long_short_data, 
              top_long_short_data, top_long_short_position_data, aggr_taker_vol_data, taker_history_data, 
              bubble_data, ahr999_data, puell_data, stock_flow_data, pi_data, golden_ratio_data, 
              rainbow_data, FearGreed_data]

for df in dataframes:
    df['t'] = pd.to_datetime(df['t'])  # Convert to datetime
    df.sort_values(by='t', inplace=True)  # Sort by datetime

# Use merge_asof with sorted DataFrames
df_merged = dataframes[0]  # Start with the first DataFrame
for df in dataframes[1:]:  # Loop through the rest of the DataFrames and merge them one by one
    df_merged = pd.merge_asof(df_merged, df, on='t')

# After merging, you may drop rows with NaN if they are not needed
df_merged.dropna(inplace=True)

print(df_merged.head())

# If Crypto/data/coinglass_ETC.csv not found, create new df, otherwise append
if not os.path.exists('Crypto/data/coinglass_ETC.csv'):
    df_merged.to_csv('Crypto/data/coinglass_ETC.csv', index=False)
else:
    df_merged.to_csv('Crypto/data/coinglass_ETC.csv', mode='a', header=False, index=False)
