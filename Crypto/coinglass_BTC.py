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

urlSupportedCoins = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urlSupportedExchanges = "https://open-api-v3.coinglass.com/api/futures/supported-exchange-pairs"

# Open Interest
urlOIOHLC = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urlOIHistory = "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-aggregated-history?symbol=BTC&interval=4h&limit=4500"
urlExchangeListOIBTC = "https://open-api-v3.coinglass.com/api/futures/openInterest/exchange-list?symbol=BTC"

# Funding
urlFunding = "https://open-api-v3.coinglass.com/api/futures/fundingRate/ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urlFundingHistoryOIWeight = "https://open-api-v3.coinglass.com/api/futures/fundingRate/oi-weight-ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urlFundingHistoryVolWeight = "https://open-api-v3.coinglass.com/api/futures/fundingRate/vol-weight-ohlc-history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"

# Liquidation
urlLiquidation = "https://open-api-v3.coinglass.com/api/futures/liquidation/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urlLiquidationHistory = "https://open-api-v3.coinglass.com/api/futures/liquidation/aggregated-history"
urlLiquidationCoinList = "https://open-api-v3.coinglass.com/api/futures/liquidation/coin-list?symbol=BTC&range=4h"
urlLiquidationExchangeList = "https://open-api-v3.coinglass.com/api/futures/liquidation/exchange-list?symbol=BTC&range=4h"

# Long/Short
urlGlobalLongShort = "https://open-api-v3.coinglass.com/api/futures/globalLongShortAccountRatio/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urlAccountLongShort = "https://open-api-v3.coinglass.com/api/futures/topLongShortAccountRatio/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urltopLongShortPositionRatio = "https://open-api-v3.coinglass.com/api/futures/topLongShortPositionRatio/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urlaggregatedTakerBuySellVolumeRatio = "https://open-api-v3.coinglass.com/api/futures/aggregatedTakerBuySellVolumeRatio/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urltakerBuySellVolume = "https://open-api-v3.coinglass.com/api/futures/takerBuySellVolume/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"
urltakerBuySellVolume = "https://open-api-v3.coinglass.com/api/futures/takerBuySellVolume/exchange-list?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=4500"


# Indicator
urlBubbleIndex = "https://open-api-v3.coinglass.com/api/index/bitcoin-bubble-index"
urlahr999 = "https://open-api-v3.coinglass.com/api/index/ahr999"
urlPuell = "https://open-api-v3.coinglass.com/api/index/puell-multiple"
urlStockFlow = "https://open-api-v3.coinglass.com/api/index/stock-flow"
urlPi = "https://open-api-v3.coinglass.com/api/index/pi"
urlGoldenRatio = "https://open-api-v3.coinglass.com/api/index/golden-ratio-multiplier"
urlBTCProfitable = "https://open-api-v3.coinglass.com/api/index/bitcoin-profitable-days"
urlRainbow = "https://open-api-v3.coinglass.com/api/index/bitcoin-rainbow-chart"
urlFearGreed = "https://open-api-v3.coinglass.com/api/index/fear-greed-history"

# Option
urlOption = "https://open-api-v3.coinglass.com/api/option/info?symbol=BTC"


ohlc = requests.get(urlOIOHLC, headers=headers)
ohlc_data = ohlc.json()['data']
# Convert to DataFrame
ohlc_data = pd.DataFrame(ohlc_data, columns=['t', 'o', 'h', 'l', 'c'])
# Convert createTime from Unix time (seconds) to datetime
ohlc_data['t'] = pd.to_datetime(ohlc_data['t'], unit='s')
# Fill NaN values with previous values
ohlc_data = ohlc_data.fillna(method='ffill')
# drop o, h, l
# ohlc_data = ohlc_data.drop(columns=['o', 'h', 'l'])
# Display the first few rows
print("PRICE:", ohlc_data.head())

# Make requests, saved as json
oi = requests.get(urlOIHistory, headers=headers)
oi_data = oi.json()['data']
# print(oi_data)
# Convert to DataFrame
oi_data = pd.DataFrame(oi_data, columns=['t', 'o', 'h', 'l', 'c'])
# Convert createTime from Unix time (seconds) to datetime
oi_data['t'] = pd.to_datetime(oi_data['t'], unit='ms')
# drop column o, h, l
# oi_data = oi_data.drop(columns=['o', 'h', 'l'])
# oi_data = oi_data.rename(columns={'c': 'c_oi'})
# Display the first few rows
print("OI:", oi_data.head())

funding = requests.get(urlFunding, headers=headers)
funding_data = funding.json()['data']
# print(funding_data)
# Convert to DataFrame
funding_data = pd.DataFrame(funding_data, columns=['t', 'o', 'h', 'l', 'c'])
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

bubble = requests.get(urlBubbleIndex, headers=headers)
# print(bubble.json())
bubble_data = bubble.json()['data']
# print(funding_data)
# Convert to DataFrame
bubble_data = pd.DataFrame(bubble_data, columns=['price', 'index', 'googleTrend', 'difficulty', 'transcations', 'sentByAddress', 'tweets', 'date'])
# drop nan rows
bubble_data = bubble_data.dropna()
bubble_data = bubble_data.reset_index(drop=True)
# Convert createTime from Unix time (seconds) to datetime
bubble_data['date'] = pd.to_datetime(bubble_data['date']).dt.date
# drop price
# bubble_data = bubble_data.drop(columns=['price'])
# rename createTime to t
bubble_data = bubble_data.rename(columns={'date': 't'})
bubble_data['t'] = pd.to_datetime(bubble_data['t'])
# Display the first few rows
print("BUBBLE:", bubble_data.head())

# merge all dataframes
merged_df = pd.merge_asof(ohlc_data, oi_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, funding_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, liquidation_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, puell_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, pi_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, golden_ratio_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, stock_flow_data, on='t', direction='nearest')
merged_df = pd.merge_asof(merged_df, bubble_data, on='t', direction='nearest')

# Load the existing data
existing_df = pd.read_csv('Crypto/data/coinglass_BTC.csv')

new_rows = merged_df[~merged_df['t'].isin(existing_df['t'])]

# Append new rows to the CSV file, if there are any
if not new_rows.empty:
    new_rows.to_csv('Crypto/data/coinglass_BTC.csv', mode='a', header=False, index=False)

print(merged_df.head())