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

urlOHLC = "https://open-api.coinglass.com/public/v2/indicator/price_ohlc?ex=Binance&pair=BTCUSDT&interval=h4&limit=4500"
urlOI = "https://open-api.coinglass.com/public/v2/indicator/open_interest_ohlc?ex=Binance&pair=BTCUSDT&interval=h4&limit=4500"
urlFunding = "https://open-api.coinglass.com/public/v2/indicator/funding_avg?symbol=BTC&interval=h4&limit=4500"
urlLiquidation = "https://open-api.coinglass.com/public/v2/indicator/liquidation_pair?ex=Binance&pair=BTCUSDT&interval=h4&limit=4500"
urlLongShort = "https://open-api.coinglass.com/public/v2/indicator/top_long_short_position_ratio?ex=Binance&pair=BTCUSDT&interval=h4&limit=4500"
urlPuell = "https://open-api.coinglass.com/public/v2/index/puell_multiple"
urlPi = "https://open-api.coinglass.com/public/v2/index/pi"
urlGoldenRatio = "https://open-api.coinglass.com/public/v2/index/golden_ratio_multiplier"
urlStockFlow = "https://open-api.coinglass.com/public/v2/index/stock_flow"
urlBubbleIndex = "https://open-api-v3.coinglass.com/api/index/bitcoin-bubble-index"

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