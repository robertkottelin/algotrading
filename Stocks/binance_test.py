import os
from binance.client import Client
from dotenv import load_dotenv

# Load API key and secret from .env file
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")
api_secret = os.getenv("BINANCE_SECRET_KEY")

# Initialize the Binance Client
client = Client(api_key, api_secret)

# Fetch BTC price history at a lower time frame (e.g., 1 minute)
btc_history = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 year ago UTC")

# Fetch funding rate (example: for BTCUSDT perpetual futures)
historical_funding_rate = client.futures_funding_rate(symbol="BTCUSDT", start_str="1 year ago UTC")

# Fetch open interest (example: for BTCUSDT perpetual futures)
open_interest = client.futures_open_interest(symbol="BTCUSDT", start_str="1 year ago UTC")

# Binance does not directly provide CVD data. It needs to be calculated using trade data.
# The calculation for CVD is complex and involves accumulating volume delta (buy volume - sell volume) over time.
# Here is a basic implementation. Note: This is a simplified version and may not be entirely accurate.
# trades = client.get_recent_trades(symbol="BTCUSDT", limit=500)
# cvd = sum([trade['quoteQty'] if trade['isBuyerMaker'] else -trade['quoteQty'] for trade in trades])

# Printing the fetched data
print("BTC Price History (LTF):", btc_history)
print("Funding Rate:", historical_funding_rate)
print("Open Interest:", open_interest)
# print("Cumulative Volume Delta (CVD):", cvd)