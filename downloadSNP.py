import yfinance as yf
import pandas as pd
import os

def download_ticker_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    # Save to the specified directory
    data.to_csv(f"data/{ticker}.csv")

if __name__ == "__main__":
    print(f"Downloading data for GSPC...")
    download_ticker_data('^GSPC')

    print("Download complete.")
