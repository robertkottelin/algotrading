import yfinance as yf
import pandas as pd
import os

def download_ticker_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    # Save to the specified directory
    data.to_csv(f"data/stocks/{ticker}.csv")

if __name__ == "__main__":
    # Read the CSV file
    df1 = pd.read_csv("data/Nasdaq_stocks.csv")
    df2 = pd.read_csv("data/Nyse_stocks.csv")

    # Create the directory if it doesn't exist
    if not os.path.exists("data/stocks"):
        os.makedirs("data/stocks")

    # Loop through the tickers in the dataframe and download the data
    for ticker in df1["Symbol"]:
        print(f"Downloading data for {ticker}...")
        try:
            download_ticker_data(ticker)
        except Exception as e:
            print(f"Failed to download data for {ticker}. Error: {e}")

    # Loop through the tickers in the dataframe and download the data
    for ticker in df2["Symbol"]:
        print(f"Downloading data for {ticker}...")
        try:
            download_ticker_data(ticker)
        except Exception as e:
            print(f"Failed to download data for {ticker}. Error: {e}")

    print("Download complete.")
