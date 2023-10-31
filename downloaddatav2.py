import yfinance as yf
import pandas as pd
import os

def download_ticker_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    
    # Save to the specified directory
    data.to_csv(f"data/nysestocks/{ticker}.csv")

if __name__ == "__main__":
    # Read the CSV file
    df = pd.read_csv("data/Nyse_stocks.csv")

    # Create the directory if it doesn't exist
    if not os.path.exists("data/nysestocks"):
        os.makedirs("data/nysestocks")

    # Loop through the tickers in the dataframe and download the data
    for ticker in df["Symbol"]:
        print(f"Downloading data for {ticker}...")
        try:
            download_ticker_data(ticker)
        except Exception as e:
            print(f"Failed to download data for {ticker}. Error: {e}")

    print("Download complete.")
