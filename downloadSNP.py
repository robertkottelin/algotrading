import yfinance as yf
import pandas as pd
import os

def download_ticker_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="max")
    # drop stock splits and dividends
    data.drop(columns=['Stock Splits', 'Dividends'], inplace=True)
    # Save to the specified directory
    data.to_csv(f"data/SNP/SNP.csv")

if __name__ == "__main__":
    print(f"Downloading data for GSPC...")
    download_ticker_data('^GSPC')

    print("Download complete.")
