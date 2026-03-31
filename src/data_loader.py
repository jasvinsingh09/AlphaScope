import yfinance as yf
import pandas as pd


def get_market_data():
    tickers = ["SPY", "^VIX", "QQQ", "IWM"]
    data = yf.download(tickers, period="2y", interval="1d", auto_adjust=True)

    close_prices = data["Close"].copy()
    close_prices = close_prices.dropna()

    return close_prices


if __name__ == "__main__":
    df = get_market_data()
    print(df.tail())