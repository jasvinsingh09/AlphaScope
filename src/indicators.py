import numpy as np


def calculate_rsi(series, window=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_indicators(df):
    df = df.copy()

    # Returns
    df["SPY_Return_1D"] = df["SPY"].pct_change()
    df["SPY_Return_5D"] = df["SPY"].pct_change(5)
    df["SPY_Return_10D"] = df["SPY"].pct_change(10)
    df["SPY_Return_20D"] = df["SPY"].pct_change(20)
    df["SPY_Return_50D"] = df["SPY"].pct_change(50)

    # Moving averages
    df["SPY_MA_10"] = df["SPY"].rolling(10).mean()
    df["SPY_MA_20"] = df["SPY"].rolling(20).mean()
    df["SPY_MA_50"] = df["SPY"].rolling(50).mean()
    df["SPY_MA_200"] = df["SPY"].rolling(200).mean()

    # Distance from moving averages
    df["SPY_vs_MA10"] = df["SPY"] / df["SPY_MA_10"] - 1
    df["SPY_vs_MA20"] = df["SPY"] / df["SPY_MA_20"] - 1
    df["SPY_vs_MA50"] = df["SPY"] / df["SPY_MA_50"] - 1
    df["SPY_vs_MA200"] = df["SPY"] / df["SPY_MA_200"] - 1

    # Volatility
    df["SPY_Vol_10D"] = df["SPY_Return_1D"].rolling(10).std() * np.sqrt(252)
    df["SPY_Vol_20D"] = df["SPY_Return_1D"].rolling(20).std() * np.sqrt(252)

    # RSI
    df["RSI_14"] = calculate_rsi(df["SPY"], 14)

    # VIX features
    df["VIX_Level"] = df["^VIX"]
    df["VIX_Change_5D"] = df["^VIX"].pct_change(5)
    df["VIX_Change_20D"] = df["^VIX"].pct_change(20)
    df["VIX_MA_20"] = df["^VIX"].rolling(20).mean()
    df["VIX_vs_MA20"] = df["^VIX"] / df["VIX_MA_20"] - 1

    # Cross-market features
    df["QQQ_Return_20D"] = df["QQQ"].pct_change(20)
    df["IWM_Return_20D"] = df["IWM"].pct_change(20)

    df["QQQ_vs_SPY"] = df["QQQ_Return_20D"] - df["SPY_Return_20D"]
    df["IWM_vs_SPY"] = df["IWM_Return_20D"] - df["SPY_Return_20D"]

    # Trend regime flags
    df["MA_Bullish"] = (df["SPY_MA_50"] > df["SPY_MA_200"]).astype(int)
    df["Price_Above_MA50"] = (df["SPY"] > df["SPY_MA_50"]).astype(int)
    df["Price_Above_MA200"] = (df["SPY"] > df["SPY_MA_200"]).astype(int)

    return df