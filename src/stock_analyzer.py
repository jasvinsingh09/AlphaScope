import yfinance as yf
import pandas as pd


def calculate_rsi(series, window=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def analyze_stock(ticker):
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=True)

    if df.empty:
        raise ValueError("No data found for this ticker.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        raise ValueError("Close price data not found.")

    df["Return_20D"] = df["Close"].pct_change(20)
    df["Return_60D"] = df["Close"].pct_change(60)

    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()

    df["RSI"] = calculate_rsi(df["Close"])

    df = df.dropna()

    if df.empty:
        raise ValueError("Not enough data to analyze this ticker.")

    latest = df.iloc[-1]

    current_price = float(latest["Close"])
    return_20d = float(latest["Return_20D"])
    return_60d = float(latest["Return_60D"])
    ma_50 = float(latest["MA_50"])
    ma_200 = float(latest["MA_200"])
    rsi = float(latest["RSI"])

    if current_price > ma_50 and ma_50 > ma_200:
        trend = "Strong Uptrend"
    elif current_price < ma_50 and ma_50 < ma_200:
        trend = "Strong Downtrend"
    else:
        trend = "Mixed Trend"

    if rsi > 70:
        rsi_signal = "Overbought"
    elif rsi < 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"
        
    score = 0

    if current_price > ma_50:
        score += 1
    else:
        score -= 1

    if ma_50 > ma_200:
        score += 1
    else:
        score -= 1

    if return_20d > 0:
        score += 1
    else:
        score -= 1

    if return_60d > 0:
        score += 1
    else:
        score -= 1

    if 40 <= rsi <= 70:
        score += 1
    elif rsi < 30:
        score += 0.5
    else:
        score -= 1

    if score >= 3:
        recommendation = "Buy"
    elif score >= 1:
        recommendation = "Hold"
    else:
        recommendation = "Sell"

    return {
        "Ticker": ticker.upper(),
        "Current Price": round(current_price, 2),
        "20D Return (%)": round(return_20d * 100, 2),
        "60D Return (%)": round(return_60d * 100, 2),
        "50D MA": round(ma_50, 2),
        "200D MA": round(ma_200, 2),
        "RSI": round(rsi, 2),
        "Trend": trend,
        "RSI Signal": rsi_signal,
        "Score": round(score, 2),
        "Recommendation": recommendation
        
    }
