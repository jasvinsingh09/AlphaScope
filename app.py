import streamlit as st
from src.data_loader import get_market_data
from src.indicators import add_indicators
from src.model import train_model
from src.sector_ranker import rank_sectors
import pandas as pd
from src.stock_analyzer import analyze_stock

st.set_page_config(page_title="Market Forecast Agent", layout="wide")

st.title("AI Market Forecast Agent")
st.write("Simple market outlook model using SPY, VIX, QQQ, and IWM.")

# Load market data
prices = get_market_data()

# Add indicators
data = add_indicators(prices)

# Train model
model, accuracy, bullish_probability, df = train_model()

# Forecast label
if bullish_probability > 0.60:
    forecast = "Bullish"
elif bullish_probability < 0.40:
    forecast = "Bearish"
else:
    forecast = "Neutral"

# Top metrics
col1, col2, col3 = st.columns(3)

col1.metric("Model Accuracy", f"{accuracy:.2%}")
col2.metric("Bullish Probability", f"{bullish_probability:.2%}")
col3.metric("Forecast", forecast)

# Market charts
st.subheader("SPY, QQQ, and IWM")
st.line_chart(prices[["SPY", "QQQ", "IWM"]])

st.subheader("VIX")
st.line_chart(prices["^VIX"])

# Latest indicators
st.subheader("Latest Indicator Snapshot")

latest = data.iloc[-1][[
    "SPY_Return_5D",
    "SPY_Return_20D",
    "SPY_vs_MA10",
    "SPY_vs_MA50",
    "SPY_Vol_20D",
    "RSI_14",
    "VIX_Change_5D",
    "QQQ_Return_20D",
    "IWM_Return_20D"
]]

latest_df = latest.reset_index()
latest_df.columns = ["Indicator", "Value"]

latest_df["Value"] = latest_df["Value"].apply(lambda x: round(float(x), 4))

st.dataframe(latest_df, use_container_width=True)
st.subheader("Market Signal Summary")

rsi = latest["RSI_14"]
vix_change = latest["VIX_Change_5D"]
spy_return = latest["SPY_Return_20D"]

if rsi > 70:
    st.warning("SPY RSI is above 70, which may indicate overbought conditions.")
elif rsi < 30:
    st.warning("SPY RSI is below 30, which may indicate oversold conditions.")
else:
    st.info("SPY RSI is in a neutral range.")

if vix_change > 0.10:
    st.error("VIX has risen sharply over the past 5 days, signaling higher market fear.")
else:
    st.success("VIX is relatively stable or falling, signaling calmer market conditions.")

if spy_return > 0:
    st.success("SPY has positive 20-day momentum.")
else:
    st.error("SPY has negative 20-day momentum.")

# Forecast explanation
st.subheader("Forecast Explanation")

if forecast == "Bullish":
    st.success(
        "The model sees a bullish short-term outlook. "
        "This is likely driven by stronger momentum, improving volatility conditions, "
        "and positive signals from QQQ and IWM."
    )
elif forecast == "Bearish":
    st.error(
        "The model sees a bearish short-term outlook. "
        "This may be driven by weak recent returns, rising volatility, "
        "and poor momentum in QQQ and IWM."
    )
else:
    st.warning(
        "The model sees a neutral short-term outlook. "
        "Signals are mixed and do not strongly point bullish or bearish."
    )
    st.subheader("Sector Rankings")

sector_scores = rank_sectors()

sector_df = pd.DataFrame(sector_scores)

sector_df["5D Return"] = sector_df["5D Return"].apply(lambda x: round(float(x) * 100, 2))
sector_df["20D Return"] = sector_df["20D Return"].apply(lambda x: round(float(x) * 100, 2))
sector_df["Relative Strength vs SPY"] = sector_df["Relative Strength vs SPY"].apply(lambda x: round(float(x) * 100, 2))
sector_df["Score"] = sector_df["Score"].apply(lambda x: round(float(x), 4))

st.dataframe(sector_df, width="stretch")

top_sector = sector_df.iloc[0]["Sector"]
bottom_sector = sector_df.iloc[-1]["Sector"]

st.subheader("Sector Takeaway")
st.success(f"Strongest sector right now: {top_sector}")
st.error(f"Weakest sector right now: {bottom_sector}")
st.subheader("Stock Analyzer")

ticker_input = st.text_input("Enter a stock ticker", value="AAPL")

if ticker_input:
    try:
        stock_data = analyze_stock(ticker_input)

        stock_df = pd.DataFrame(
            list(stock_data.items()),
            columns=["Metric", "Value"]
        )

        st.dataframe(stock_df, width="stretch")

        trend = stock_data["Trend"]
        recommendation = stock_data["Recommendation"]
        score = stock_data["Score"]
        rsi_signal = stock_data["RSI Signal"]
        
        st.subheader("Recommendation")

        if recommendation == "Buy":
            st.success(f"Recommendation: BUY (Score: {score})")
            st.success("The stock has a postive momentum, strong moving average support, and favorable RSI conditions.")
        elif recommendation == "Hold":
         st.warning(f"Recommendation: HOLD (Score: {score})")
         st.warning("The stock has mixed signals. Some indicators are postive, but momentum is not fully confirmed.")
        else:
            st.error(f"Recommendation: SELL (Score: {score})")
            st.error("The stock has more weak momentum, poor recent returns, or unfavorable moving average trends.")

        st.subheader("Stock Summary")

        if trend == "Strong Uptrend":
            st.success(f"{ticker_input.upper()} is currently in a strong uptrend.")
        elif trend == "Strong Downtrend":
            st.error(f"{ticker_input.upper()} is currently in a strong downtrend.")
        else:
            st.info(f"{ticker_input.upper()} has a mixed trend.")

        if rsi_signal == "Overbought":
            st.warning(f"{ticker_input.upper()} may be overbought based on RSI.")
        elif rsi_signal == "Oversold":
            st.warning(f"{ticker_input.upper()} may be oversold based on RSI.")
        else:
            st.info(f"{ticker_input.upper()} has a neutral RSI signal.")

    except:
        st.error("Unable to analyze ticker. Please enter a valid stock symbol.")