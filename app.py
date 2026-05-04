import pandas as pd
import streamlit as st

from src.data_loader import get_market_data
from src.indicators import add_indicators
from src.model import train_model
from src.sector_ranker import rank_sectors
from src.stock_analyzer import analyze_stock


st.set_page_config(
    page_title="AlphaScope",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(180deg, #081120 0%, #0b1220 100%);
}

/* Main container */
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1320px;
}

/* Typography */
h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #f8fafc !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e293b;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
}

/* Metric text */
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 700;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: transparent;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 10px;
    margin-bottom: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #111827;
    border-radius: 12px;
    color: #cbd5e1 !important;
    padding: 10px 18px;
    border: 1px solid #1e293b;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: 1px solid #3b82f6 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 16px;
    overflow: hidden;
}

/* Inputs */
[data-testid="stTextInput"] input {
    background-color: #0f172a !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1rem;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
    color: white;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Custom boxes */
.alpha-card {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.18);
}

.alpha-subtle {
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## AlphaScope")
    st.caption("Market intelligence dashboard")
    if st.button("Refresh Dashboard"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("### What it does")
    st.write(
        "Tracks market regime, sector strength, and stock-level momentum using live market data and machine learning signals."
    )

# Load core data
prices = get_market_data()
data = add_indicators(prices)
model_results = train_model()

bullish_probability = model_results["bullish_probability"]
best_model_name = model_results["model_name"]
feature_importance = model_results["feature_importance"]

# Forecast logic
if bullish_probability >= 0.65:
    forecast = "Bullish"
elif bullish_probability <= 0.35:
    forecast = "Bearish"
else:
    forecast = "Neutral"

# Sector data
sector_scores = rank_sectors()
sector_df = pd.DataFrame(sector_scores)

if sector_df.empty:
    top_sector = "Unavailable"
else:
    top_sector = sector_df.iloc[0]["Sector"]

# Header
st.markdown("""
<div style="padding: 6px 0 18px 0;">
    <h1 style="margin-bottom: 0; font-size: 3rem;">AlphaScope</h1>
    <p style="margin-top: 6px; color: #94a3b8; font-size: 1.08rem;">
        AI-Powered Market Intelligence Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("Live market data, sector rotation, and stock analysis in one place.")

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Best Model", best_model_name)
col2.metric("Bullish Probability", f"{bullish_probability:.2%}")
col3.metric("Forecast", forecast)
col4.metric("Strongest Sector", top_sector)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Market Overview",
    "Sector Rotation",
    "Stock Analyzer",
    "Forecast Drivers"
])

with tab1:
    st.markdown("### Market Forecast")

    if forecast == "Bullish":
        st.success(
            "The model sees a bullish short-term outlook. Positive momentum and stronger risk appetite are supporting a higher probability of upside over the next 10 trading days."
        )
    elif forecast == "Bearish":
        st.error(
            "The model sees a bearish short-term outlook. Weak momentum and more defensive market conditions are increasing downside risk over the next 10 trading days."
        )
    else:
        st.warning(
            "The model sees a neutral short-term outlook. Current signals are mixed and do not strongly favor either bulls or bears."
        )

    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.markdown("### Major Index Performance")
        st.line_chart(prices[["SPY", "QQQ", "IWM"]], height=350)

    with chart_col2:
        st.markdown("### Volatility Index")
        st.line_chart(prices["^VIX"], height=350)

    st.markdown("### Latest Indicator Snapshot")

    latest = data.iloc[-1][[
        "SPY_Return_5D",
        "SPY_Return_10D",
        "SPY_Return_20D",
        "SPY_vs_MA20",
        "SPY_vs_MA50",
        "SPY_Vol_20D",
        "RSI_14",
        "VIX_Level",
        "VIX_Change_5D",
        "QQQ_vs_SPY",
        "IWM_vs_SPY"
    ]]

    latest_df = latest.reset_index()
    latest_df.columns = ["Indicator", "Value"]
    latest_df["Value"] = latest_df["Value"].apply(lambda x: round(float(x), 4))

    st.dataframe(latest_df, width="stretch", hide_index=True)

    st.markdown("### Signal Summary")

    rsi = float(latest["RSI_14"])
    vix_change = float(latest["VIX_Change_5D"])
    spy_20d = float(latest["SPY_Return_20D"])

    signal_col1, signal_col2, signal_col3 = st.columns(3)

    with signal_col1:
        if rsi > 70:
            st.warning("RSI is elevated and may indicate overbought conditions.")
        elif rsi < 30:
            st.warning("RSI is depressed and may indicate oversold conditions.")
        else:
            st.info("RSI is in a neutral range.")

    with signal_col2:
        if vix_change > 0.10:
            st.error("VIX has risen sharply, signaling higher market fear.")
        else:
            st.success("VIX is stable or falling, signaling calmer conditions.")

    with signal_col3:
        if spy_20d > 0:
            st.success("SPY maintains positive 20-day momentum.")
        else:
            st.error("SPY has negative 20-day momentum.")

with tab2:
    st.markdown("### Sector Rankings")

    if sector_df.empty:
        st.warning("Sector data is temporarily unavailable.")
    else:
        sector_df["5D Return"] = sector_df["5D Return"].apply(lambda x: round(float(x) * 100, 2))
        sector_df["20D Return"] = sector_df["20D Return"].apply(lambda x: round(float(x) * 100, 2))
        sector_df["Relative Strength vs SPY"] = sector_df["Relative Strength vs SPY"].apply(lambda x: round(float(x) * 100, 2))
        sector_df["Score"] = sector_df["Score"].apply(lambda x: round(float(x), 4))

        st.dataframe(sector_df, width="stretch", hide_index=True)

        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            st.markdown("### Sector Strength")
            sector_chart = sector_df.set_index("Sector")["Score"]
            st.bar_chart(sector_chart, height=350)

        with chart_col2:
            bottom_sector = sector_df.iloc[-1]["Sector"]

            st.markdown("### Sector Takeaway")
            st.success(f"Strongest sector: {top_sector}")
            st.error(f"Weakest sector: {bottom_sector}")

            st.info(
                "This ranking blends recent returns and relative strength versus SPY to highlight current market leadership."
            )

with tab3:
    st.markdown("### Stock Analyzer")

    ticker_input = st.text_input("Enter a stock ticker", value="AAPL")

    if ticker_input:
        try:
            stock_data = analyze_stock(ticker_input)

            stock_df = pd.DataFrame(
                list(stock_data.items()),
                columns=["Metric", "Value"]
            )

            st.dataframe(stock_df, width="stretch", hide_index=True)

            trend = stock_data["Trend"]
            rsi_signal = stock_data["RSI Signal"]
            recommendation = stock_data["Recommendation"]
            score = stock_data["Score"]

            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.markdown("### Recommendation")

                if recommendation == "Buy":
                    st.success(f"Recommendation: BUY (Score: {score})")
                    st.success(
                        "The stock shows constructive momentum, supportive moving averages, and favorable technical conditions."
                    )
                elif recommendation == "Hold":
                    st.warning(f"Recommendation: HOLD (Score: {score})")
                    st.warning(
                        "Signals are mixed. Some indicators are supportive, but trend confirmation is not strong enough for a higher-conviction call."
                    )
                else:
                    st.error(f"Recommendation: SELL (Score: {score})")
                    st.error(
                        "The stock shows weaker momentum, less supportive moving averages, or deteriorating technical conditions."
                    )

            with rec_col2:
                st.markdown("### Stock Summary")

                if trend == "Strong Uptrend":
                    st.success(f"{ticker_input.upper()} is in a strong uptrend.")
                elif trend == "Strong Downtrend":
                    st.error(f"{ticker_input.upper()} is in a strong downtrend.")
                else:
                    st.info(f"{ticker_input.upper()} is showing a mixed trend.")

                if rsi_signal == "Overbought":
                    st.warning(f"{ticker_input.upper()} may be overbought based on RSI.")
                elif rsi_signal == "Oversold":
                    st.warning(f"{ticker_input.upper()} may be oversold based on RSI.")
                else:
                    st.info(f"{ticker_input.upper()} has a neutral RSI signal.")

        except Exception as e:
            st.error(f"Unable to analyze ticker: {e}")

with tab4:
    st.markdown("### Forecast Drivers")

    insight_col1, insight_col2 = st.columns([1, 2])

    with insight_col1:
        st.metric("Selected Model", best_model_name)
        st.metric("Bullish Probability", f"{bullish_probability:.2%}")

        if bullish_probability >= 0.65:
            st.success("High-conviction bullish setup.")
        elif bullish_probability <= 0.35:
            st.error("High-conviction bearish setup.")
        else:
            st.warning("Mixed setup with lower conviction.")

        st.info(
            "This forecast reflects the model’s estimate of the probability that SPY is higher over the next 10 trading days."
        )

    with insight_col2:
        st.markdown("### Top Feature Importance")
        top_features = feature_importance.head(10).set_index("Feature")
        st.bar_chart(top_features, height=350)

    st.markdown("### How to Read AlphaScope")
    st.write(
        "AlphaScope combines live market data, technical indicators, sector rotation, and machine learning forecasts into a single decision-support dashboard. "
        "Instead of only showing charts, it summarizes what the market looks like, where leadership is emerging, and how individual stocks are positioned."
    )