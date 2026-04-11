import yfinance as yf
import streamlit as st
import pandas as pd


SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}


@st.cache_data(ttl=300)
def get_market_data_for_sectors(tickers):
    data = yf.download(
        tickers,
        period="6mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False
    )

    if data.empty:
        return pd.DataFrame()

    close_frames = {}

    for ticker in tickers:
        try:
            if ticker in data.columns.get_level_values(0):
                close_frames[ticker] = data[ticker]["Close"]
        except Exception:
            pass

    if not close_frames:
        return pd.DataFrame()

    close_prices = pd.DataFrame(close_frames).dropna(how="all")
    return close_prices


def rank_sectors():
    tickers = list(SECTOR_ETFS.values()) + ["SPY"]
    prices = get_market_data_for_sectors(tickers)

    if prices.empty:
        return []

    if "SPY" not in prices.columns:
        return []

    spy_series = prices["SPY"].dropna()
    if len(spy_series) < 21:
        return []

    spy_20d = spy_series.pct_change(20).dropna().iloc[-1]

    sector_scores = []

    for sector_name, ticker in SECTOR_ETFS.items():
        if ticker not in prices.columns:
            continue

        sector_series = prices[ticker].dropna()
        if len(sector_series) < 21:
            continue

        sector_20d_series = sector_series.pct_change(20).dropna()
        sector_5d_series = sector_series.pct_change(5).dropna()

        if sector_20d_series.empty or sector_5d_series.empty:
            continue

        sector_20d = sector_20d_series.iloc[-1]
        sector_5d = sector_5d_series.iloc[-1]
        relative_strength = sector_20d - spy_20d

        score = (
            0.5 * sector_20d +
            0.3 * sector_5d +
            0.2 * relative_strength
        )

        sector_scores.append({
            "Sector": sector_name,
            "Ticker": ticker,
            "5D Return": sector_5d,
            "20D Return": sector_20d,
            "Relative Strength vs SPY": relative_strength,
            "Score": score
        })

    sector_scores = sorted(sector_scores, key=lambda x: x["Score"], reverse=True)
    return sector_scores