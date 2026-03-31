import yfinance as yf


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


def get_market_data_for_sectors(tickers):
    data = yf.download(tickers, period="6mo", interval="1d", auto_adjust=True)
    close_prices = data["Close"].dropna()
    return close_prices


def rank_sectors():
    tickers = list(SECTOR_ETFS.values()) + ["SPY"]
    prices = get_market_data_for_sectors(tickers)

    sector_scores = []
    spy_20d = prices["SPY"].pct_change(20).iloc[-1]

    for sector_name, ticker in SECTOR_ETFS.items():
        sector_20d = prices[ticker].pct_change(20).iloc[-1]
        sector_5d = prices[ticker].pct_change(5).iloc[-1]
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