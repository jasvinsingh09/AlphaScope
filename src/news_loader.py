import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

RSS_FEEDS = {
    "wsj": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "bloomberg": "https://www.bloomberg.com/markets/rss.xml"
}

POSITIVE_KEYWORDS = [
    "gain", "rise", "up", "positive", "strong", "outperform", "bull", "record", "beat", "surge", "optimism", "rally"
]

NEGATIVE_KEYWORDS = [
    "drop", "fall", "down", "weak", "negative", "miss", "selloff", "bear", "plunge", "recession", "fear", "slump"
]


def _parse_pub_date(pub_date_text):
    if not pub_date_text:
        return None

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(pub_date_text.strip(), fmt)
        except ValueError:
            continue

    return None


def _score_headline(text):
    if not text:
        return 0.0

    normalized = text.lower()
    positive = sum(1 for word in POSITIVE_KEYWORDS if word in normalized)
    negative = sum(1 for word in NEGATIVE_KEYWORDS if word in normalized)

    if positive + negative == 0:
        return 0.0

    return (positive - negative) / (positive + negative)


def _fetch_feed_items(feed_url, max_items=50):
    try:
        response = requests.get(feed_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()
    except Exception:
        return []

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        return []

    items = []
    for element in root.findall(".//item")[:max_items]:
        title = element.findtext("title") or ""
        pub_date = element.findtext("pubDate") or element.findtext("{http://purl.org/dc/elements/1.1/}date")
        parsed_date = _parse_pub_date(pub_date)

        items.append({
            "title": title,
            "date": parsed_date,
            "score": _score_headline(title),
        })

    return [item for item in items if item["date"] is not None]


def get_news_sentiment_history(days=30):
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    records = []
    for source, feed_url in RSS_FEEDS.items():
        feed_items = _fetch_feed_items(feed_url, max_items=100)
        for item in feed_items:
            pub_date = item["date"].date()
            if pub_date < start_date or pub_date > end_date:
                continue

            records.append(
                {
                    "date": pub_date,
                    "source": source,
                    "score": item["score"],
                }
            )

    if not records:
        return None

    data = {}
    for record in records:
        key = (record["date"], record["source"])
        data.setdefault(key, []).append(record["score"])

    sentiment_rows = []
    for (date, source), scores in data.items():
        sentiment_rows.append(
            {
                "date": date,
                f"news_sentiment_{source}": float(np.mean(scores)),
                f"news_headline_count_{source}": len(scores),
            }
        )

    sentiment_df = pd.DataFrame(sentiment_rows)
    sentiment_df = sentiment_df.groupby("date").sum(min_count=1)

    sentiment_df["news_sentiment_wsj"] = sentiment_df.get("news_sentiment_wsj", 0.0).fillna(0.0)
    sentiment_df["news_sentiment_bloomberg"] = sentiment_df.get("news_sentiment_bloomberg", 0.0).fillna(0.0)
    sentiment_df["news_headline_count_wsj"] = sentiment_df.get("news_headline_count_wsj", 0).fillna(0).astype(int)
    sentiment_df["news_headline_count_bloomberg"] = sentiment_df.get("news_headline_count_bloomberg", 0).fillna(0).astype(int)

    sentiment_df["news_sentiment_combined"] = (
        sentiment_df["news_sentiment_wsj"] + sentiment_df["news_sentiment_bloomberg"]
    ) / 2
    sentiment_df["news_headline_count_combined"] = (
        sentiment_df["news_headline_count_wsj"] + sentiment_df["news_headline_count_bloomberg"]
    )

    return sentiment_df


def get_recent_market_news_signals():
    history = get_news_sentiment_history(days=14)
    if history is None or history.empty:
        return {
            "news_sentiment_wsj": 0.0,
            "news_sentiment_bloomberg": 0.0,
            "news_sentiment_combined": 0.0,
            "news_headline_count_combined": 0,
        }

    last_row = history.sort_index().iloc[-1]
    return {
        "news_sentiment_wsj": float(last_row.get("news_sentiment_wsj", 0.0)),
        "news_sentiment_bloomberg": float(last_row.get("news_sentiment_bloomberg", 0.0)),
        "news_sentiment_combined": float(last_row.get("news_sentiment_combined", 0.0)),
        "news_headline_count_combined": int(last_row.get("news_headline_count_combined", 0)),
    }
