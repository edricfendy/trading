"""
Sentiment & Catalyst Analysis Module

Fetches recent news via Google News RSS and classifies sentiment
using keyword-based analysis. No API key required.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import requests

# ─── Keyword dictionaries ────────────────────────────────────────────────────

POSITIVE_KEYWORDS = [
    # Business expansion
    "expansion", "ekspansi", "acquire", "akuisisi", "merger", "partnership",
    "kerjasama", "new plant", "pabrik baru", "capacity", "kapasitas",
    "diversif", "new product", "produk baru", "launch", "peluncuran",
    "contract", "kontrak", "deal", "MoU", "agreement",
    # Financial performance
    "profit", "laba", "revenue growth", "pendapatan naik", "earnings beat",
    "record high", "rekor", "dividend", "dividen", "buyback", "buy back",
    "upgrade", "outperform", "overweight", "target price raised",
    "strong result", "beat estimate", "exceed", "melampaui",
    # Sector/macro positive
    "demand increase", "permintaan naik", "price surge", "harga naik",
    "bullish", "rally", "recovery", "pemulihan", "boom", "growth",
    "pertumbuhan", "optimis", "positive outlook",
    # Commodity positive
    "gold price rise", "harga emas naik", "nickel surge", "coal price up",
    "CPO price", "palm oil demand", "commodity boom", "oil price rise",
    "copper demand", "EV demand", "battery metal",
]

NEGATIVE_KEYWORDS = [
    # Business risk
    "free float increase", "dilution", "dilusi", "rights issue",
    "secondary offering", "penawaran saham", "insider sell", "divestasi",
    "divest", "downsize", "restructur", "restrukturisasi", "layoff", "PHK",
    "default", "gagal bayar", "fraud", "korupsi", "corruption",
    "investigation", "investigasi", "lawsuit", "gugatan",
    # Financial negative
    "loss", "rugi", "revenue decline", "pendapatan turun", "miss estimate",
    "below expectation", "downgrade", "underperform", "underweight",
    "target price cut", "debt concern", "utang", "cash burn",
    "negative cashflow", "arus kas negatif", "impairment",
    # Sector/macro negative
    "demand decrease", "permintaan turun", "price drop", "harga turun",
    "bearish", "crash", "recession", "resesi", "slowdown", "perlambatan",
    "oversupply", "kelebihan pasokan", "trade war", "tariff",
    # Commodity negative
    "gold price drop", "nickel slump", "coal price down", "CPO decline",
    "oil price crash", "commodity slump",
    # Regulatory risk
    "regulation tighten", "regulasi ketat", "tax increase", "pajak naik",
    "ban", "larangan", "restriction", "pembatasan",
]

CATEGORY_PATTERNS = {
    "company_action": [
        "expansion", "ekspansi", "acquire", "akuisisi", "merger", "buyback",
        "rights issue", "free float", "divest", "new plant", "contract",
        "launch", "peluncuran", "partnership", "kerjasama",
    ],
    "sector_demand": [
        "demand", "permintaan", "supply", "pasokan", "sector", "sektor",
        "industry", "industri", "market share", "pangsa pasar",
    ],
    "commodity": [
        "gold", "emas", "nickel", "nikel", "coal", "batubara", "CPO",
        "palm oil", "sawit", "oil", "minyak", "copper", "tembaga",
        "tin", "timah", "battery", "EV", "commodity", "komoditas",
    ],
    "ownership_change": [
        "free float", "insider", "major shareholder", "pemegang saham",
        "dilution", "dilusi", "rights issue", "buyback", "buy back",
        "secondary offering",
    ],
    "macro": [
        "GDP", "PDB", "inflation", "inflasi", "interest rate", "suku bunga",
        "BI rate", "Fed", "trade war", "tariff", "recession", "resesi",
        "currency", "rupiah", "IDR", "dollar", "fiscal", "fiskal",
    ],
    "regulatory": [
        "regulation", "regulasi", "OJK", "tax", "pajak", "policy",
        "kebijakan", "ban", "larangan", "restriction", "pembatasan",
        "compliance", "kepatuhan",
    ],
}


@dataclass
class SentimentItem:
    headline: str
    date: str
    sentiment: str          # "positive" or "negative" or "neutral"
    category: str           # company_action, sector_demand, commodity, etc.
    relevance_score: float  # 0-1 how many keywords matched


# ─── RSS Fetching ────────────────────────────────────────────────────────────

_RSS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


def _fetch_google_news_rss(query: str, max_items: int = 15) -> list[dict]:
    """Fetch news headlines from Google News RSS."""
    encoded = quote_plus(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=ID&ceid=ID:en"
    try:
        resp = requests.get(url, timeout=10, headers=_RSS_HEADERS)
        if not resp.ok:
            return []
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall(".//item"):
            title_el = item.find("title")
            pubdate_el = item.find("pubDate")
            if title_el is None or title_el.text is None:
                continue
            title = title_el.text.strip()
            # Clean HTML tags from title
            title = re.sub(r"<[^>]+>", "", title)
            pub_date = ""
            if pubdate_el is not None and pubdate_el.text:
                try:
                    dt = datetime.strptime(
                        pubdate_el.text.strip(),
                        "%a, %d %b %Y %H:%M:%S %Z",
                    )
                    pub_date = dt.strftime("%Y-%m-%d")
                except ValueError:
                    pub_date = pubdate_el.text.strip()[:10]
            items.append({"title": title, "date": pub_date})
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


# ─── Sentiment Classification ────────────────────────────────────────────────

def _classify_headline(headline: str) -> tuple[str, float, str]:
    """
    Classify a headline as positive/negative/neutral.
    Returns (sentiment, relevance_score, category).
    """
    lower = headline.lower()
    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw.lower() in lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw.lower() in lower)

    total = pos_count + neg_count
    if total == 0:
        return "neutral", 0.0, "general"

    relevance = min(1.0, total / 5.0)

    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    # Determine category
    category = "general"
    best_match = 0
    for cat, patterns in CATEGORY_PATTERNS.items():
        matches = sum(1 for p in patterns if p.lower() in lower)
        if matches > best_match:
            best_match = matches
            category = cat

    return sentiment, relevance, category


def fetch_ticker_sentiment(
    ticker: str,
    company_name: Optional[str] = None,
    max_items: int = 10,
) -> list[SentimentItem]:
    """
    Fetch and classify news sentiment for a specific ticker.
    """
    code = ticker.replace(".JK", "").upper()
    queries = [f"{code} stock IDX Indonesia"]
    if company_name:
        queries.append(f"{company_name} stock")

    seen_titles: set[str] = set()
    results: list[SentimentItem] = []

    for query in queries:
        items = _fetch_google_news_rss(query, max_items=max_items)
        for item in items:
            title = item["title"]
            if title in seen_titles:
                continue
            seen_titles.add(title)
            sentiment, relevance, category = _classify_headline(title)
            results.append(SentimentItem(
                headline=title,
                date=item["date"],
                sentiment=sentiment,
                category=category,
                relevance_score=relevance,
            ))

    # Sort: relevant first, then by date
    results.sort(key=lambda x: (-x.relevance_score, x.date), reverse=False)
    return results[:max_items]


def fetch_sector_sentiment(
    sector: str,
    max_items: int = 10,
) -> list[SentimentItem]:
    """
    Fetch and classify news sentiment for a sector.
    """
    query = f"{sector} sector Indonesia stock market"
    items = _fetch_google_news_rss(query, max_items=max_items)

    results: list[SentimentItem] = []
    for item in items:
        sentiment, relevance, category = _classify_headline(item["title"])
        results.append(SentimentItem(
            headline=item["title"],
            date=item["date"],
            sentiment=sentiment,
            category=category,
            relevance_score=relevance,
        ))

    results.sort(key=lambda x: (-x.relevance_score, x.date), reverse=False)
    return results[:max_items]


def sentiment_summary(items: list[SentimentItem]) -> dict:
    """
    Summarize sentiment items into counts and overall sentiment.
    """
    pos = sum(1 for i in items if i.sentiment == "positive")
    neg = sum(1 for i in items if i.sentiment == "negative")
    neu = sum(1 for i in items if i.sentiment == "neutral")
    total = len(items)

    if total == 0:
        overall = "neutral"
    elif pos > neg:
        overall = "positive"
    elif neg > pos:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "total": total,
        "overall": overall,
    }
