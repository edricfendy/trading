"""
Indonesia Stock Data Fetcher - Real-time and historical OHLCV data
"""
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

from config import IDX_STOCKS, get_data_period
from universe import get_universe

DOWNLOAD_BATCH_SIZE = 25
SLEEP_BETWEEN_BATCHES_SEC = 1.0


def _resolve_period(period: Optional[str], interval: str) -> str:
    return period if period else get_data_period(interval)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    return df


def _chunked(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _download_batch(
    tickers: list[str],
    period: Optional[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    period = _resolve_period(period, interval)
    data = yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=False,  # reduce parallel requests to avoid rate limits
        progress=False,
    )

    if data is None or data.empty:
        return {}

    results: dict[str, pd.DataFrame] = {}
    if isinstance(data.columns, pd.MultiIndex):
        level0 = data.columns.get_level_values(0)
        if tickers[0] in set(level0):
            # group_by="ticker": columns are (TICKER, field)
            for t in tickers:
                if t in level0:
                    sub = data[t].dropna(how="all")
                    if not sub.empty:
                        results[t] = _normalize_ohlcv(sub)
        else:
            # fallback: try (field, TICKER)
            level1 = data.columns.get_level_values(1)
            for t in tickers:
                if t in level1:
                    sub = data.xs(t, axis=1, level=1, drop_level=True).dropna(how="all")
                    if not sub.empty:
                        results[t] = _normalize_ohlcv(sub)
    else:
        # single ticker
        results[tickers[0]] = _normalize_ohlcv(data)

    return results


def fetch_stock_data(
    ticker: str,
    period: Optional[str] = "3mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single Indonesia stock.
    ticker should include .JK suffix (e.g., BBCA.JK)
    """
    try:
        batch = _download_batch([ticker], period=period, interval=interval)
    except YFRateLimitError:
        raise

    df = batch.get(ticker, pd.DataFrame())
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    return df


def fetch_multiple_stocks(
    tickers: Optional[list[str]] = None,
    period: Optional[str] = "3mo",
    interval: str = "1d",
    all_idx: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch data for multiple Indonesia stocks.
    If tickers is None, uses:
      - all_idx=True: dynamic full IDX universe (fallback to IDX_STOCKS)
      - all_idx=False: static IDX_STOCKS list only
    Returns dict of {ticker: DataFrame}
    """
    if tickers is None:
        tickers = get_universe(all_idx=all_idx)

    tickers = [t for t in (tickers or []) if isinstance(t, str) and t.strip()]
    if not tickers:
        return {}

    results: dict[str, pd.DataFrame] = {}
    batches = list(_chunked(tickers, DOWNLOAD_BATCH_SIZE))
    for i, batch in enumerate(batches):
        try:
            batch_data = _download_batch(batch, period=period, interval=interval)
        except YFRateLimitError:
            # bubble up to UI for a friendly message
            raise
        results.update(batch_data)
        if i < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    return results


def fetch_intraday(ticker: str, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch intraday data for shorter-term analysis.
    Note: yfinance intraday has limitations; 1h is typically available.
    """
    try:
        batch = _download_batch([ticker], period="5d", interval=interval)
    except YFRateLimitError:
        raise
    df = batch.get(ticker, pd.DataFrame())
    if df.empty:
        return pd.DataFrame()
    return df


def get_latest_price(ticker: str) -> Optional[float]:
    """Get latest close price for a ticker"""
    df = fetch_stock_data(ticker, period="5d")
    if df.empty:
        return None
    return float(df["close"].iloc[-1])
