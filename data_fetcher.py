"""
Indonesia Stock Data Fetcher - Real-time and historical OHLCV data
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from config import IDX_STOCKS, get_data_period


def fetch_stock_data(
    ticker: str,
    period: Optional[str] = "3mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single Indonesia stock.
    ticker should include .JK suffix (e.g., BBCA.JK)
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_multiple_stocks(
    tickers: Optional[list[str]] = None,
    period: str = "3mo",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """
    Fetch data for multiple Indonesia stocks.
    Returns dict of {ticker: DataFrame}
    """
    tickers = tickers or IDX_STOCKS
    results = {}
    for t in tickers:
        df = fetch_stock_data(t, period=period, interval=interval)
        if not df.empty:
            results[t] = df
    return results


def fetch_intraday(ticker: str, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch intraday data for shorter-term analysis.
    Note: yfinance intraday has limitations; 1h is typically available.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period="5d", interval=interval)
    if df.empty:
        return pd.DataFrame()
    df.columns = [c.lower() for c in df.columns]
    return df


def get_latest_price(ticker: str) -> Optional[float]:
    """Get latest close price for a ticker"""
    df = fetch_stock_data(ticker, period="5d")
    if df.empty:
        return None
    return float(df["close"].iloc[-1])
