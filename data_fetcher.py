"""
Indonesia Stock Data Fetcher - Real-time and historical OHLCV data
"""
import time
from typing import Optional

import pandas as pd
import requests

from config import (
    DATA_PROVIDER,
    GOAPI_API_KEY,
    GOAPI_OHLCV_ENDPOINT,
    GOAPI_QUOTE_ENDPOINT,
    TWELVEDATA_API_KEY,
    TWELVEDATA_BASE_URL,
    get_data_period,
)
from universe import get_universe

DOWNLOAD_BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES_SEC = 1.0


class DataProviderError(RuntimeError):
    """Base error for data provider failures."""


class DataRateLimitError(DataProviderError):
    """Raised when a provider rate limit is hit."""


_INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}


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

    if DATA_PROVIDER == "twelvedata":
        return _download_twelvedata_batch(tickers, period, interval)
    if DATA_PROVIDER == "goapi":
        return _download_goapi_batch(tickers, period, interval)

    raise DataProviderError(f"Unsupported DATA_PROVIDER: {DATA_PROVIDER}")


def _td_interval(interval: str) -> str:
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "45m": "45min",
        "60m": "1h",
        "90m": "1h",
        "1h": "1h",
        "1d": "1day",
        "1wk": "1week",
        "1mo": "1month",
    }
    if interval not in mapping:
        raise DataProviderError(f"Unsupported interval for Twelve Data: {interval}")
    return mapping[interval]


def _td_symbol(ticker: str) -> str:
    if ":" in ticker:
        return ticker
    if ticker.endswith(".JK"):
        return f"{ticker[:-3]}:IDX"
    return ticker


def _period_to_trading_days(period: str) -> Optional[int]:
    if not period:
        return None
    p = period.lower()
    try:
        if p.endswith("mo"):
            return int(p[:-2]) * 22
        if p.endswith("wk"):
            return int(p[:-2]) * 5
        if p.endswith("y"):
            return int(p[:-1]) * 252
        if p.endswith("d"):
            return int(p[:-1])
    except ValueError:
        return None
    return None


def _resolve_outputsize(period: Optional[str], interval: str) -> int:
    period = _resolve_period(period, interval)
    days = _period_to_trading_days(period) or 200
    size = days
    if interval in _INTRADAY_INTERVALS:
        size = max(days * 8, 200)
    return int(min(max(size, 50), 5000))


def _parse_twelvedata_payload(payload: dict) -> pd.DataFrame:
    values = payload.get("values") or []
    if not values:
        return pd.DataFrame()

    df = pd.DataFrame(values)
    if "datetime" not in df.columns:
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")
    df = df.rename(columns=str.lower)
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0

    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    return _normalize_ohlcv(df)


def _raise_twelvedata_error(resp: requests.Response, payload: dict) -> None:
    if resp.status_code == 429 or payload.get("code") == 429:
        raise DataRateLimitError(payload.get("message") or "Twelve Data rate limit reached.")
    if payload.get("status") == "error":
        raise DataProviderError(payload.get("message") or "Twelve Data error.")
    if resp.status_code >= 400:
        raise DataProviderError(f"Twelve Data HTTP {resp.status_code}.")


def _download_twelvedata_batch(
    tickers: list[str],
    period: Optional[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    if not TWELVEDATA_API_KEY:
        raise DataProviderError("TWELVEDATA_API_KEY is not set.")

    td_interval = _td_interval(interval)
    outputsize = _resolve_outputsize(period, interval)
    symbol_map = {t: _td_symbol(t) for t in tickers}
    symbols = ",".join(symbol_map.values())

    url = f"{TWELVEDATA_BASE_URL}/time_series"
    params = {
        "symbol": symbols,
        "interval": td_interval,
        "apikey": TWELVEDATA_API_KEY,
        "outputsize": outputsize,
        "format": "JSON",
    }
    resp = requests.get(url, params=params, timeout=20)
    try:
        payload = resp.json()
    except ValueError:
        raise DataProviderError("Twelve Data returned non-JSON response.")

    _raise_twelvedata_error(resp, payload if isinstance(payload, dict) else {})

    results: dict[str, pd.DataFrame] = {}
    if isinstance(payload, dict) and "values" in payload:
        ticker = tickers[0]
        results[ticker] = _parse_twelvedata_payload(payload)
        return {k: v for k, v in results.items() if not v.empty}

    if isinstance(payload, dict):
        reverse_map = {v: k for k, v in symbol_map.items()}
        for sym, data in payload.items():
            if not isinstance(data, dict):
                continue
            if data.get("status") == "error":
                continue
            ticker = reverse_map.get(sym, sym)
            df = _parse_twelvedata_payload(data)
            if not df.empty:
                results[ticker] = df

    return results


def _download_goapi_batch(
    tickers: list[str],
    period: Optional[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    if not GOAPI_API_KEY:
        raise DataProviderError("GOAPI_API_KEY is not set.")
    if not GOAPI_OHLCV_ENDPOINT:
        raise DataProviderError("GOAPI_OHLCV_ENDPOINT is not set.")
    if not GOAPI_QUOTE_ENDPOINT:
        raise DataProviderError("GOAPI_QUOTE_ENDPOINT is not set.")

    raise DataProviderError(
        "GOAPI integration requires endpoint details. "
        "Provide GOAPI_OHLCV_ENDPOINT/GOAPI_QUOTE_ENDPOINT with the expected params."
    )


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
    except DataRateLimitError:
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
        except DataRateLimitError:
            # bubble up to UI for a friendly message
            raise
        results.update(batch_data)
        if i < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

    return results


def fetch_intraday(ticker: str, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch intraday data for shorter-term analysis.
    Note: intraday availability depends on the data provider and plan.
    """
    try:
        batch = _download_batch([ticker], period="5d", interval=interval)
    except DataRateLimitError:
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
