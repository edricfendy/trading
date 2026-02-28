"""
Indonesia Stock Data Fetcher - Real-time and historical OHLCV data
"""
from __future__ import annotations
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
import yfinance as yf

from config import (
    DATA_PROVIDER,
    GOAPI_API_KEY,
    GOAPI_BASE_URL,
    GOAPI_OHLCV_ENDPOINT,
    GOAPI_QUOTE_ENDPOINT,
    TWELVEDATA_API_KEY,
    TWELVEDATA_BASE_URL,
    get_data_period,
)
from universe import get_universe

DOWNLOAD_BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES_SEC = 1.0
DATA_CACHE_TTL_SEC = int(os.getenv("DATA_CACHE_TTL_SEC", "30"))
TWELVEDATA_MAX_RETRIES = int(os.getenv("TWELVEDATA_MAX_RETRIES", "2"))
TWELVEDATA_RETRY_BACKOFF_SEC = float(os.getenv("TWELVEDATA_RETRY_BACKOFF_SEC", "2.0"))
GOAPI_SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("GOAPI_SLEEP_BETWEEN_CALLS_SEC", "0.2"))
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHAVANTAGE_MAX_RETRIES = int(os.getenv("ALPHAVANTAGE_MAX_RETRIES", "2"))
ALPHAVANTAGE_RETRY_BACKOFF_SEC = float(os.getenv("ALPHAVANTAGE_RETRY_BACKOFF_SEC", "2.0"))
YF_MAX_RETRIES = int(os.getenv("YF_MAX_RETRIES", "3"))
YF_RETRY_BACKOFF_SEC = float(os.getenv("YF_RETRY_BACKOFF_SEC", "1.0"))

_CACHE: dict[tuple, tuple[float, dict[str, pd.DataFrame]]] = {}


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


def _cache_key(tickers: list[str], period: Optional[str], interval: str, all_idx: bool) -> tuple:
    return (tuple(tickers), period or "", interval, bool(all_idx))


def _clone_data_map(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {k: v.copy() for k, v in data.items()}


def _get_cached(key: tuple) -> Optional[dict[str, pd.DataFrame]]:
    if DATA_CACHE_TTL_SEC <= 0:
        return None
    entry = _CACHE.get(key)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts > DATA_CACHE_TTL_SEC:
        _CACHE.pop(key, None)
        return None
    return _clone_data_map(data)


def _set_cached(key: tuple, data: dict[str, pd.DataFrame]) -> None:
    if DATA_CACHE_TTL_SEC <= 0:
        return
    _CACHE[key] = (time.time(), _clone_data_map(data))


def _download_yfinance_batch(
    tickers: list[str],
    period: Optional[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data using yfinance. Best support for .JK (IDX) tickers."""
    period = _resolve_period(period, interval)
    results: dict[str, pd.DataFrame] = {}

    # Map yfinance interval names
    yf_interval_map = {
        "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m",
        "30m": "30m", "60m": "60m", "90m": "90m", "1h": "1h",
        "1d": "1d", "1wk": "1wk", "1mo": "1mo",
    }
    yf_interval = yf_interval_map.get(interval, "1d")

    for ticker in tickers:
        for attempt in range(YF_MAX_RETRIES + 1):
            try:
                t = yf.Ticker(ticker)
                df = t.history(period=period, interval=yf_interval, auto_adjust=True, proxy=None, prepost=True)
                if df is not None and not df.empty:
                    df = df.rename(columns=str.lower)
                    # Ensure standard column names
                    col_map = {"stock splits": "stock_splits"}
                    df = df.rename(columns=col_map)
                    # Keep only OHLCV columns
                    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
                    if len(keep) >= 4:  # at least OHLC
                        df = df[keep]
                        df = _normalize_ohlcv(df)
                        if not df.empty:
                            results[ticker] = df
                            break
                # If we get here with no data, but it's not the last attempt, retry
                if df is None or df.empty:
                    if attempt < YF_MAX_RETRIES:
                        time.sleep(YF_RETRY_BACKOFF_SEC * (2**attempt))
                        continue
                    break
                break
            except Exception as e:
                if attempt < YF_MAX_RETRIES:
                    time.sleep(YF_RETRY_BACKOFF_SEC * (2**attempt))
                    continue
                break

    return results


def _download_batch(
    tickers: list[str],
    period: Optional[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}

    period = _resolve_period(period, interval)

    if DATA_PROVIDER == "yfinance":
        return _download_yfinance_batch(tickers, period, interval)
    if DATA_PROVIDER == "twelvedata":
        return _download_twelvedata_batch(tickers, period, interval)
    if DATA_PROVIDER == "goapi":
        return _download_goapi_batch(tickers, period, interval)
    if DATA_PROVIDER == "alphavantage":
        return _download_alphavantage_batch(tickers, period, interval)

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


def _av_interval(interval: str) -> str:
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "60min",
        "1d": "daily",
        "1wk": "weekly",
        "1mo": "monthly"
    }
    if interval not in mapping:
        raise DataProviderError(f"Unsupported interval for Alpha Vantage: {interval}")
    return mapping[interval]


def _td_symbol(ticker: str) -> str:
    if ":" in ticker:
        return ticker
    if ticker.endswith(".JK"):
        return f"{ticker[:-3]}:IDX"
    return ticker


def _av_symbol(ticker: str) -> str:
    if ticker.endswith(".JK"):
        # For Indonesia stocks, Alpha Vantage uses .JK extension
        return ticker
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
    payload = None
    for attempt in range(TWELVEDATA_MAX_RETRIES + 1):
        resp = requests.get(url, params=params, timeout=20)
        try:
            payload = resp.json()
        except ValueError:
            payload = None

        if payload is None:
            if attempt >= TWELVEDATA_MAX_RETRIES:
                raise DataProviderError("Twelve Data returned non-JSON response.")
            time.sleep(TWELVEDATA_RETRY_BACKOFF_SEC * (2**attempt))
            continue

        try:
            _raise_twelvedata_error(resp, payload if isinstance(payload, dict) else {})
            break
        except DataRateLimitError:
            if attempt >= TWELVEDATA_MAX_RETRIES:
                raise
            retry_after = resp.headers.get("Retry-After")
            try:
                wait_sec = float(retry_after) if retry_after else None
            except ValueError:
                wait_sec = None
            time.sleep(wait_sec or (TWELVEDATA_RETRY_BACKOFF_SEC * (2**attempt)))

    if payload is None:
        return {}

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


def _parse_alphavantage_payload(function: str, payload: dict) -> pd.DataFrame:
    """Parse AlphaVantage API response into a DataFrame."""
    # Different keys depending on function
    if function.startswith("TIME_SERIES_INTRADAY"):
        metadata_key = "Meta Data"
        data_key = [key for key in payload.keys() if key.startswith("Time Series")]
        data_key = data_key[0] if data_key else None
    elif function == "TIME_SERIES_DAILY":
        data_key = "Time Series (Daily)"
    elif function == "TIME_SERIES_WEEKLY":
        data_key = "Weekly Time Series"
    elif function == "TIME_SERIES_MONTHLY":
        data_key = "Monthly Time Series"
    else:
        return pd.DataFrame()  # Unsupported function
    
    if data_key not in payload:
        return pd.DataFrame()

    # Convert time series to DataFrame
    data = payload[data_key]
    df = pd.DataFrame.from_dict(data, orient='index')
    
    # Rename columns to standard OHLCV
    col_map = {
        "1. open": "open", "2. high": "high", "3. low": "low", "4. close": "close",
        "5. volume": "volume", "5. adjusted close": "adjusted_close"
    }
    df.rename(columns=col_map, inplace=True)
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Convert values to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Make sure we have the standard columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            df[col] = 0.0
    
    # Keep only standard OHLCV columns
    keep = ["open", "high", "low", "close", "volume"]
    if all(col in df.columns for col in keep):
        df = df[keep]
        return _normalize_ohlcv(df)
    
    return pd.DataFrame()


def _raise_alphavantage_error(resp: requests.Response, payload: dict) -> None:
    """Check for Alpha Vantage API errors and raise appropriate exception."""
    if resp.status_code == 429:
        raise DataRateLimitError("Alpha Vantage rate limit reached.")
    
    # Alpha Vantage returns error messages in the "Note" or "Error Message" key
    if "Note" in payload and "call frequency" in payload["Note"]:
        raise DataRateLimitError(f"Alpha Vantage: {payload['Note']}")
    
    if "Error Message" in payload:
        raise DataProviderError(f"Alpha Vantage: {payload['Error Message']}")
    
    if resp.status_code >= 400:
        raise DataProviderError(f"Alpha Vantage HTTP {resp.status_code}.")


def _download_alphavantage_batch(
    tickers: list[str],
    period: Optional[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data from Alpha Vantage API."""
    if not ALPHAVANTAGE_API_KEY:
        raise DataProviderError("ALPHAVANTAGE_API_KEY is not set.")

    results: dict[str, pd.DataFrame] = {}
    
    # Alpha Vantage doesn't support batch downloads, so we need to do one ticker at a time
    for ticker in tickers:
        symbol = _av_symbol(ticker)
        av_interval = _av_interval(interval)
        
        # Different functions for different intervals
        if interval in _INTRADAY_INTERVALS:
            function = "TIME_SERIES_INTRADAY"
            params = {
                "function": function,
                "symbol": symbol,
                "interval": av_interval,
                "outputsize": "full" if interval == "1m" else "compact",
                "apikey": ALPHAVANTAGE_API_KEY,
            }
        elif interval == "1d":
            function = "TIME_SERIES_DAILY"
            params = {
                "function": function,
                "symbol": symbol,
                "outputsize": "full",  # Get all available data
                "apikey": ALPHAVANTAGE_API_KEY,
            }
        elif interval == "1wk":
            function = "TIME_SERIES_WEEKLY"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": ALPHAVANTAGE_API_KEY,
            }
        elif interval == "1mo":
            function = "TIME_SERIES_MONTHLY"
            params = {
                "function": function,
                "symbol": symbol,
                "apikey": ALPHAVANTAGE_API_KEY,
            }
        else:
            continue  # Skip unsupported intervals
        
        for attempt in range(ALPHAVANTAGE_MAX_RETRIES + 1):
            try:
                resp = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=20)
                
                try:
                    payload = resp.json()
                except ValueError:
                    if attempt < ALPHAVANTAGE_MAX_RETRIES:
                        time.sleep(ALPHAVANTAGE_RETRY_BACKOFF_SEC * (2**attempt))
                        continue
                    raise DataProviderError("Alpha Vantage returned non-JSON response.")
                
                # Check for API errors
                try:
                    _raise_alphavantage_error(resp, payload)
                except DataRateLimitError:
                    if attempt < ALPHAVANTAGE_MAX_RETRIES:
                        time.sleep(ALPHAVANTAGE_RETRY_BACKOFF_SEC * (2**attempt) + 5)  # Add extra delay for rate limits
                        continue
                    raise
                
                # Parse the response
                df = _parse_alphavantage_payload(function, payload)
                if not df.empty:
                    results[ticker] = df
                
                # Alpha Vantage has strict rate limits, so add a delay between requests
                if ticker != tickers[-1]:
                    time.sleep(1.0)  # Sleep 1 second between requests to avoid rate limits
                
                break  # Success, break retry loop
                
            except (requests.RequestException, ValueError) as e:
                if attempt < ALPHAVANTAGE_MAX_RETRIES:
                    time.sleep(ALPHAVANTAGE_RETRY_BACKOFF_SEC * (2**attempt))
                    continue
                # Skip this ticker on final failure
                break
    
    return results


def _goapi_symbol(ticker: str) -> str:
    t = ticker.strip().upper()
    if t.endswith(".JK"):
        return t[:-3]
    return t


def _build_goapi_url(endpoint: str, symbol: Optional[str] = None) -> str:
    ep = endpoint.strip()
    if symbol:
        ep = ep.replace("{symbol}", symbol).replace(":symbol", symbol)
    if ep.startswith("http://") or ep.startswith("https://"):
        return ep
    if not ep.startswith("/"):
        ep = "/" + ep
    return f"{GOAPI_BASE_URL}{ep}"


def _period_to_calendar_days(period: str) -> int:
    if not period:
        return 30
    p = period.lower()
    try:
        if p.endswith("mo"):
            return int(p[:-2]) * 30
        if p.endswith("wk"):
            return int(p[:-2]) * 7
        if p.endswith("y"):
            return int(p[:-1]) * 365
        if p.endswith("d"):
            return int(p[:-1])
    except ValueError:
        return 30
    return 30


def _goapi_date_range(period: Optional[str], interval: str) -> tuple[str, str]:
    period = _resolve_period(period, interval)
    days = _period_to_calendar_days(period)
    if interval in _INTRADAY_INTERVALS:
        days = min(days, 7)
    end = datetime.utcnow().date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def _extract_goapi_items(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("data", "result", "results", "values"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                for k2 in ("items", "results", "values", "data"):
                    v2 = val.get(k2)
                    if isinstance(v2, list):
                        return v2
        # Some responses may use "results" directly inside "data" as dict of lists
        if "data" in payload and isinstance(payload["data"], dict):
            for v in payload["data"].values():
                if isinstance(v, list):
                    return v
    return []


def _parse_goapi_ohlcv(items: list[dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()

    def _get(item: dict, keys: list[str]):
        for k in keys:
            if k in item:
                return item[k]
        return None

    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lower = {str(k).lower(): v for k, v in item.items()}
        dt_val = _get(lower, ["datetime", "date", "time", "timestamp", "t"])
        if dt_val is None:
            continue

        dt = pd.to_datetime(dt_val, errors="coerce", utc=True)
        if pd.isna(dt):
            try:
                ts = float(dt_val)
                if ts > 1e12:
                    dt = pd.to_datetime(int(ts), unit="ms", utc=True)
                elif ts > 1e10:
                    dt = pd.to_datetime(int(ts), unit="s", utc=True)
            except Exception:
                dt = pd.NaT
        if pd.isna(dt):
            continue

        row = {
            "datetime": dt,
            "open": _get(lower, ["open", "o"]),
            "high": _get(lower, ["high", "h"]),
            "low": _get(lower, ["low", "l"]),
            "close": _get(lower, ["close", "c", "last", "price"]),
            "volume": _get(lower, ["volume", "v"]),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df.sort_values("datetime")
    df = df.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    return _normalize_ohlcv(df)


def _raise_goapi_error(resp: requests.Response, payload: object) -> None:
    if resp.status_code == 429:
        raise DataRateLimitError("GOAPI rate limit reached.")
    if resp.status_code >= 400:
        raise DataProviderError(f"GOAPI HTTP {resp.status_code}.")
    if isinstance(payload, dict):
        code = payload.get("code")
        status = str(payload.get("status", "")).lower()
        msg = payload.get("message") or payload.get("error")
        if str(code) == "429":
            raise DataRateLimitError(str(msg) or "GOAPI rate limit reached.")
        if status in ("error", "failed"):
            raise DataProviderError(str(msg) or "GOAPI error.")


def _goapi_latest_price(ticker: str) -> Optional[float]:
    if not GOAPI_QUOTE_ENDPOINT:
        return None
    url = _build_goapi_url(GOAPI_QUOTE_ENDPOINT)
    symbol = _goapi_symbol(ticker)
    headers = {"X-API-KEY": GOAPI_API_KEY}
    resp = requests.get(url, headers=headers, params={"symbols": symbol}, timeout=20)
    try:
        payload = resp.json()
    except ValueError:
        raise DataProviderError("GOAPI returned non-JSON response.")

    _raise_goapi_error(resp, payload)

    items = _extract_goapi_items(payload)
    if not items:
        return None
    first = items[0] if isinstance(items, list) else None
    if not isinstance(first, dict):
        return None
    lower = {str(k).lower(): v for k, v in first.items()}
    for key in ("price", "last", "close", "c"):
        if key in lower:
            try:
                return float(lower[key])
            except (TypeError, ValueError):
                return None
    return None


def _download_goapi_batch(
    tickers: list[str],
    period: Optional[str],
    interval: str,
) -> dict[str, pd.DataFrame]:
    if not GOAPI_API_KEY:
        raise DataProviderError("GOAPI_API_KEY is not set.")
    if not GOAPI_OHLCV_ENDPOINT:
        raise DataProviderError("GOAPI_OHLCV_ENDPOINT is not set.")

    start_date, end_date = _goapi_date_range(period, interval)
    headers = {"X-API-KEY": GOAPI_API_KEY}
    results: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(tickers):
        symbol = _goapi_symbol(ticker)
        url = _build_goapi_url(GOAPI_OHLCV_ENDPOINT, symbol)
        params = {"from": start_date, "to": end_date}
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        try:
            payload = resp.json()
        except ValueError:
            raise DataProviderError("GOAPI returned non-JSON response.")

        _raise_goapi_error(resp, payload)
        items = _extract_goapi_items(payload)
        df = _parse_goapi_ohlcv(items)
        if not df.empty:
            results[ticker] = df

        if i > 0 and (i + 1) < len(tickers):
            time.sleep(GOAPI_SLEEP_BETWEEN_CALLS_SEC)

    return results


def _fetch_symbols_batch(tickers: list[str], period: str, interval: str) -> dict[str, pd.DataFrame]:
    """Fetch multiple symbols in parallel, using appropriate batch size."""
    all_results: dict[str, pd.DataFrame] = {}
    batches = list(_chunked(tickers, DOWNLOAD_BATCH_SIZE))

    if len(batches) <= 1:
        return _download_batch(tickers, period, interval)

    with ThreadPoolExecutor(max_workers=min(8, len(batches))) as executor:
        futures = []
        for i, batch in enumerate(batches):
            futures.append(
                executor.submit(_download_batch, batch, period, interval)
            )
            if i > 0 and i < len(batches) - 1:
                time.sleep(SLEEP_BETWEEN_BATCHES_SEC)

        for future in as_completed(futures):
            try:
                batch_result = future.result()
                all_results.update(batch_result)
            except Exception:
                # One batch failing shouldn't fail the entire request
                pass

    return all_results


def fetch_ohlcv(
    tickers: list[str],
    period: Optional[str] = None,
    interval: str = "1d",
    all_idx: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) time series for the specified tickers.

    Args:
        tickers: List of ticker symbols e.g. ["BBCA.JK", "BBRI.JK"]
        period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y")
        interval: Data interval ("1m", "5m", "15m", "30m", "60m", "1h", "1d", "1wk", "1mo")
        all_idx: If True and tickers is empty, fetch all IDX stocks

    Returns:
        Dict of DataFrames with OHLCV data, keyed by ticker symbol
    """
    if not tickers and not all_idx:
        return {}

    # If no tickers provided, load all stocks
    effective_tickers = tickers
    if not tickers and all_idx:
        effective_tickers = get_universe(include_all_idx=True)
    if not effective_tickers:
        return {}

    # Check cache first
    cache_key = _cache_key(effective_tickers, period, interval, all_idx)
    cached = _get_cached(cache_key)
    if cached:
        return cached

    results = _fetch_symbols_batch(effective_tickers, period, interval)
    if results:
        _set_cached(cache_key, results)
    return results


# For compatibility with streamlit_app.py
def fetch_stock_data(ticker: str, period: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker."""
    result = fetch_ohlcv([ticker], period, interval, False)
    return result.get(ticker, pd.DataFrame())


def fetch_realtime_prices(tickers: list[str]) -> tuple[dict[str, float], str]:
    """Fetch realtime prices for tickers. Returns (prices_dict, source)."""
    prices = get_latest_quotes(tickers)
    source = DATA_PROVIDER
    return prices, source


def update_last_candle_with_realtime(df: pd.DataFrame, price: float, volume: Optional[float] = None) -> pd.DataFrame:
    """Update the last candle with realtime price."""
    if df is None or df.empty:
        return df
    df = df.copy()
    if "close" in df.columns:
        df.iloc[-1, df.columns.get_loc("close")] = price
    if volume is not None and "volume" in df.columns:
        df.iloc[-1, df.columns.get_loc("volume")] = volume
    return df


# Additional compatibility aliases for analyzer.py
def fetch_multiple_stocks(tickers: list[str], period: Optional[str] = None, interval: str = "1d", progress_callback=None, bypass_cache: bool = False) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers. Alias for fetch_ohlcv."""
    return fetch_ohlcv(tickers, period, interval, False)


def fetch_multiple_stocks_bulk(tickers: list[str], interval: str = "1d", period: Optional[str] = None, progress_callback=None, bypass_cache: bool = False, return_data: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers in bulk mode."""
    return fetch_ohlcv(tickers, period, interval, False)


def get_latest_quotes(tickers: list[str]) -> dict[str, float]:
    """Get the latest quotes for a list of tickers.

    Args:
        tickers: List of ticker symbols e.g. ["BBCA.JK", "BBRI.JK"]

    Returns:
        Dict of latest prices, keyed by ticker symbol
    """
    if not tickers:
        return {}

    if DATA_PROVIDER == "goapi":
        return {ticker: _goapi_latest_price(ticker) or 0.0 for ticker in tickers}

    # For other providers, fall back to getting the most recent close price
    # from OHLCV data
    results = {}
    ohlcvs = fetch_ohlcv(tickers, "1d", "1d", False)
    for ticker, df in ohlcvs.items():
        if df is not None and not df.empty:
            results[ticker] = df["close"].iloc[-1]
        else:
            results[ticker] = 0.0
    return results
