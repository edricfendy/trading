"""
Indonesia Stock Data Fetcher - Real-time and historical OHLCV data
"""
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
        try:
            t = yf.Ticker(ticker)
            df = t.history(period=period, interval=yf_interval, auto_adjust=True)
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
        except Exception:
            continue

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

        if i < len(tickers) - 1 and GOAPI_SLEEP_BETWEEN_CALLS_SEC > 0:
            time.sleep(GOAPI_SLEEP_BETWEEN_CALLS_SEC)

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

    cache_key = _cache_key(tickers, period, interval, all_idx)
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

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

    _set_cached(cache_key, results)
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
    if DATA_PROVIDER == "goapi":
        try:
            price = _goapi_latest_price(ticker)
            if price is not None:
                return price
        except DataProviderError:
            pass
    df = fetch_stock_data(ticker, period="5d")
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


# ─── Real-time price fetching ────────────────────────────────────────────────

_RT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _fetch_rt_google(ticker: str) -> Optional[float]:
    """Fetch real-time price from Google Finance for an IDX stock."""
    code = ticker.replace(".JK", "").upper()
    url = f"https://www.google.com/finance/quote/{code}:IDX"
    try:
        resp = requests.get(url, timeout=10, headers=_RT_HEADERS)
        if not resp.ok:
            return None
        # Google Finance stores the price in a data-last-price attribute
        match = re.search(r'data-last-price="([\d.]+)"', resp.text)
        if match:
            price = float(match.group(1))
            return price if price > 0 else None
    except Exception:
        return None
    return None


def _fetch_rt_yahoo_chart(ticker: str) -> Optional[float]:
    """Fetch price from Yahoo Finance chart API (may be ~15-min delayed)."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        resp = requests.get(url, timeout=10, headers=_RT_HEADERS)
        if not resp.ok:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result")
        if result and len(result) > 0:
            meta = result[0].get("meta", {})
            price = meta.get("regularMarketPrice")
            if price and float(price) > 0:
                return float(price)
    except Exception:
        return None
    return None


def fetch_realtime_prices(tickers: list[str]) -> tuple[dict[str, float], str]:
    """
    Fetch real-time prices for IDX stocks using multiple sources.
    Priority: Google Finance (real-time) → Yahoo chart API (fallback).

    Returns:
        (prices_dict, source_label)
        prices_dict: {ticker: price} for tickers with successful fetch
        source_label: description of which source provided the data
    """
    prices: dict[str, float] = {}
    google_count = 0
    yahoo_count = 0

    def _fetch_one(ticker: str) -> tuple[str, Optional[float], str]:
        # Try Google Finance first (true real-time during market hours)
        price = _fetch_rt_google(ticker)
        if price is not None:
            return ticker, price, "google"
        # Fallback to Yahoo chart API
        price = _fetch_rt_yahoo_chart(ticker)
        if price is not None:
            return ticker, price, "yahoo"
        return ticker, None, "none"

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                ticker, price, source = future.result()
                if price is not None:
                    prices[ticker] = price
                    if source == "google":
                        google_count += 1
                    elif source == "yahoo":
                        yahoo_count += 1
            except Exception:
                continue

    # Build source label
    parts = []
    if google_count > 0:
        parts.append(f"Google Finance: {google_count}")
    if yahoo_count > 0:
        parts.append(f"Yahoo Finance: {yahoo_count}")
    source_label = " | ".join(parts) if parts else "No real-time source"

    return prices, source_label


def update_last_candle_with_realtime(
    df: pd.DataFrame, rt_price: float
) -> pd.DataFrame:
    """
    Patch the last candle of an OHLCV DataFrame with a real-time price.

    Updates close to the live price and adjusts high/low if the live price
    exceeds the historical range. This ensures that all indicators computed
    on this DataFrame reflect the most recent market price.
    """
    if df is None or df.empty or rt_price is None or rt_price <= 0:
        return df

    df = df.copy()
    idx = df.index[-1]
    df.loc[idx, "close"] = rt_price
    # Extend high/low if the real-time price broke intraday range
    if rt_price > df.loc[idx, "high"]:
        df.loc[idx, "high"] = rt_price
    if rt_price < df.loc[idx, "low"]:
        df.loc[idx, "low"] = rt_price
    return df
