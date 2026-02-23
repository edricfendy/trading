"""
Technical Indicators: Stochastic RSI, Smart Money, MACD, ROC, Support/Resistance
"""
import pandas as pd
import numpy as np
from typing import Tuple
from config import STOCH_RSI_PERIOD, STOCH_RSI_K, STOCH_RSI_D, SMI_PERIOD


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic RSI.
    Returns (stoch_rsi_k, stoch_rsi_d)
    """
    rsi_val = rsi(close, period=rsi_period)
    rsi_min = rsi_val.rolling(window=k_period).min()
    rsi_max = rsi_val.rolling(window=k_period).max()
    stoch_rsi_raw = (rsi_val - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    stoch_k = stoch_rsi_raw.rolling(window=k_period).mean()
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD indicator.
    Returns (macd_line, signal_line, histogram)
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    """Percentage Rate of Change"""
    shifted = close.shift(period)
    roc = (close - shifted) / shifted.replace(0, np.nan) * 100
    return roc


def smart_money_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Smart Money Index (SMI) / Accumulation indicator.
    Measures institutional/smart money flow:
    SMI = rolling sum of [(Close - Open) / (High - Low)] * Volume
    Normalized (z-score style) for cross-stock comparison.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    open_ = df["open"]
    volume = df["volume"]
    hl_range = (high - low).replace(0, np.nan)
    smi_raw = ((close - open_) / hl_range) * volume
    smi = smi_raw.rolling(window=period).sum()
    mean = smi.rolling(period * 2).mean()
    std = smi.rolling(period * 2).std().replace(0, np.nan)
    smi_norm = (smi - mean) / (std + 1e-8)
    return smi_norm


def smart_money_accumulation(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Alternative Smart Money Accumulation:
    Cumulative (Close - Open) * Volume over rolling window.
    Positive = more buying pressure (closing higher than opening).
    """
    diff = df["close"] - df["open"]
    acc = (diff * df["volume"]).rolling(period).sum()
    return acc


def support_resistance(
    close: pd.Series,
    window: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple support/resistance from rolling lows/highs.
    """
    support = close.rolling(window=window).min()
    resistance = close.rolling(window=window).max()
    return support, resistance


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add TA indicators to OHLCV DataFrame"""
    df = df.copy()
    stoch_k, stoch_d = stochastic_rsi(
        df["close"],
        rsi_period=STOCH_RSI_PERIOD,
        k_period=STOCH_RSI_K,
        d_period=STOCH_RSI_D,
    )
    macd_line, macd_signal, macd_hist = macd(df["close"])
    roc_12 = rate_of_change(df["close"], period=12)
    roc_24 = rate_of_change(df["close"], period=24)
    supp, resist = support_resistance(df["close"], window=20)

    df = df.assign(
        stoch_rsi_k=stoch_k,
        stoch_rsi_d=stoch_d,
        smi=smart_money_index(df, period=SMI_PERIOD),
        smi_acc=smart_money_accumulation(df, period=SMI_PERIOD),
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        roc_12=roc_12,
        roc_24=roc_24,
        support_20=supp,
        resistance_20=resist,
    )
    return df
