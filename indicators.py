"""
Technical Indicators:
- Stochastic RSI
- Smart Money Index (SMI)
- MACD (golden cross, dead cross, bullish/bearish divergence)
- Rate of Change (ROC)
- Support / Resistance (rolling + pivot)
- Bandar (Big Player) Volume Detection
- Foreign Flow Proxy (OBV-based)
- Average True Range (ATR) for volatility
- Volume-Weighted Average Price (VWAP)
- Bollinger Band squeeze
"""
import pandas as pd
import numpy as np
from typing import Tuple
from config import STOCH_RSI_PERIOD, STOCH_RSI_K, STOCH_RSI_D, SMI_PERIOD


# ─── Core RSI ────────────────────────────────────────────────────────────────
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─── Stochastic RSI ──────────────────────────────────────────────────────────
def stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    k_period: int = 3,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    rsi_val = rsi(close, period=rsi_period)
    rsi_min = rsi_val.rolling(window=k_period).min()
    rsi_max = rsi_val.rolling(window=k_period).max()
    raw = (rsi_val - rsi_min) / (rsi_max - rsi_min).replace(0, np.nan) * 100
    stoch_k = raw.rolling(window=k_period).mean()
    stoch_d = stoch_k.rolling(window=d_period).mean()
    return stoch_k, stoch_d


# ─── MACD ─────────────────────────────────────────────────────────────────────
def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def detect_macd_divergence(
    close: pd.Series,
    macd_line: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """
    Detect MACD bullish/bearish divergence.
    Returns: 1 = bullish divergence, -1 = bearish divergence, 0 = none
    Uses last `lookback` candles to compare swing highs/lows.
    """
    result = pd.Series(0, index=close.index)
    if len(close) < lookback + 2:
        return result

    for i in range(lookback, len(close)):
        window_close = close.iloc[i - lookback : i + 1]
        window_macd = macd_line.iloc[i - lookback : i + 1]

        # Price makes lower low, MACD makes higher low → bullish divergence
        if window_close.iloc[-1] == window_close.min() and window_macd.iloc[-1] > window_macd.min():
            result.iloc[i] = 1
        # Price makes higher high, MACD makes lower high → bearish divergence
        elif window_close.iloc[-1] == window_close.max() and window_macd.iloc[-1] < window_macd.max():
            result.iloc[i] = -1

    return result


# ─── Smart Money ─────────────────────────────────────────────────────────────
def smart_money_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Normalized SMI based on (Close-Open)/Range * Volume"""
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    raw = ((df["close"] - df["open"]) / hl) * df["volume"]
    smi = raw.rolling(period).sum()
    mean = smi.rolling(period * 2).mean()
    std = smi.rolling(period * 2).std().replace(0, np.nan)
    return (smi - mean) / (std + 1e-8)


def smart_money_accumulation(df: pd.DataFrame, period: int = 14) -> pd.Series:
    diff = df["close"] - df["open"]
    return (diff * df["volume"]).rolling(period).sum()


# ─── Rate of Change ──────────────────────────────────────────────────────────
def rate_of_change(close: pd.Series, period: int = 12) -> pd.Series:
    shifted = close.shift(period)
    return (close - shifted) / shifted.replace(0, np.nan) * 100


# ─── Support / Resistance ────────────────────────────────────────────────────
def support_resistance(
    close: pd.Series, window: int = 20
) -> Tuple[pd.Series, pd.Series]:
    return close.rolling(window).min(), close.rolling(window).max()


def pivot_levels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Classic pivot point: PP, S1, R1"""
    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    prev_close = df["close"].shift(1)
    pp = (prev_high + prev_low + prev_close) / 3
    s1 = 2 * pp - prev_high
    r1 = 2 * pp - prev_low
    return pp, s1, r1


# ─── ATR ─────────────────────────────────────────────────────────────────────
def average_true_range(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - c).abs(), (df["low"] - c).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ─── OBV (Foreign Flow Proxy) ────────────────────────────────────────────────
def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    """OBV as proxy for institutional / foreign flow direction."""
    direction = np.sign(df["close"].diff()).fillna(0)
    obv = (direction * df["volume"]).cumsum()
    return obv


def obv_trend(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Normalized OBV momentum: positive = accumulation, negative = distribution."""
    obv = on_balance_volume(df)
    obv_ma = obv.rolling(period).mean()
    return obv - obv_ma  # above MA = inflow


# ─── Bandar / Big Player Volume Detection ────────────────────────────────────
def bandar_volume_score(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Bandar (Market Maker / Big Player) volume score.
    Detects unusual volume spikes (> 2σ above mean) with price direction confirmation.
    Score: +1 = big buying, -1 = big selling, 0 = normal.
    """
    vol_mean = df["volume"].rolling(period).mean()
    vol_std = df["volume"].rolling(period).std().replace(0, np.nan)
    z_score = (df["volume"] - vol_mean) / vol_std
    is_spike = z_score > 2.0
    price_up = df["close"] > df["open"]
    bandar = pd.Series(0.0, index=df.index)
    bandar[is_spike & price_up] = z_score[is_spike & price_up]    # bullish big vol
    bandar[is_spike & ~price_up] = -z_score[is_spike & ~price_up] # bearish big vol
    return bandar.rolling(5).sum()  # aggregate last 5 days


# ─── VWAP ────────────────────────────────────────────────────────────────────
def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


# ─── Bollinger Bands ─────────────────────────────────────────────────────────
def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + std_dev * std
    lower = ma - std_dev * std
    return upper, ma, lower


# ─── Composite ───────────────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    stoch_k, stoch_d = stochastic_rsi(
        df["close"],
        rsi_period=STOCH_RSI_PERIOD,
        k_period=STOCH_RSI_K,
        d_period=STOCH_RSI_D,
    )
    macd_line, macd_signal, macd_hist = macd(df["close"])
    macd_div = detect_macd_divergence(df["close"], macd_line)
    roc_12 = rate_of_change(df["close"], 12)
    roc_24 = rate_of_change(df["close"], 24)
    supp, resist = support_resistance(df["close"], 20)
    pp, s1, r1 = pivot_levels(df)
    atr = average_true_range(df)
    obv_mom = obv_trend(df)
    bandar = bandar_volume_score(df)
    vwap_val = vwap(df)
    bb_upper, bb_mid, bb_lower = bollinger_bands(df["close"])
    rsi_val = rsi(df["close"])

    df = df.assign(
        rsi=rsi_val,
        stoch_rsi_k=stoch_k,
        stoch_rsi_d=stoch_d,
        smi=smart_money_index(df, period=SMI_PERIOD),
        smi_acc=smart_money_accumulation(df, period=SMI_PERIOD),
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        macd_divergence=macd_div,
        roc_12=roc_12,
        roc_24=roc_24,
        support_20=supp,
        resistance_20=resist,
        pivot=pp,
        pivot_s1=s1,
        pivot_r1=r1,
        atr=atr,
        obv_momentum=obv_mom,
        bandar_score=bandar,
        vwap=vwap_val,
        bb_upper=bb_upper,
        bb_mid=bb_mid,
        bb_lower=bb_lower,
    )
    return df
