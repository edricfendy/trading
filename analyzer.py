"""
Trading AI Analyzer - Buy/Sell timing and Rebound screening
"""
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass
from config import (
    OVERSOLD_THRESHOLD,
    OVERBOUGHT_THRESHOLD,
    SMI_BULLISH_THRESHOLD,
)
from data_fetcher import fetch_multiple_stocks, fetch_stock_data
from indicators import add_indicators


@dataclass
class TimingSignal:
    ticker: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1
    reason: str
    stoch_rsi_k: Optional[float]
    stoch_rsi_d: Optional[float]
    smi: Optional[float]
    price: float
    timestamp: str


@dataclass
class ReboundCandidate:
    ticker: str
    rebound_score: float  # 0-100
    stoch_rsi_k: float
    smi_trend: str  # ACCUMULATING / DISTRIBUTING / NEUTRAL
    recent_change_pct: float
    reasons: list[str]
    price: float
    timestamp: str


def analyze_buy_sell_timing(
    df: pd.DataFrame,
    ticker: str,
) -> TimingSignal:
    """
    Analyze best time to buy/sell based on Stochastic RSI and SMI.
    """
    if df is None or df.empty or len(df) < 30:
        return TimingSignal(
            ticker=ticker,
            action="HOLD",
            confidence=0.0,
            reason="Insufficient data",
            stoch_rsi_k=None,
            stoch_rsi_d=None,
            smi=None,
            price=0.0,
            timestamp=pd.Timestamp.now().isoformat(),
        )

    df = add_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    price = float(latest["close"])
    stoch_k = latest.get("stoch_rsi_k")
    stoch_d = latest.get("stoch_rsi_d")
    smi_val = latest.get("smi")

    reasons = []
    buy_score = 0.0
    sell_score = 0.0

    # Stochastic RSI signals
    if pd.notna(stoch_k):
        if stoch_k < OVERSOLD_THRESHOLD:
            reasons.append(f"Stoch RSI oversold ({stoch_k:.1f} < {OVERSOLD_THRESHOLD})")
            buy_score += 0.4
        elif stoch_k > OVERBOUGHT_THRESHOLD:
            reasons.append(f"Stoch RSI overbought ({stoch_k:.1f} > {OVERBOUGHT_THRESHOLD})")
            sell_score += 0.4

        # %K crossing above %D = bullish
        if pd.notna(stoch_d) and stoch_k > stoch_d and prev.get("stoch_rsi_k", 0) <= prev.get("stoch_rsi_d", 0):
            reasons.append("Stoch RSI %K crossed above %D (bullish)")
            buy_score += 0.2
        elif pd.notna(stoch_d) and stoch_k < stoch_d and prev.get("stoch_rsi_k", 100) >= prev.get("stoch_rsi_d", 100):
            reasons.append("Stoch RSI %K crossed below %D (bearish)")
            sell_score += 0.2

    # Smart Money accumulation
    if pd.notna(smi_val):
        if smi_val > SMI_BULLISH_THRESHOLD:
            reasons.append(f"Smart money accumulating (SMI={smi_val:.2f})")
            buy_score += 0.3
        elif smi_val < -0.5:
            reasons.append(f"Smart money distributing (SMI={smi_val:.2f})")
            sell_score += 0.3

    # Determine action
    action = "HOLD"
    confidence = 0.0
    if buy_score > sell_score and buy_score >= 0.3:
        action = "BUY"
        confidence = min(1.0, buy_score)
    elif sell_score > buy_score and sell_score >= 0.3:
        action = "SELL"
        confidence = min(1.0, sell_score)
    else:
        action = "HOLD"
        confidence = 0.5

    reason_str = "; ".join(reasons) if reasons else "No strong signal"

    return TimingSignal(
        ticker=ticker,
        action=action,
        confidence=confidence,
        reason=reason_str,
        stoch_rsi_k=float(stoch_k) if pd.notna(stoch_k) else None,
        stoch_rsi_d=float(stoch_d) if pd.notna(stoch_d) else None,
        smi=float(smi_val) if pd.notna(smi_val) else None,
        price=price,
        timestamp=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
    )


def screen_rebound_candidates(
    tickers: Optional[list[str]] = None,
    min_score: float = 50,
) -> list[ReboundCandidate]:
    """
    Screen stocks with potential rebound based on:
    - Stochastic RSI oversold (recovery potential)
    - Smart Money accumulation (institutional interest)
    - Recent price decline (more room to rebound)
    """
    data = fetch_multiple_stocks(tickers=tickers, period="3mo")
    candidates = []

    for ticker, df in data.items():
        if df.empty or len(df) < 30:
            continue

        df = add_indicators(df)
        latest = df.iloc[-1]

        price = float(latest["close"])
        stoch_k = latest.get("stoch_rsi_k")
        smi = latest.get("smi")
        smi_acc = latest.get("smi_acc")

        if pd.isna(stoch_k):
            continue

        reasons = []
        score = 0.0

        # 1. Oversold = rebound potential (max 40 pts)
        if stoch_k < OVERSOLD_THRESHOLD:
            oversold_pts = 40 * (1 - stoch_k / OVERSOLD_THRESHOLD)
            score += oversold_pts
            reasons.append(f"Stoch RSI oversold ({stoch_k:.1f}) - rebound potential")

        # 2. Smart money accumulating (max 35 pts)
        smi_trend = "NEUTRAL"
        if pd.notna(smi):
            if smi > SMI_BULLISH_THRESHOLD:
                score += 35
                smi_trend = "ACCUMULATING"
                reasons.append("Smart money accumulating")
            elif smi < -0.5:
                smi_trend = "DISTRIBUTING"
                reasons.append("Smart money distributing (lower rebound odds)")
            else:
                score += 15
                reasons.append("SMI neutral")

        # 3. Recent decline - room for rebound (max 25 pts)
        recent_change = 0.0
        if len(df) >= 5:
            recent_change = (price - df["close"].iloc[-5]) / df["close"].iloc[-5] * 100
        if len(df) >= 20:
            recent_high = df["close"].iloc[-20:].max()
            change_from_high = (price - recent_high) / recent_high * 100
            if change_from_high < -5:  # Down 5%+ from recent high
                decline_pts = min(25, abs(change_from_high) * 2)
                score += decline_pts
                reasons.append(f"Down {change_from_high:.1f}% from recent high - room to rebound")

        if score >= min_score:
            candidates.append(
                ReboundCandidate(
                    ticker=ticker,
                    rebound_score=round(min(100, score), 1),
                    stoch_rsi_k=float(stoch_k),
                    smi_trend=smi_trend,
                    recent_change_pct=recent_change,
                    reasons=reasons,
                    price=price,
                    timestamp=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
                )
            )

    candidates.sort(key=lambda x: x.rebound_score, reverse=True)
    return candidates


def get_all_signals(tickers: Optional[list[str]] = None) -> list[TimingSignal]:
    """Get buy/sell timing signals for all tracked stocks"""
    data = fetch_multiple_stocks(tickers=tickers, period="3mo")
    signals = []
    for ticker, df in data.items():
        sig = analyze_buy_sell_timing(df, ticker)
        signals.append(sig)
    return sorted(signals, key=lambda x: (x.action != "HOLD", -x.confidence))
