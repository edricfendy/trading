"""
Trading AI Analyzer - Buy/Sell timing, Rebound screening, Fundamentals
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
from fundamentals import FundamentalSnapshot, fetch_fundamentals


@dataclass
class TimingSignal:
    ticker: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0-1 overall technical+fundamental
    reason: str
    stoch_rsi_k: Optional[float]
    stoch_rsi_d: Optional[float]
    smi: Optional[float]
    macd_trend: Optional[str]
    roc_12: Optional[float]
    support_20: Optional[float]
    resistance_20: Optional[float]
    # Fundamentals / valuation
    pbv: Optional[float]
    per: Optional[float]
    roe: Optional[float]
    roa: Optional[float]
    free_float_ratio: Optional[float]
    horizon: str  # "long-term", "short-term", "speculative", "neutral"
    take_profit: Optional[float]
    stop_loss: Optional[float]
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


def _classify_valuation(f: FundamentalSnapshot) -> str:
    """Rough valuation bucket from PBV/PER."""
    if f.pbv is None or f.per is None:
        return "unknown"
    if f.pbv < 1.5 and f.per < 15:
        return "undervalued"
    if f.pbv > 3 or f.per > 25:
        return "expensive"
    return "fair"


def _liquidity_note(f: FundamentalSnapshot) -> str:
    if f.free_float_ratio is None:
        return "free float unknown"
    pct = f.free_float_ratio * 100
    if pct < 15:
        return f"low free float ({pct:.1f}%) – watch liquidity"
    if pct > 40:
        return f"high free float ({pct:.1f}%) – good liquidity"
    return f"moderate free float ({pct:.1f}%)"


def analyze_buy_sell_timing(
    df: pd.DataFrame,
    ticker: str,
) -> TimingSignal:
    """
    Analyze best time to buy/sell based on TA + fundamentals.
    """
    if df is None or df.empty or len(df) < 30:
        now = pd.Timestamp.now().isoformat()
        return TimingSignal(
            ticker=ticker,
            action="HOLD",
            confidence=0.0,
            reason="Insufficient data",
            stoch_rsi_k=None,
            stoch_rsi_d=None,
            smi=None,
            macd_trend=None,
            roc_12=None,
            support_20=None,
            resistance_20=None,
            pbv=None,
            per=None,
            roe=None,
            roa=None,
            free_float_ratio=None,
            horizon="neutral",
            take_profit=None,
            stop_loss=None,
            price=0.0,
            timestamp=now,
        )

    df = add_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    price = float(latest["close"])
    stoch_k = latest.get("stoch_rsi_k")
    stoch_d = latest.get("stoch_rsi_d")
    smi_val = latest.get("smi")
    macd_line = latest.get("macd_line")
    macd_signal = latest.get("macd_signal")
    roc_12 = latest.get("roc_12")
    support_20 = latest.get("support_20")
    resistance_20 = latest.get("resistance_20")

    reasons: list[str] = []
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

    # MACD trend
    macd_trend: Optional[str] = None
    if pd.notna(macd_line) and pd.notna(macd_signal):
        prev_macd = prev.get("macd_line")
        prev_sig = prev.get("macd_signal")
        if pd.notna(prev_macd) and pd.notna(prev_sig):
            if macd_line > macd_signal and prev_macd <= prev_sig:
                macd_trend = "golden_cross"
                reasons.append("MACD golden cross (bullish)")
                buy_score += 0.25
            elif macd_line < macd_signal and prev_macd >= prev_sig:
                macd_trend = "dead_cross"
                reasons.append("MACD dead cross (bearish)")
                sell_score += 0.25

    # Smart Money accumulation
    if pd.notna(smi_val):
        if smi_val > SMI_BULLISH_THRESHOLD:
            reasons.append(f"Smart money accumulating (SMI={smi_val:.2f})")
            buy_score += 0.3
        elif smi_val < -0.5:
            reasons.append(f"Smart money distributing (SMI={smi_val:.2f})")
            sell_score += 0.3

    # ROC momentum
    if pd.notna(roc_12):
        if roc_12 > 5:
            reasons.append(f"Positive momentum ROC12={roc_12:.1f}%")
            buy_score += 0.1
        elif roc_12 < -5:
            reasons.append(f"Weak momentum ROC12={roc_12:.1f}%")
            sell_score += 0.1

    # Determine technical action
    action = "HOLD"
    tech_conf = 0.0
    if buy_score > sell_score and buy_score >= 0.4:
        action = "BUY"
        tech_conf = min(1.0, buy_score)
    elif sell_score > buy_score and sell_score >= 0.4:
        action = "SELL"
        tech_conf = min(1.0, sell_score)
    else:
        action = "HOLD"
        tech_conf = 0.5

    # Fundamentals
    f = fetch_fundamentals(ticker)
    valuation = _classify_valuation(f)
    liq_note = _liquidity_note(f)

    horizon = "neutral"
    if action == "BUY":
        if valuation == "undervalued":
            horizon = "long-term"
            tech_conf += 0.2
            reasons.append("PBV & PER indicate undervaluation – suitable for long-term hold")
        elif valuation == "expensive":
            horizon = "short-term"
            reasons.append("PBV/PER rich – treat as short-term/momentum trade")
        else:
            horizon = "balanced"
    elif action == "SELL" and valuation == "expensive":
        reasons.append("Rich valuation supports taking profit / reducing exposure")

    reasons.append(liq_note)

    # TP / SL based on support / resistance
    tp = None
    sl = None
    if action == "BUY":
        if pd.notna(resistance_20):
            tp = float(resistance_20 * 0.98)
            reasons.append(f"Take-profit near resistance ~Rp {tp:,.0f}")
        if pd.notna(support_20):
            sl = float(support_20 * 0.97)
            reasons.append(f"Stop-loss slightly below support ~Rp {sl:,.0f}")

    confidence = max(0.0, min(1.0, tech_conf))
    reason_str = "; ".join(reasons) if reasons else "No strong signal"

    return TimingSignal(
        ticker=ticker,
        action=action,
        confidence=confidence,
        reason=reason_str,
        stoch_rsi_k=float(stoch_k) if pd.notna(stoch_k) else None,
        stoch_rsi_d=float(stoch_d) if pd.notna(stoch_d) else None,
        smi=float(smi_val) if pd.notna(smi_val) else None,
        macd_trend=macd_trend,
        roc_12=float(roc_12) if pd.notna(roc_12) else None,
        support_20=float(support_20) if pd.notna(support_20) else None,
        resistance_20=float(resistance_20) if pd.notna(resistance_20) else None,
        pbv=f.pbv,
        per=f.per,
        roe=f.roe,
        roa=f.roa,
        free_float_ratio=f.free_float_ratio,
        horizon=horizon,
        take_profit=tp,
        stop_loss=sl,
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
