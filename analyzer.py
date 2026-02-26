"""
Trading AI Analyzer - Enhanced with comprehensive TA + Fundamentals
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from config import OVERSOLD_THRESHOLD, OVERBOUGHT_THRESHOLD, SMI_BULLISH_THRESHOLD
from data_fetcher import fetch_multiple_stocks, fetch_multiple_stocks_bulk, fetch_stock_data, fetch_realtime_prices, update_last_candle_with_realtime
from indicators import add_indicators
from fundamentals import FundamentalSnapshot, fetch_fundamentals


@dataclass
class TimingSignal:
    ticker: str
    action: str                      # BUY, SELL, HOLD
    confidence: float                # 0-1
    horizon: str                     # long-term / short-term / speculative / neutral
    reason: str
    # Technical
    stoch_rsi_k: Optional[float]
    stoch_rsi_d: Optional[float]
    rsi: Optional[float]
    smi: Optional[float]
    macd_trend: Optional[str]
    macd_divergence: Optional[str]   # bullish / bearish / none
    roc_12: Optional[float]
    bandar_score: Optional[float]
    obv_momentum: Optional[str]      # INFLOW / OUTFLOW / NEUTRAL
    support_20: Optional[float]
    resistance_20: Optional[float]
    pivot: Optional[float]
    pivot_s1: Optional[float]
    pivot_r1: Optional[float]
    atr: Optional[float]
    vwap: Optional[float]
    bb_position: Optional[str]       # ABOVE / BELOW / INSIDE
    # Fundamentals
    pbv: Optional[float]
    per: Optional[float]
    per_forward: Optional[float]
    roe: Optional[float]
    roa: Optional[float]
    roic: Optional[float]
    eps: Optional[float]
    eps_growth: Optional[float]
    free_cashflow: Optional[float]
    operating_cashflow: Optional[float]
    revenue: Optional[float]
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    gross_margins: Optional[float]
    profit_margins: Optional[float]
    dividend_yield: Optional[float]
    free_float_ratio: Optional[float]
    market_cap: Optional[float]
    valuation_label: str
    sector: Optional[str]
    industry: Optional[str]
    # Valuation pricing
    book_value_per_share: Optional[float]
    valuation_price: Optional[float]       # PER × EPS (TTM)
    price_vs_valuation: Optional[str]      # CHEAP / EXPENSIVE / FAIR
    # Identity / Ownership
    company_name: Optional[str]
    major_holders: Optional[str]
    # Levels
    take_profit: Optional[float]
    stop_loss: Optional[float]
    price: float
    timestamp: str


@dataclass
class ReboundCandidate:
    ticker: str
    rebound_score: float
    stoch_rsi_k: float
    smi_trend: str
    bandar_trend: str
    recent_change_pct: float
    reasons: list
    price: float
    timestamp: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _valuation_label(f: FundamentalSnapshot) -> str:
    """Classify valuation based on PBV + PER + ROE."""
    if f.pbv is None or f.per is None:
        return "unknown"
    # Adjust for ROE quality
    roe_ok = f.roe is not None and f.roe > 0.10
    if f.pbv < 1.5 and f.per < 15:
        return "undervalued"
    if f.pbv < 2.5 and f.per < 20 and roe_ok:
        return "fair-quality"
    if f.pbv > 4 or f.per > 30:
        return "expensive"
    return "fair"


def _free_float_signal(f: FundamentalSnapshot) -> tuple[str, float]:
    """Returns (note, confidence_adj)"""
    if f.free_float_ratio is None:
        return "free float unknown – treat with caution", 0.0
    pct = f.free_float_ratio * 100
    if pct < 15:
        return f"⚠ LOW free float ({pct:.1f}%) – illiquid, high manipulation risk", -0.10
    if pct > 40:
        return f"✓ HIGH free float ({pct:.1f}%) – good market liquidity", +0.10
    return f"Moderate free float ({pct:.1f}%)", 0.0


def _solvency_notes(f: FundamentalSnapshot) -> list[str]:
    notes = []
    if f.debt_to_equity is not None:
        if f.debt_to_equity > 2:
            notes.append(f"High leverage D/E={f.debt_to_equity:.1f}x – solvency risk")
        elif f.debt_to_equity < 0.5:
            notes.append(f"Low debt D/E={f.debt_to_equity:.1f}x – strong balance sheet")
    if f.current_ratio is not None:
        if f.current_ratio < 1:
            notes.append(f"Current ratio {f.current_ratio:.1f}x < 1 – liquidity concern")
        elif f.current_ratio > 2:
            notes.append(f"Current ratio {f.current_ratio:.1f}x – healthy short-term position")
    return notes


def _fundamental_conf_adj(f: FundamentalSnapshot, action: str) -> tuple[float, list[str]]:
    """Return confidence adjustment and reasons from fundamentals."""
    adj = 0.0
    reasons = []
    label = _valuation_label(f)

    if action == "BUY":
        if label == "undervalued":
            adj += 0.25
            reasons.append(f"Undervalued PBV={f.pbv:.2f}x PER={f.per:.1f}x – strong long-term case")
        elif label == "fair-quality":
            adj += 0.10
            reasons.append(f"Fair valuation PBV={f.pbv:.2f}x PER={f.per:.1f}x with good ROE={f.roe*100:.1f}%")
        elif label == "expensive":
            adj -= 0.10
            reasons.append(f"Rich valuation PBV={f.pbv:.2f}x PER={f.per:.1f}x – momentum/short-term only")
    elif action == "SELL":
        if label in ("expensive",):
            adj += 0.10
            reasons.append("Overvaluation supports reducing exposure")

    if f.roe is not None and f.roe > 0.15:
        adj += 0.05
        reasons.append(f"ROE={f.roe*100:.1f}% – high capital efficiency")
    if f.roic is not None and f.roic > 0.12:
        adj += 0.05
        reasons.append(f"ROIC={f.roic*100:.1f}% – good returns on invested capital")
    if f.free_cashflow is not None and f.free_cashflow > 0:
        adj += 0.05
        reasons.append("Positive free cash flow")
    elif f.free_cashflow is not None and f.free_cashflow < 0:
        adj -= 0.05
        reasons.append("Negative free cash flow – monitor burn rate")

    if f.eps_growth is not None and f.eps_growth > 0.1:
        adj += 0.05
        reasons.append(f"EPS growth {f.eps_growth*100:.1f}% YoY")

    return adj, reasons


def _determine_horizon(action: str, label: str, macd_trend: Optional[str], roc: Optional[float]) -> str:
    if action != "BUY":
        return "neutral"
    if label == "undervalued":
        return "long-term"
    if label in ("fair-quality",):
        return "balanced"
    if label == "expensive":
        return "short-term"
    if roc is not None and roc > 10:
        return "speculative"
    return "neutral"


# ─── Main analyzer ───────────────────────────────────────────────────────────

def analyze_buy_sell_timing(df: pd.DataFrame, ticker: str) -> TimingSignal:
    now = pd.Timestamp.now().isoformat()

    def _empty(action="HOLD"):
        f = FundamentalSnapshot()
        return TimingSignal(
            ticker=ticker, action=action, confidence=0.0,
            horizon="neutral", reason="Insufficient data",
            stoch_rsi_k=None, stoch_rsi_d=None, rsi=None, smi=None,
            macd_trend=None, macd_divergence=None, roc_12=None,
            bandar_score=None, obv_momentum=None,
            support_20=None, resistance_20=None,
            pivot=None, pivot_s1=None, pivot_r1=None,
            atr=None, vwap=None, bb_position=None,
            pbv=None, per=None, per_forward=None,
            roe=None, roa=None, roic=None, eps=None,
            eps_growth=None, free_cashflow=None, operating_cashflow=None,
            revenue=None, debt_to_equity=None, current_ratio=None,
            gross_margins=None, profit_margins=None, dividend_yield=None,
            free_float_ratio=None, market_cap=None, valuation_label="unknown",
            sector=None, industry=None,
            book_value_per_share=None, valuation_price=None,
            price_vs_valuation=None, company_name=None, major_holders=None,
            take_profit=None, stop_loss=None, price=0.0, timestamp=now,
        )

    if df is None or df.empty or len(df) < 30:
        return _empty()

    df = add_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    price = float(latest["close"])
    stoch_k = latest.get("stoch_rsi_k")
    stoch_d = latest.get("stoch_rsi_d")
    rsi_val = latest.get("rsi")
    smi_val = latest.get("smi")
    macd_line = latest.get("macd_line")
    macd_signal_val = latest.get("macd_signal")
    macd_div_val = latest.get("macd_divergence")
    roc_12 = latest.get("roc_12")
    bandar = latest.get("bandar_score")
    obv_mom = latest.get("obv_momentum")
    support_20 = latest.get("support_20")
    resistance_20 = latest.get("resistance_20")
    pivot = latest.get("pivot")
    s1 = latest.get("pivot_s1")
    r1 = latest.get("pivot_r1")
    atr_val = latest.get("atr")
    vwap_val = latest.get("vwap")
    bb_upper = latest.get("bb_upper")
    bb_lower = latest.get("bb_lower")

    reasons: list[str] = []
    buy_score = 0.0
    sell_score = 0.0

    # ── Stochastic RSI ──────────────────────────────────────────────────────
    if pd.notna(stoch_k):
        if stoch_k < OVERSOLD_THRESHOLD:
            reasons.append(f"Stoch RSI oversold ({stoch_k:.1f} < {OVERSOLD_THRESHOLD})")
            buy_score += 0.4
        elif stoch_k > OVERBOUGHT_THRESHOLD:
            reasons.append(f"Stoch RSI overbought ({stoch_k:.1f} > {OVERBOUGHT_THRESHOLD})")
            sell_score += 0.4

        if pd.notna(stoch_d):
            pk = prev.get("stoch_rsi_k", np.nan)
            pd_ = prev.get("stoch_rsi_d", np.nan)
            if pd.notna(pk) and pd.notna(pd_):
                if stoch_k > stoch_d and pk <= pd_:
                    reasons.append("Stoch RSI %K crossed above %D (bullish crossover)")
                    buy_score += 0.2
                elif stoch_k < stoch_d and pk >= pd_:
                    reasons.append("Stoch RSI %K crossed below %D (bearish crossover)")
                    sell_score += 0.2

    # ── RSI ─────────────────────────────────────────────────────────────────
    if pd.notna(rsi_val):
        if rsi_val < 30:
            reasons.append(f"RSI oversold ({rsi_val:.1f})")
            buy_score += 0.15
        elif rsi_val > 70:
            reasons.append(f"RSI overbought ({rsi_val:.1f})")
            sell_score += 0.15

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_trend: Optional[str] = None
    macd_div_str: Optional[str] = None
    if pd.notna(macd_line) and pd.notna(macd_signal_val):
        pm = prev.get("macd_line", np.nan)
        ps = prev.get("macd_signal", np.nan)
        if pd.notna(pm) and pd.notna(ps):
            if macd_line > macd_signal_val and pm <= ps:
                macd_trend = "golden_cross"
                reasons.append("MACD golden cross – bullish momentum")
                buy_score += 0.3
            elif macd_line < macd_signal_val and pm >= ps:
                macd_trend = "dead_cross"
                reasons.append("MACD dead cross – bearish momentum")
                sell_score += 0.3

    if pd.notna(macd_div_val):
        if macd_div_val == 1:
            macd_div_str = "bullish"
            reasons.append("MACD bullish divergence – potential reversal upward")
            buy_score += 0.2
        elif macd_div_val == -1:
            macd_div_str = "bearish"
            reasons.append("MACD bearish divergence – potential reversal downward")
            sell_score += 0.2

    # ── Smart Money ──────────────────────────────────────────────────────────
    if pd.notna(smi_val):
        if smi_val > SMI_BULLISH_THRESHOLD:
            reasons.append(f"Smart money accumulating (SMI={smi_val:.2f})")
            buy_score += 0.3
        elif smi_val < -0.5:
            reasons.append(f"Smart money distributing (SMI={smi_val:.2f})")
            sell_score += 0.3

    # ── Bandar Volume ────────────────────────────────────────────────────────
    if pd.notna(bandar):
        if bandar > 3:
            reasons.append(f"Bandar/big player buying detected (score={bandar:.1f})")
            buy_score += 0.25
        elif bandar < -3:
            reasons.append(f"Bandar/big player selling detected (score={bandar:.1f})")
            sell_score += 0.25
        elif bandar > 1:
            reasons.append(f"Moderate big-player buying (score={bandar:.1f})")
            buy_score += 0.10

    # ── OBV / Foreign Flow ───────────────────────────────────────────────────
    obv_label: Optional[str] = None
    if pd.notna(obv_mom):
        if obv_mom > 0:
            obv_label = "INFLOW"
            reasons.append("OBV momentum positive – institutional/foreign inflow signal")
            buy_score += 0.15
        elif obv_mom < 0:
            obv_label = "OUTFLOW"
            reasons.append("OBV momentum negative – foreign/institutional outflow signal")
            sell_score += 0.15

    # ── ROC Momentum ─────────────────────────────────────────────────────────
    if pd.notna(roc_12):
        if roc_12 > 5:
            reasons.append(f"Positive momentum ROC12={roc_12:.1f}%")
            buy_score += 0.10
        elif roc_12 < -5:
            reasons.append(f"Negative momentum ROC12={roc_12:.1f}%")
            sell_score += 0.10

    # ── Price vs VWAP ────────────────────────────────────────────────────────
    bb_pos: Optional[str] = None
    if pd.notna(bb_upper) and pd.notna(bb_lower):
        if price < float(bb_lower):
            bb_pos = "BELOW"
            reasons.append("Price below Bollinger lower band – potential oversold bounce")
            buy_score += 0.10
        elif price > float(bb_upper):
            bb_pos = "ABOVE"
            reasons.append("Price above Bollinger upper band – overbought zone")
            sell_score += 0.10
        else:
            bb_pos = "INSIDE"

    # ── Action decision ──────────────────────────────────────────────────────
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

    # ── Fundamentals ─────────────────────────────────────────────────────────
    f = fetch_fundamentals(ticker)
    label = _valuation_label(f)
    ff_note, ff_adj = _free_float_signal(f)
    fund_adj, fund_reasons = _fundamental_conf_adj(f, action)
    solv_notes = _solvency_notes(f)

    reasons.append(ff_note)
    reasons.extend(fund_reasons)
    reasons.extend(solv_notes)

    confidence = max(0.0, min(1.0, tech_conf + fund_adj + ff_adj))
    horizon = _determine_horizon(action, label, macd_trend, roc_12 if pd.notna(roc_12) else None)

    # ── Take Profit / Stop Loss (ATR-based + support/resistance) ─────────────
    tp: Optional[float] = None
    sl: Optional[float] = None
    if action == "BUY" and pd.notna(atr_val):
        atr_f = float(atr_val)
        # TP: near resistance or +2.5 ATR
        if pd.notna(resistance_20):
            tp = float(min(float(resistance_20) * 0.98, price + 2.5 * atr_f))
        else:
            tp = price + 2.5 * atr_f
        # SL: below support or -1.5 ATR
        if pd.notna(support_20):
            sl = float(max(float(support_20) * 0.97, price - 1.5 * atr_f))
        else:
            sl = price - 1.5 * atr_f
        reasons.append(f"TP ~Rp {tp:,.0f} | SL ~Rp {sl:,.0f} (ATR={atr_f:,.0f})")

    reason_str = " | ".join(reasons) if reasons else "No strong signal"

    return TimingSignal(
        ticker=ticker, action=action, confidence=confidence,
        horizon=horizon, reason=reason_str,
        stoch_rsi_k=float(stoch_k) if pd.notna(stoch_k) else None,
        stoch_rsi_d=float(stoch_d) if pd.notna(stoch_d) else None,
        rsi=float(rsi_val) if pd.notna(rsi_val) else None,
        smi=float(smi_val) if pd.notna(smi_val) else None,
        macd_trend=macd_trend,
        macd_divergence=macd_div_str,
        roc_12=float(roc_12) if pd.notna(roc_12) else None,
        bandar_score=float(bandar) if pd.notna(bandar) else None,
        obv_momentum=obv_label,
        support_20=float(support_20) if pd.notna(support_20) else None,
        resistance_20=float(resistance_20) if pd.notna(resistance_20) else None,
        pivot=float(pivot) if pd.notna(pivot) else None,
        pivot_s1=float(s1) if pd.notna(s1) else None,
        pivot_r1=float(r1) if pd.notna(r1) else None,
        atr=float(atr_val) if pd.notna(atr_val) else None,
        vwap=float(vwap_val) if pd.notna(vwap_val) else None,
        bb_position=bb_pos,
        pbv=f.pbv, per=f.per, per_forward=f.per_forward,
        roe=f.roe, roa=f.roa, roic=f.roic,
        eps=f.eps, eps_growth=f.eps_growth,
        free_cashflow=f.free_cashflow, operating_cashflow=f.operating_cashflow,
        revenue=f.revenue, debt_to_equity=f.debt_to_equity,
        current_ratio=f.current_ratio, gross_margins=f.gross_margins,
        profit_margins=f.profit_margins, dividend_yield=f.dividend_yield,
        free_float_ratio=f.free_float_ratio, market_cap=f.market_cap,
        valuation_label=label,
        sector=f.sector, industry=f.industry,
        book_value_per_share=f.book_value_per_share,
        valuation_price=f.valuation_price,
        price_vs_valuation=(
            "CHEAP" if f.valuation_price and price < f.valuation_price
            else "EXPENSIVE" if f.valuation_price and price > f.valuation_price
            else None
        ),
        company_name=f.company_name,
        major_holders=f.major_holders,
        take_profit=tp, stop_loss=sl,
        price=price,
        timestamp=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
    )


def screen_rebound_candidates(
    tickers: Optional[list] = None,
    min_score: float = 50,
    rt_prices: Optional[dict] = None,
    data: Optional[dict[str, pd.DataFrame]] = None,
) -> list:
    if data is None:
        data = fetch_multiple_stocks(tickers=tickers, period="3mo")
    if not data:
        return []

    # Fetch real-time prices if not supplied
    if rt_prices is None:
        try:
            rt_prices, _ = fetch_realtime_prices(list(data.keys()))
        except Exception:
            rt_prices = {}

    candidates = []

    for ticker, df in data.items():
        if df.empty or len(df) < 30:
            continue

        # Inject real-time price into the last candle BEFORE computing indicators
        if ticker in rt_prices:
            df = update_last_candle_with_realtime(df, rt_prices[ticker])

        df = add_indicators(df)
        latest = df.iloc[-1]
        price = float(latest["close"])
        stoch_k = latest.get("stoch_rsi_k")
        smi = latest.get("smi")
        bandar = latest.get("bandar_score")

        if pd.isna(stoch_k):
            continue

        reasons = []
        score = 0.0

        if stoch_k < OVERSOLD_THRESHOLD:
            score += 40 * (1 - stoch_k / OVERSOLD_THRESHOLD)
            reasons.append(f"Stoch RSI oversold ({stoch_k:.1f})")

        smi_trend = "NEUTRAL"
        if pd.notna(smi):
            if smi > SMI_BULLISH_THRESHOLD:
                score += 30
                smi_trend = "ACCUMULATING"
                reasons.append("Smart money accumulating")
            elif smi < -0.5:
                smi_trend = "DISTRIBUTING"
                reasons.append("Smart money distributing")
            else:
                score += 10

        bandar_trend = "NEUTRAL"
        if pd.notna(bandar):
            if bandar > 2:
                score += 20
                bandar_trend = "BUYING"
                reasons.append(f"Big player buying (bandar={bandar:.1f})")
            elif bandar < -2:
                bandar_trend = "SELLING"
                reasons.append(f"Big player selling (bandar={bandar:.1f})")

        recent_change = 0.0
        if len(df) >= 5:
            recent_change = (price - df["close"].iloc[-5]) / df["close"].iloc[-5] * 100
        if len(df) >= 20:
            recent_high = df["close"].iloc[-20:].max()
            change_from_high = (price - recent_high) / recent_high * 100
            if change_from_high < -5:
                score += min(20, abs(change_from_high) * 1.5)
                reasons.append(f"Down {change_from_high:.1f}% from 20d high – rebound room")

        if score >= min_score:
            candidates.append(ReboundCandidate(
                ticker=ticker,
                rebound_score=round(min(100, score), 1),
                stoch_rsi_k=float(stoch_k),
                smi_trend=smi_trend,
                bandar_trend=bandar_trend,
                recent_change_pct=recent_change,
                reasons=reasons,
                price=price,
                timestamp=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
            ))

    return sorted(candidates, key=lambda x: x.rebound_score, reverse=True)


def get_all_signals(
    tickers: Optional[list] = None,
) -> tuple[list, dict, str]:
    """
    Fetch data, inject real-time prices into OHLCV, then compute indicators.

    Returns:
        (signals_list, rt_prices_dict, rt_source_label)
    """
    data = fetch_multiple_stocks(tickers=tickers, period="3mo")

    # Fetch real-time prices for all tickers in one batch
    rt_prices: dict[str, float] = {}
    rt_source_label = ""
    try:
        rt_prices, rt_source_label = fetch_realtime_prices(list(data.keys()))
    except Exception:
        rt_prices = {}
        rt_source_label = ""

    signals = []
    for ticker, df in data.items():
        # Inject real-time price into the last candle BEFORE computing indicators
        if ticker in rt_prices:
            df = update_last_candle_with_realtime(df, rt_prices[ticker])
        sig = analyze_buy_sell_timing(df, ticker)
        signals.append(sig)

    sorted_signals = sorted(signals, key=lambda x: (x.action != "HOLD", -x.confidence))
    return sorted_signals, rt_prices, rt_source_label


def analyze_ta_only(df: pd.DataFrame, ticker: str) -> TimingSignal:
    """
    Lightweight TA-only analysis — skips the expensive fetch_fundamentals() call.
    Used for bulk scanning of 800+ stocks in sector overview.
    """
    now = pd.Timestamp.now().isoformat()

    def _empty(action="HOLD"):
        return TimingSignal(
            ticker=ticker, action=action, confidence=0.0,
            horizon="neutral", reason="Insufficient data",
            stoch_rsi_k=None, stoch_rsi_d=None, rsi=None, smi=None,
            macd_trend=None, macd_divergence=None, roc_12=None,
            bandar_score=None, obv_momentum=None,
            support_20=None, resistance_20=None,
            pivot=None, pivot_s1=None, pivot_r1=None,
            atr=None, vwap=None, bb_position=None,
            pbv=None, per=None, per_forward=None,
            roe=None, roa=None, roic=None, eps=None,
            eps_growth=None, free_cashflow=None, operating_cashflow=None,
            revenue=None, debt_to_equity=None, current_ratio=None,
            gross_margins=None, profit_margins=None, dividend_yield=None,
            free_float_ratio=None, market_cap=None, valuation_label="unknown",
            sector=None, industry=None,
            book_value_per_share=None, valuation_price=None,
            price_vs_valuation=None, company_name=None, major_holders=None,
            take_profit=None, stop_loss=None, price=0.0, timestamp=now,
        )

    if df is None or df.empty or len(df) < 30:
        return _empty()

    df = add_indicators(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest

    price = float(latest["close"])
    stoch_k = latest.get("stoch_rsi_k")
    stoch_d = latest.get("stoch_rsi_d")
    rsi_val = latest.get("rsi")
    smi_val = latest.get("smi")
    macd_line = latest.get("macd_line")
    macd_signal_val = latest.get("macd_signal")
    macd_div_val = latest.get("macd_divergence")
    roc_12 = latest.get("roc_12")
    bandar = latest.get("bandar_score")
    obv_mom = latest.get("obv_momentum")
    support_20 = latest.get("support_20")
    resistance_20 = latest.get("resistance_20")
    pivot = latest.get("pivot")
    s1 = latest.get("pivot_s1")
    r1 = latest.get("pivot_r1")
    atr_val = latest.get("atr")
    vwap_val = latest.get("vwap")
    bb_upper = latest.get("bb_upper")
    bb_lower = latest.get("bb_lower")

    reasons: list[str] = []
    buy_score = 0.0
    sell_score = 0.0

    # ── Stochastic RSI
    if pd.notna(stoch_k):
        if stoch_k < OVERSOLD_THRESHOLD:
            reasons.append(f"Stoch RSI oversold ({stoch_k:.1f})")
            buy_score += 0.4
        elif stoch_k > OVERBOUGHT_THRESHOLD:
            reasons.append(f"Stoch RSI overbought ({stoch_k:.1f})")
            sell_score += 0.4
        if pd.notna(stoch_d):
            pk = prev.get("stoch_rsi_k", np.nan)
            pd_ = prev.get("stoch_rsi_d", np.nan)
            if pd.notna(pk) and pd.notna(pd_):
                if stoch_k > stoch_d and pk <= pd_:
                    reasons.append("Bullish crossover")
                    buy_score += 0.2
                elif stoch_k < stoch_d and pk >= pd_:
                    reasons.append("Bearish crossover")
                    sell_score += 0.2

    # ── RSI
    if pd.notna(rsi_val):
        if rsi_val < 30:
            reasons.append(f"RSI oversold ({rsi_val:.1f})")
            buy_score += 0.15
        elif rsi_val > 70:
            reasons.append(f"RSI overbought ({rsi_val:.1f})")
            sell_score += 0.15

    # ── MACD
    macd_trend: Optional[str] = None
    macd_div_str: Optional[str] = None
    if pd.notna(macd_line) and pd.notna(macd_signal_val):
        pm = prev.get("macd_line", np.nan)
        ps = prev.get("macd_signal", np.nan)
        if pd.notna(pm) and pd.notna(ps):
            if macd_line > macd_signal_val and pm <= ps:
                macd_trend = "golden_cross"
                reasons.append("MACD golden cross")
                buy_score += 0.3
            elif macd_line < macd_signal_val and pm >= ps:
                macd_trend = "dead_cross"
                reasons.append("MACD dead cross")
                sell_score += 0.3
    if pd.notna(macd_div_val):
        if macd_div_val == 1:
            macd_div_str = "bullish"
            buy_score += 0.2
        elif macd_div_val == -1:
            macd_div_str = "bearish"
            sell_score += 0.2

    # ── Smart Money
    if pd.notna(smi_val):
        if smi_val > SMI_BULLISH_THRESHOLD:
            reasons.append(f"Smart money accumulating (SMI={smi_val:.2f})")
            buy_score += 0.3
        elif smi_val < -0.5:
            reasons.append(f"Smart money distributing (SMI={smi_val:.2f})")
            sell_score += 0.3

    # ── Bandar
    if pd.notna(bandar):
        if bandar > 3:
            reasons.append(f"Bandar buying (score={bandar:.1f})")
            buy_score += 0.25
        elif bandar < -3:
            reasons.append(f"Bandar selling (score={bandar:.1f})")
            sell_score += 0.25

    # ── OBV
    obv_label: Optional[str] = None
    if pd.notna(obv_mom):
        if obv_mom > 0:
            obv_label = "INFLOW"
            buy_score += 0.15
        elif obv_mom < 0:
            obv_label = "OUTFLOW"
            sell_score += 0.15

    # ── ROC
    if pd.notna(roc_12):
        if roc_12 > 5:
            buy_score += 0.10
        elif roc_12 < -5:
            sell_score += 0.10

    # ── Bollinger
    bb_pos: Optional[str] = None
    if pd.notna(bb_upper) and pd.notna(bb_lower):
        if price < float(bb_lower):
            bb_pos = "BELOW"
            buy_score += 0.10
        elif price > float(bb_upper):
            bb_pos = "ABOVE"
            sell_score += 0.10
        else:
            bb_pos = "INSIDE"

    # ── Action
    action = "HOLD"
    confidence = 0.5
    if buy_score > sell_score and buy_score >= 0.4:
        action = "BUY"
        confidence = min(1.0, buy_score)
    elif sell_score > buy_score and sell_score >= 0.4:
        action = "SELL"
        confidence = min(1.0, sell_score)

    # ── TP / SL
    tp: Optional[float] = None
    sl: Optional[float] = None
    if action == "BUY" and pd.notna(atr_val):
        atr_f = float(atr_val)
        tp = float(min(float(resistance_20) * 0.98, price + 2.5 * atr_f)) if pd.notna(resistance_20) else price + 2.5 * atr_f
        sl = float(max(float(support_20) * 0.97, price - 1.5 * atr_f)) if pd.notna(support_20) else price - 1.5 * atr_f

    reason_str = " | ".join(reasons) if reasons else "No strong signal"

    return TimingSignal(
        ticker=ticker, action=action, confidence=confidence,
        horizon="neutral", reason=reason_str,
        stoch_rsi_k=float(stoch_k) if pd.notna(stoch_k) else None,
        stoch_rsi_d=float(stoch_d) if pd.notna(stoch_d) else None,
        rsi=float(rsi_val) if pd.notna(rsi_val) else None,
        smi=float(smi_val) if pd.notna(smi_val) else None,
        macd_trend=macd_trend,
        macd_divergence=macd_div_str,
        roc_12=float(roc_12) if pd.notna(roc_12) else None,
        bandar_score=float(bandar) if pd.notna(bandar) else None,
        obv_momentum=obv_label,
        support_20=float(support_20) if pd.notna(support_20) else None,
        resistance_20=float(resistance_20) if pd.notna(resistance_20) else None,
        pivot=float(pivot) if pd.notna(pivot) else None,
        pivot_s1=float(s1) if pd.notna(s1) else None,
        pivot_r1=float(r1) if pd.notna(r1) else None,
        atr=float(atr_val) if pd.notna(atr_val) else None,
        vwap=float(vwap_val) if pd.notna(vwap_val) else None,
        bb_position=bb_pos,
        pbv=None, per=None, per_forward=None,
        roe=None, roa=None, roic=None,
        eps=None, eps_growth=None,
        free_cashflow=None, operating_cashflow=None,
        revenue=None, debt_to_equity=None, current_ratio=None,
        gross_margins=None, profit_margins=None, dividend_yield=None,
        free_float_ratio=None, market_cap=None,
        valuation_label="unknown",
        sector=None, industry=None,
        book_value_per_share=None, valuation_price=None,
        price_vs_valuation=None,
        company_name=None, major_holders=None,
        take_profit=tp, stop_loss=sl,
        price=price,
        timestamp=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
    )


def get_all_signals_bulk(
    tickers: Optional[list] = None,
    progress_callback=None,
    interval: str = "1d",
    period: Optional[str] = None,
    bypass_cache: bool = False,
    return_data: bool = False,
) -> tuple:
    """
    Bulk-fetch OHLCV data for all tickers using yf.download(), then compute
    TA-only signals (no fundamentals). Much faster for 800+ stocks.

    Returns:
        (signals_list, rt_prices_dict, rt_source_label)
    """
    data = fetch_multiple_stocks_bulk(
        tickers=tickers,
        period=period,
        interval=interval,
        progress_callback=progress_callback,
        bypass_cache=bypass_cache,
    )

    # Fetch real-time prices for all tickers in one batch
    rt_prices: dict[str, float] = {}
    rt_source_label = ""
    try:
        rt_prices, rt_source_label = fetch_realtime_prices(list(data.keys()))
    except Exception:
        rt_prices = {}
        rt_source_label = ""

    signals = []
    for ticker, df in data.items():
        # Inject real-time price into the last candle
        if ticker in rt_prices:
            df = update_last_candle_with_realtime(df, rt_prices[ticker])
        sig = analyze_ta_only(df, ticker)
        signals.append(sig)

    sorted_signals = sorted(signals, key=lambda x: (x.action != "HOLD", -x.confidence))
    if return_data:
        return sorted_signals, rt_prices, rt_source_label, data
    return sorted_signals, rt_prices, rt_source_label
