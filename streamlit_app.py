"""
Indonesia Stock Trading AI - Streamlit App (Enhanced)
"""
import streamlit as st
import pandas as pd
import warnings
from datetime import datetime
from yfinance.exceptions import YFRateLimitError

warnings.filterwarnings("ignore", category=FutureWarning)

from analyzer import (
    get_all_signals,
    screen_rebound_candidates,
    TimingSignal,
    ReboundCandidate,
)
from universe import get_universe
from config import IDX_STOCKS

st.set_page_config(
    page_title="Indonesia Stock Trading AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Indonesia Stock Trading AI")
st.caption(
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
    "Stochastic RSI | MACD | Bandar Volume | Smart Money | Fundamentals"
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    universe_mode = st.radio(
        "Universe",
        ["All IDX stocks (~800+)", "LQ45 / core list (26)"],
        index=1,
    )
    batch_limit = st.slider(
        "Max stocks to scan",
        min_value=10, max_value=200,
        value=50 if "All IDX" in universe_mode else 26,
        step=10,
        help="Limit for 'All IDX' mode to avoid rate limits",
    )
    scan_all = False
    if "All IDX" in universe_mode:
        scan_all = st.checkbox(
            "Scan all IDX stocks (slow, may rate limit)",
            value=False,
        )
    min_rebound = st.slider("Rebound Min Score", 20, 80, 40)
    view_mode = st.radio(
        "View",
        ["By Action", "By Sector/Industry"],
        index=0,
    )
    group_mode = st.selectbox(
        "Group signals by",
        ["None", "Sector", "Industry"],
        index=1,
    )
    st.markdown("---")
    st.info(
        "**Indicators used:**\n"
        "- Stochastic RSI (14,3,3)\n"
        "- RSI (14)\n"
        "- MACD (12,26,9) + Divergence\n"
        "- Smart Money Index\n"
        "- Bandar Volume Detection\n"
        "- OBV / Foreign Flow\n"
        "- Bollinger Bands (20,2)\n"
        "- ATR-based TP/SL\n"
        "- Pivot S1/R1\n"
        "- PBV, PER, ROE, ROA, ROIC\n"
        "- EPS, FCF, Revenue\n"
        "- D/E, Current Ratio\n"
        "- Free Float Warning\n"
    )
    if st.button("🔄 Refresh Analysis"):
        st.cache_data.clear()
        st.rerun()

# ─── Ticker selection ─────────────────────────────────────────────────────────
if "All IDX" in universe_mode:
    all_tickers = get_universe(all_idx=True)
    tickers = all_tickers if scan_all else all_tickers[:batch_limit]
    st.sidebar.write(f"Scanning {len(tickers)} of {len(all_tickers)} IDX stocks.")
else:
    tickers = list(IDX_STOCKS)


# ─── Run analysis ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def run_signals(ticker_tuple):
    return get_all_signals(tickers=list(ticker_tuple))

@st.cache_data(ttl=600, show_spinner=False)
def run_rebounds(ticker_tuple, min_score):
    return screen_rebound_candidates(tickers=list(ticker_tuple), min_score=min_score)


try:
    with st.spinner("⏳ Loading stock data and running analysis (this may take a minute)..."):
        signals = run_signals(tuple(tickers))
        buy_signals  = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]
        hold_signals = [s for s in signals if s.action == "HOLD"]
        rebounds = run_rebounds(tuple(tickers), min_rebound)
except YFRateLimitError:
    st.error("Yahoo Finance rate limit reached. Please wait and try again.")
    st.info("Tips: reduce 'Max stocks to scan', switch to LQ45/core list, or retry in a few minutes.")
    st.stop()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def _fmt(val, fmt=".2f", suffix="", scale=1, na="-"):
    if val is None:
        return na
    return f"{val * scale:{fmt}}{suffix}"

def _pct(val, na="-"):
    return _fmt(val, ".1f", "%", 100, na)

def _rp(val, na="-"):
    if val is None:
        return na
    return f"Rp {val:,.0f}"

def _billions(val, na="-"):
    if val is None:
        return na
    return f"Rp {val/1e9:,.1f}B"


def signals_to_df(sig_list: list[TimingSignal]) -> pd.DataFrame:
    rows = []
    for s in sig_list:
        horizon_emoji = {
            "long-term": "🏦", "balanced": "⚖️",
            "short-term": "⚡", "speculative": "🎯",
            "neutral": "—",
        }.get(s.horizon, "—")

        macd_lbl = s.macd_trend or "-"
        if s.macd_divergence:
            macd_lbl += f" + {s.macd_divergence} div"

        ff_pct = f"{s.free_float_ratio*100:.1f}%" if s.free_float_ratio else "-"
        ff_flag = "⚠️" if s.free_float_ratio and s.free_float_ratio < 0.15 else (
                  "✅" if s.free_float_ratio and s.free_float_ratio > 0.40 else "")

        rows.append({
            "Ticker": s.ticker,
            "Action": s.action,
            "Conf": f"{s.confidence:.0%}",
            "Horizon": f"{horizon_emoji} {s.horizon}",
            "Valuation": s.valuation_label,
            "Sector": s.sector or "-",
            "Industry": s.industry or "-",
            # TA
            "StochRSI": _fmt(s.stoch_rsi_k, ".1f"),
            "RSI": _fmt(s.rsi, ".1f"),
            "SMI": _fmt(s.smi, ".2f"),
            "MACD": macd_lbl,
            "Bandar": _fmt(s.bandar_score, ".1f"),
            "OBV Flow": s.obv_momentum or "-",
            "ROC12%": _fmt(s.roc_12, ".1f", "%"),
            "BB Pos": s.bb_position or "-",
            # Levels
            "Price": _rp(s.price),
            "VWAP": _rp(s.vwap),
            "Support": _rp(s.support_20),
            "Resist": _rp(s.resistance_20),
            "Pivot": _rp(s.pivot),
            "S1": _rp(s.pivot_s1),
            "R1": _rp(s.pivot_r1),
            "ATR": _rp(s.atr),
            "Take Profit": _rp(s.take_profit),
            "Stop Loss": _rp(s.stop_loss),
            # Fundamentals – Valuation
            "PBV": _fmt(s.pbv, ".2f", "x"),
            "PER": _fmt(s.per, ".1f", "x"),
            "PER Fwd": _fmt(s.per_forward, ".1f", "x"),
            # Profitability
            "ROE": _pct(s.roe),
            "ROA": _pct(s.roa),
            "ROIC": _pct(s.roic),
            "EPS": _fmt(s.eps, ",.0f", " Rp") if s.eps else "-",
            "EPS Growth": _pct(s.eps_growth),
            "Gross Mgn": _pct(s.gross_margins),
            "Net Mgn": _pct(s.profit_margins),
            # Cash Flow
            "FCF": _billions(s.free_cashflow),
            "Op CF": _billions(s.operating_cashflow),
            "Revenue": _billions(s.revenue),
            # Solvency
            "D/E": _fmt(s.debt_to_equity, ".1f", "x"),
            "Curr Ratio": _fmt(s.current_ratio, ".1f", "x"),
            # Liquidity
            f"Free Float {ff_flag}": f"{ff_pct}",
            "Mkt Cap": _billions(s.market_cap),
            "Div Yield": _pct(s.dividend_yield),
            "Reason": s.reason[:150] + "..." if len(s.reason) > 150 else s.reason,
        })
    return pd.DataFrame(rows)


def _group_signals(sig_list: list[TimingSignal], mode: str) -> dict[str, list[TimingSignal]]:
    if mode == "None":
        return {"All": sig_list}
    groups: dict[str, list[TimingSignal]] = {}
    for s in sig_list:
        if mode == "Industry":
            key = s.industry or s.sector or "Unknown"
        else:
            key = s.sector or "Unknown"
        groups.setdefault(key, []).append(s)
    return dict(sorted(groups.items(), key=lambda x: x[0]))


def render_signals(sig_list: list[TimingSignal], group: str):
    if not sig_list:
        return
    if group == "None":
        st.dataframe(signals_to_df(sig_list), use_container_width=True, hide_index=True)
        return
    grouped = _group_signals(sig_list, group)
    for label, items in grouped.items():
        st.markdown(f"**{label} ({len(items)})**")
        st.dataframe(signals_to_df(items), use_container_width=True, hide_index=True)


def rebounds_to_df(candidates: list[ReboundCandidate]) -> pd.DataFrame:
    rows = []
    for c in candidates[:20]:
        rows.append({
            "Ticker": c.ticker,
            "Rebound Score": f"{c.rebound_score:.1f}",
            "Stoch RSI": f"{c.stoch_rsi_k:.1f}",
            "SMI Trend": c.smi_trend,
            "Bandar": c.bandar_trend,
            "5d Change": f"{c.recent_change_pct:+.1f}%",
            "Price (Rp)": f"{c.price:,.0f}",
            "Reasons": " | ".join(c.reasons[:3]),
        })
    return pd.DataFrame(rows)


# ─── Summary metrics ──────────────────────────────────────────────────────────
st.header("📊 Signal Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("🟢 BUY", len(buy_signals))
c2.metric("🔴 SELL", len(sell_signals))
c3.metric("⚪ HOLD", len(hold_signals))
c4.metric("🔁 Rebound Candidates", len(rebounds))

# ─── Signals View ─────────────────────────────────────────────────────────────
if view_mode == "By Sector/Industry":
    st.header("🗂️ All Signals Grouped")
    render_signals(signals, group_mode)
else:
    # ─── BUY ──────────────────────────────────────────────────────────────────
    if buy_signals:
        st.header("🟢 Buy Opportunities")
        with st.expander("🏦 Long-Term (Undervalued)", expanded=True):
            lt = [s for s in buy_signals if s.horizon == "long-term"]
            if lt:
                render_signals(lt, group_mode)
            else:
                st.info("No long-term undervalued buys at this time.")
        with st.expander("⚡ Short-Term / Momentum", expanded=True):
            st_ = [s for s in buy_signals if s.horizon in ("short-term", "speculative", "balanced", "neutral")]
            if st_:
                render_signals(st_, group_mode)
            else:
                st.info("No short-term momentum buys at this time.")

    # ─── SELL ─────────────────────────────────────────────────────────────────
    if sell_signals:
        st.header("🔴 Sell Signals")
        render_signals(sell_signals, group_mode)

    # ─── HOLD ─────────────────────────────────────────────────────────────────
    if hold_signals:
        with st.expander("⚪ Hold / No Strong Signal (first 15)", expanded=False):
            render_signals(hold_signals[:15], group_mode)

# ─── REBOUND ──────────────────────────────────────────────────────────────────
st.header("🔁 Potential Rebound Candidates")
st.caption("Oversold Stoch RSI + Smart Money + Bandar accumulation")
if rebounds:
    st.dataframe(rebounds_to_df(rebounds), use_container_width=True, hide_index=True)
else:
    st.info("No strong rebound candidates. Try lowering the min score in the sidebar.")

# ─── Guide ────────────────────────────────────────────────────────────────────
with st.expander("📖 Indicator & Signal Guide"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Technical Signals**
        - **Stoch RSI < 20** → oversold → potential buy
        - **Stoch RSI > 80** → overbought → potential sell
        - **RSI < 30** → deeply oversold
        - **MACD golden cross** → bullish momentum
        - **MACD dead cross** → bearish momentum
        - **MACD bullish divergence** → price new low, MACD higher low → reversal up
        - **MACD bearish divergence** → price new high, MACD lower high → reversal down
        - **Bandar > 3** → big player buying (unusual volume + up candle)
        - **Bandar < -3** → big player selling
        - **OBV INFLOW** → institutional/foreign accumulation proxy
        - **BB BELOW** → price below Bollinger lower band → potential bounce
        """)
    with col2:
        st.markdown("""
        **Fundamental Signals**
        - **PBV < 1.5 + PER < 15** → undervalued → long-term horizon
        - **PBV > 4 / PER > 30** → expensive → short-term only
        - **ROE > 15%** → high capital efficiency
        - **ROIC > 12%** → strong return on invested capital
        - **D/E > 2x** → high leverage risk
        - **Current Ratio < 1** → short-term liquidity risk
        - **Positive FCF** → healthy cash generation
        - **Free Float < 15%** → ⚠️ illiquid, manipulation risk
        - **Free Float > 40%** → ✅ good market liquidity

        **Confidence Score**
        - Technical signals (0–100%)
        - +25% if undervalued
        - ±10% for free float
        - +5% each: high ROE, ROIC, positive FCF, EPS growth
        """)

with st.expander("⚠️ Disclaimer"):
    st.warning(
        "This tool is for **educational and research purposes only**. "
        "Data sourced from Yahoo Finance (≈15 min delay). "
        "Past performance does not guarantee future results. "
        "Always do your own research (DYOR) before making any investment decisions. "
        "This is not financial advice."
    )
