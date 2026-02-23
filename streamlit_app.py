"""
Indonesia Stock Trading AI - Streamlit App
"""
import streamlit as st
import pandas as pd
import warnings
from datetime import datetime

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
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Stochastic RSI + Smart Money Analysis")

# Sidebar
with st.sidebar:
    st.header("Settings")
    universe_mode = st.radio(
        "Universe",
        ["All IDX stocks", "LQ45 / core list"],
        index=0,
    )
    min_rebound = st.slider("Rebound Min Score", 20, 80, 40)
    if st.button("🔄 Refresh Analysis"):
        st.rerun()

if universe_mode == "All IDX stocks":
    # Limit to first 50 to avoid yfinance rate limits
    all_tickers = get_universe(all_idx=True)
    tickers = all_tickers[:50]
    st.sidebar.write(f"Scanning {len(tickers)} out of {len(all_tickers)} IDX stocks (first 50).")
else:
    tickers = list(IDX_STOCKS)

# Run analysis
with st.spinner("Loading stock data and running analysis..."):
    signals = get_all_signals(tickers=tickers)
    buy_signals = [s for s in signals if s.action == "BUY"]
    sell_signals = [s for s in signals if s.action == "SELL"]
    hold_signals = [s for s in signals if s.action == "HOLD"]
    rebounds = screen_rebound_candidates(tickers=tickers, min_score=min_rebound)


def signals_to_df(signals_list: list[TimingSignal]) -> pd.DataFrame:
    rows = []
    for s in signals_list:
        stoch = f"{s.stoch_rsi_k:.1f}" if s.stoch_rsi_k is not None else "-"
        smi = f"{s.smi:.2f}" if s.smi is not None else "-"
        pbv = f"{s.pbv:.2f}" if s.pbv is not None else "-"
        per = f"{s.per:.1f}" if s.per is not None else "-"
        roe = f"{s.roe * 100:.1f}%" if s.roe is not None else "-"
        ff = f"{s.free_float_ratio * 100:.1f}%" if s.free_float_ratio is not None else "-"
        tp = f"{s.take_profit:,.0f}" if s.take_profit is not None else "-"
        sl = f"{s.stop_loss:,.0f}" if s.stop_loss is not None else "-"
        rows.append({
            "Ticker": s.ticker,
            "Action": s.action,
            "Horizon": s.horizon,
            "Confidence": f"{s.confidence:.0%}",
            "Stoch RSI": stoch,
            "SMI": smi,
            "MACD": s.macd_trend or "-",
            "PBV": pbv,
            "PER": per,
            "ROE": roe,
            "Free Float": ff,
            "Price (Rp)": f"{s.price:,.0f}",
            "TP (Rp)": tp,
            "SL (Rp)": sl,
            "Reason": s.reason[:120] + "..." if len(s.reason) > 120 else s.reason,
        })
    return pd.DataFrame(rows)


def rebounds_to_df(candidates: list[ReboundCandidate]) -> pd.DataFrame:
    rows = []
    for c in candidates[:15]:
        rows.append({
            "Ticker": c.ticker,
            "Rebound Score": f"{c.rebound_score:.1f}",
            "Stoch RSI": f"{c.stoch_rsi_k:.1f}",
            "SMI Trend": c.smi_trend,
            "5d Change": f"{c.recent_change_pct:+.1f}%",
            "Price (Rp)": f"{c.price:,.0f}",
            "Reasons": "; ".join(c.reasons[:2]),
        })
    return pd.DataFrame(rows)


# Buy/Sell signals
st.header("Buy / Sell Timing Signals")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("BUY", len(buy_signals))
with col2:
    st.metric("SELL", len(sell_signals))
with col3:
    st.metric("HOLD", len(hold_signals))

if buy_signals:
    st.subheader("🟢 Buy Opportunities")
    st.dataframe(signals_to_df(buy_signals), use_container_width=True, hide_index=True)

if sell_signals:
    st.subheader("🔴 Sell Signals")
    st.dataframe(signals_to_df(sell_signals), use_container_width=True, hide_index=True)

if hold_signals:
    st.subheader("⚪ Hold (no strong signal)")
    st.dataframe(signals_to_df(hold_signals[:10]), use_container_width=True, hide_index=True)

# Rebound candidates
st.header("Potential Rebound Candidates")
st.caption("Oversold Stoch RSI + Smart Money accumulation")

if rebounds:
    st.dataframe(rebounds_to_df(rebounds), use_container_width=True, hide_index=True)
else:
    st.info("No strong rebound candidates. Try lowering the min score in the sidebar.")

# Tips
with st.expander("💡 Indicator Guide"):
    st.markdown("""
    - **Stoch RSI < 20**: Oversold (potential buy)
    - **Stoch RSI > 80**: Overbought (potential sell)
    - **SMI > 0**: Smart money accumulating
    - **SMI < 0**: Smart money distributing
    - Run regularly for real-time updates
    """)
