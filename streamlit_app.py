"""
Indonesia Stock Trading AI - Streamlit App (Two-Tab Layout)
Tab 1: Ticker Search & Deep Analysis (Valuation, Sentiment, Catalysts)
Tab 2: Sector Overview (Grouped view with ownership, color-coded signals)
"""
import streamlit as st
import pandas as pd
import time
import warnings
from datetime import datetime
from data_fetcher import DataProviderError, DataRateLimitError

warnings.filterwarnings("ignore", category=FutureWarning)

from analyzer import (
    get_all_signals,
    get_all_signals_bulk,
    analyze_buy_sell_timing,
    screen_rebound_candidates,
    TimingSignal,
    ReboundCandidate,
)
from data_fetcher import fetch_stock_data, fetch_realtime_prices, update_last_candle_with_realtime
from universe import get_universe
from config import IDX_STOCKS
from sentiment import fetch_ticker_sentiment, fetch_sector_sentiment, sentiment_summary

st.set_page_config(
    page_title="Indonesia Stock Trading AI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Indonesia Stock Trading AI")
st.caption(
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
    "Stochastic RSI | MACD | Bandar Volume | Smart Money | Fundamentals | Sentiment"
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
        min_value=10, max_value=1000,
        value=200 if "All IDX" in universe_mode else 26,
        step=50,
        help="Limit for 'All IDX' mode. Use bulk download for 200+ stocks.",
    )
    scan_all = False
    if "All IDX" in universe_mode:
        scan_all = st.checkbox(
            "Scan all IDX stocks (slow, may rate limit)",
            value=False,
        )
    min_rebound = st.slider("Rebound Min Score", 20, 80, 40)

    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 2 min)", value=False)
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
        "- News Sentiment Analysis\n"
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
@st.cache_data(ttl=120, show_spinner=False)
def run_signals(ticker_tuple):
    """Use bulk download + TA-only for fast scanning of many tickers."""
    return get_all_signals_bulk(tickers=list(ticker_tuple))

@st.cache_data(ttl=120, show_spinner=False)
def run_rebounds(ticker_tuple, min_score, _rt_prices_tuple):
    rt_prices = dict(_rt_prices_tuple) if _rt_prices_tuple else {}
    return screen_rebound_candidates(
        tickers=list(ticker_tuple), min_score=min_score, rt_prices=rt_prices
    )

@st.cache_data(ttl=300, show_spinner=False)
def run_single_ticker_analysis(ticker):
    """Run analysis for a single ticker with real-time price injection."""
    df = fetch_stock_data(ticker, period="3mo")
    if df is None or df.empty:
        return None, None, ""
    # Get real-time price
    rt_prices = {}
    rt_source = ""
    try:
        rt_prices, rt_source = fetch_realtime_prices([ticker])
    except Exception:
        pass
    if ticker in rt_prices:
        df = update_last_candle_with_realtime(df, rt_prices[ticker])
    sig = analyze_buy_sell_timing(df, ticker)
    return sig, rt_prices.get(ticker), rt_source

@st.cache_data(ttl=600, show_spinner=False)
def run_sentiment_ticker(ticker, company_name):
    return fetch_ticker_sentiment(ticker, company_name, max_items=12)

@st.cache_data(ttl=600, show_spinner=False)
def run_sentiment_sector(sector):
    return fetch_sector_sentiment(sector, max_items=10)


try:
    with st.spinner(f"⏳ Bulk downloading {len(tickers)} stocks & running TA analysis..."):
        signals, rt_prices, rt_source_label = run_signals(tuple(tickers))
        buy_signals  = [s for s in signals if s.action == "BUY"]
        sell_signals = [s for s in signals if s.action == "SELL"]
        hold_signals = [s for s in signals if s.action == "HOLD"]
        rt_prices_tuple = tuple(sorted(rt_prices.items())) if rt_prices else ()
        rebounds = run_rebounds(tuple(tickers), min_rebound, rt_prices_tuple)
except DataRateLimitError:
    st.error("Data provider rate limit reached. Please wait and try again.")
    st.info("Tips: reduce 'Max stocks to scan', switch to LQ45/core list, or retry in a few minutes.")
    st.stop()
except DataProviderError as exc:
    st.error(f"Data provider error: {exc}")
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


def _color_text(text: str, sentiment: str) -> str:
    """Return HTML-styled text with color based on sentiment."""
    if sentiment == "positive":
        return f'<span style="color: #22c55e; font-weight: 600;">{text}</span>'
    elif sentiment == "negative":
        return f'<span style="color: #ef4444; font-weight: 600;">{text}</span>'
    return text


def _color_valuation(label: str) -> str:
    """Color the valuation label."""
    if label == "CHEAP":
        return '<span style="color: #22c55e; font-weight: 700; font-size: 1.1em;">✅ CHEAP (Below Fair Value)</span>'
    elif label == "EXPENSIVE":
        return '<span style="color: #ef4444; font-weight: 700; font-size: 1.1em;">⚠️ EXPENSIVE (Above Fair Value)</span>'
    return '<span style="color: #94a3b8;">N/A</span>'


def _reason_bullets(reason_str: str) -> str:
    """Convert pipe-separated reasons to HTML bullet list with color coding."""
    if not reason_str:
        return "-"
    parts = [r.strip() for r in reason_str.split("|") if r.strip()]
    lines = []
    for p in parts:
        lower = p.lower()
        # Determine sentiment of each reason line
        positive_kw = ["oversold", "bullish", "accumulating", "buying", "undervalued",
                       "positive", "inflow", "high capital", "strong", "good",
                       "healthy", "low debt", "growth", "bounce", "golden cross"]
        negative_kw = ["overbought", "bearish", "distributing", "selling", "overvalued",
                       "negative", "outflow", "high leverage", "concern", "risk",
                       "dead cross", "burn rate", "expensive", "rich valuation"]
        is_pos = any(kw in lower for kw in positive_kw)
        is_neg = any(kw in lower for kw in negative_kw)
        if is_pos and not is_neg:
            lines.append(f'<span style="color: #22c55e;">• {p}</span>')
        elif is_neg and not is_pos:
            lines.append(f'<span style="color: #ef4444;">• {p}</span>')
        else:
            lines.append(f"• {p}")
    return "<br>".join(lines)


# ─── TAB LAYOUT ──────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Ticker Search & Analysis", "🗂️ Sector Overview"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: TICKER SEARCH & DEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Ticker Search & Deep Analysis")

    # Build ticker list for search (with company names from signals if available)
    ticker_options = sorted(set(t for t in tickers))
    # Map ticker -> company name from signals
    ticker_name_map = {}
    for s in signals:
        if s.company_name:
            ticker_name_map[s.ticker] = s.company_name

    display_options = [
        f"{t} — {ticker_name_map.get(t, '')}" if ticker_name_map.get(t) else t
        for t in ticker_options
    ]

    col_search, col_extra = st.columns([3, 1])
    with col_search:
        selected_display = st.selectbox(
            "Search Ticker (Emiten)",
            options=display_options,
            index=None,
            placeholder="Type or select a ticker (e.g. BBCA.JK)...",
        )
    with col_extra:
        # Allow manual entry for tickers not in scanned list
        manual_ticker = st.text_input("Or enter ticker manually", placeholder="e.g. BBCA.JK")

    # Determine selected ticker
    selected_ticker = None
    if selected_display:
        selected_ticker = selected_display.split(" — ")[0].strip()
    elif manual_ticker:
        t = manual_ticker.strip().upper()
        if not t.endswith(".JK"):
            t += ".JK"
        selected_ticker = t

    if selected_ticker:
        # Try to find from pre-scanned signals first
        sig = next((s for s in signals if s.ticker == selected_ticker), None)

        if sig is None:
            # Fetch fresh for this ticker
            with st.spinner(f"⏳ Analyzing {selected_ticker}..."):
                try:
                    sig, rt_price, rt_src = run_single_ticker_analysis(selected_ticker)
                except Exception as e:
                    st.error(f"Error analyzing {selected_ticker}: {e}")
                    sig = None

        if sig and sig.price > 0:
            # ─── Header ──────────────────────────────────────────────────
            st.markdown("---")
            name_label = f" — {sig.company_name}" if sig.company_name else ""
            st.subheader(f"📊 {sig.ticker}{name_label}")

            # ─── Valuation Panel ─────────────────────────────────────────
            st.markdown("### 💰 Valuation Analysis")
            v_cols = st.columns(5)
            v_cols[0].metric("Current Price", _rp(sig.price))
            v_cols[1].metric("PER", _fmt(sig.per, ".1f", "x") if sig.per else "-")
            v_cols[2].metric("EPS (TTM)", _fmt(sig.eps, ",.0f", " Rp") if sig.eps else "-")
            v_cols[3].metric("Valuation Price", _rp(sig.valuation_price))
            v_cols[4].metric("PBV", _fmt(sig.pbv, ".2f", "x") if sig.pbv else "-")

            # Cheap/Expensive label
            if sig.price_vs_valuation:
                st.markdown(_color_valuation(sig.price_vs_valuation), unsafe_allow_html=True)
                if sig.valuation_price:
                    diff = sig.price - sig.valuation_price
                    diff_pct = (diff / sig.valuation_price) * 100
                    direction = "above" if diff > 0 else "below"
                    st.caption(f"Current price is Rp {abs(diff):,.0f} ({abs(diff_pct):.1f}%) {direction} the PER-implied fair value")
            else:
                st.caption("Valuation price unavailable (missing PER or EPS data)")

            # ─── Action & Confidence ─────────────────────────────────────
            st.markdown("### 🎯 Signal")
            a_cols = st.columns(4)
            action_colors = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}
            a_cols[0].metric(f"{action_colors.get(sig.action, '')} Action", sig.action)
            a_cols[1].metric("Confidence", f"{sig.confidence:.0%}")
            a_cols[2].metric("Horizon", sig.horizon.title())
            a_cols[3].metric("Valuation", sig.valuation_label.title())

            # ─── Technical Analysis Panel ────────────────────────────────
            st.markdown("### 📈 Technical Analysis")
            t_cols = st.columns(6)
            t_cols[0].metric("StochRSI", _fmt(sig.stoch_rsi_k, ".1f"))
            t_cols[1].metric("RSI", _fmt(sig.rsi, ".1f"))
            t_cols[2].metric("SMI", _fmt(sig.smi, ".2f"))
            macd_lbl = sig.macd_trend or "-"
            if sig.macd_divergence:
                macd_lbl += f" + {sig.macd_divergence} div"
            t_cols[3].metric("MACD", macd_lbl)
            t_cols[4].metric("Bandar Score", _fmt(sig.bandar_score, ".1f"))
            t_cols[5].metric("OBV Flow", sig.obv_momentum or "-")

            t2_cols = st.columns(6)
            t2_cols[0].metric("Support", _rp(sig.support_20))
            t2_cols[1].metric("Resistance", _rp(sig.resistance_20))
            t2_cols[2].metric("VWAP", _rp(sig.vwap))
            t2_cols[3].metric("ATR", _rp(sig.atr))
            t2_cols[4].metric("Take Profit", _rp(sig.take_profit))
            t2_cols[5].metric("Stop Loss", _rp(sig.stop_loss))

            # ─── Fundamentals Panel ──────────────────────────────────────
            st.markdown("### 📋 Fundamentals")
            f_cols = st.columns(6)
            f_cols[0].metric("ROE", _pct(sig.roe))
            f_cols[1].metric("ROA", _pct(sig.roa))
            f_cols[2].metric("ROIC", _pct(sig.roic))
            f_cols[3].metric("EPS", _fmt(sig.eps, ",.0f", " Rp") if sig.eps else "-")
            f_cols[4].metric("EPS Growth", _pct(sig.eps_growth))
            f_cols[5].metric("D/E Ratio", _fmt(sig.debt_to_equity, ".1f", "x") if sig.debt_to_equity else "-")

            f2_cols = st.columns(6)
            f2_cols[0].metric("FCF", _billions(sig.free_cashflow))
            f2_cols[1].metric("Op Cash Flow", _billions(sig.operating_cashflow))
            f2_cols[2].metric("Revenue", _billions(sig.revenue))
            f2_cols[3].metric("Current Ratio", _fmt(sig.current_ratio, ".1f", "x") if sig.current_ratio else "-")
            f2_cols[4].metric("Market Cap", _billions(sig.market_cap))
            f2_cols[5].metric("Div Yield", _pct(sig.dividend_yield))

            ff_pct = f"{sig.free_float_ratio*100:.1f}%" if sig.free_float_ratio else "-"
            if sig.free_float_ratio and sig.free_float_ratio < 0.15:
                st.warning(f"⚠️ LOW Free Float: {ff_pct} — illiquid, high manipulation risk")
            elif sig.free_float_ratio and sig.free_float_ratio > 0.40:
                st.success(f"✅ HIGH Free Float: {ff_pct} — good market liquidity")
            else:
                st.info(f"Free Float: {ff_pct}")

            # ─── Ownership ───────────────────────────────────────────────
            if sig.major_holders:
                st.markdown("### 👥 Major Holders / Ownership")
                st.markdown(f"**{sig.major_holders}**")

            # ─── Analysis Reasons (color-coded bullets) ──────────────────
            st.markdown("### 📝 Analysis Reasons")
            st.markdown(_reason_bullets(sig.reason), unsafe_allow_html=True)

            # ─── Sentiment & Catalysts ───────────────────────────────────
            st.markdown("### 📰 Sentiment & Catalyst Analysis")
            with st.spinner("Fetching news sentiment..."):
                sent_items = run_sentiment_ticker(selected_ticker, sig.company_name)

            if sent_items:
                summary = sentiment_summary(sent_items)
                s_cols = st.columns(4)
                s_cols[0].metric("Overall", summary["overall"].upper())
                s_cols[1].metric("🟢 Positive", summary["positive"])
                s_cols[2].metric("🔴 Negative", summary["negative"])
                s_cols[3].metric("⚪ Neutral", summary["neutral"])

                st.markdown("#### Recent News & Catalysts")
                for item in sent_items:
                    emoji = "🟢" if item.sentiment == "positive" else ("🔴" if item.sentiment == "negative" else "⚪")
                    cat_emoji = {
                        "company_action": "🏢",
                        "sector_demand": "📊",
                        "commodity": "🪨",
                        "ownership_change": "👥",
                        "macro": "🌍",
                        "regulatory": "📜",
                    }.get(item.category, "📰")
                    color = "#22c55e" if item.sentiment == "positive" else ("#ef4444" if item.sentiment == "negative" else "#94a3b8")
                    st.markdown(
                        f'{emoji} {cat_emoji} <span style="color: {color};">{item.headline}</span> '
                        f'<span style="color: #64748b; font-size: 0.85em;">({item.date}) [{item.category}]</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No recent news found for this ticker.")
        elif selected_ticker:
            st.warning(f"No data available for {selected_ticker}. Check if the ticker is valid.")
    else:
        # Show full analysis for ALL stocks when no ticker selected
        st.info("👆 Select or type a ticker above for deep-dive analysis with sentiment.")
        st.markdown("---")

        # Show summary metrics
        st.markdown("### 📊 Market Overview")
        if rt_source_label:
            st.caption(f"📡 **Live prices + indicators**: {rt_source_label} | Bulk download mode")
        else:
            st.caption("📊 Prices & Indicators: yfinance bulk download (may be ~15-min delayed)")
        st.info("💡 **Tip**: Fundamentals (PER, ROE, etc.) are loaded on-demand when you select a ticker above. Overview shows TA signals only for speed.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY", len(buy_signals))
        c2.metric("🔴 SELL", len(sell_signals))
        c3.metric("⚪ HOLD", len(hold_signals))
        c4.metric("🔁 Rebound Candidates", len(rebounds))

        # ─── Helper: full signal table with ALL indicators ───────────────
        def _full_signal_row(s):
            """Build a complete row dict with all TA + fundamentals."""
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

            val_label = ""
            if s.price_vs_valuation:
                val_label = f"{'🟢' if s.price_vs_valuation == 'CHEAP' else '🔴'} {s.price_vs_valuation}"
            else:
                val_label = s.valuation_label or "-"

            return {
                "Ticker": s.ticker,
                "Action": s.action,
                "Conf": f"{s.confidence:.0%}",
                "Horizon": f"{horizon_emoji} {s.horizon}",
                "Valuation": val_label,
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
                "Val Price": _rp(s.valuation_price),
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
                f"Free Float {ff_flag}": ff_pct,
                "Mkt Cap": _billions(s.market_cap),
                "Div Yield": _pct(s.dividend_yield),
                "Reason": s.reason[:200] + "..." if len(s.reason) > 200 else s.reason,
            }

        # ─── BUY SIGNALS ─────────────────────────────────────────────────
        if buy_signals:
            st.header("🟢 Buy Opportunities")
            with st.expander("🏦 Long-Term (Undervalued — PBV+PER suggest hold)", expanded=True):
                lt = [s for s in buy_signals if s.horizon == "long-term"]
                if lt:
                    st.dataframe(
                        pd.DataFrame([_full_signal_row(s) for s in lt]),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No long-term undervalued buys at this time.")
            with st.expander("⚡ Short-Term / Momentum (TA buy but expensive valuation)", expanded=True):
                st_ = [s for s in buy_signals if s.horizon in ("short-term", "speculative", "balanced", "neutral")]
                if st_:
                    st.dataframe(
                        pd.DataFrame([_full_signal_row(s) for s in st_]),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.info("No short-term momentum buys at this time.")

        # ─── SELL SIGNALS ────────────────────────────────────────────────
        if sell_signals:
            st.header("🔴 Sell Signals")
            st.dataframe(
                pd.DataFrame([_full_signal_row(s) for s in sell_signals]),
                use_container_width=True, hide_index=True,
            )

        # ─── HOLD ────────────────────────────────────────────────────────
        if hold_signals:
            with st.expander(f"⚪ Hold / No Strong Signal ({len(hold_signals)} stocks)", expanded=False):
                st.dataframe(
                    pd.DataFrame([_full_signal_row(s) for s in hold_signals[:20]]),
                    use_container_width=True, hide_index=True,
                )

        # ─── REBOUND CANDIDATES ──────────────────────────────────────────
        st.header("🔁 Potential Rebound Candidates")
        st.caption("Oversold Stoch RSI + Smart Money + Bandar accumulation")
        if rebounds:
            rb_data = []
            for c in rebounds[:20]:
                rb_data.append({
                    "Ticker": c.ticker,
                    "Rebound Score": f"{c.rebound_score:.1f}",
                    "Stoch RSI": f"{c.stoch_rsi_k:.1f}",
                    "SMI Trend": c.smi_trend,
                    "Bandar": c.bandar_trend,
                    "5d Change": f"{c.recent_change_pct:+.1f}%",
                    "Price (Rp)": f"{c.price:,.0f}",
                    "Reasons": " | ".join(c.reasons[:3]),
                })
            st.dataframe(pd.DataFrame(rb_data), use_container_width=True, hide_index=True)
        else:
            st.info("No strong rebound candidates. Try lowering the min score in the sidebar.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SECTOR OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🗂️ Sector Overview")

    # Collect unique sectors from signals
    all_sectors = sorted(set(s.sector for s in signals if s.sector))
    if not all_sectors:
        all_sectors = ["Financial Services", "Technology", "Consumer Cyclical",
                       "Basic Materials", "Energy", "Industrials",
                       "Consumer Defensive", "Healthcare", "Real Estate",
                       "Communication Services", "Utilities"]

    selected_sector = st.selectbox(
        "Select Sector",
        options=all_sectors,
        index=0,
        placeholder="Choose a sector...",
    )

    if selected_sector:
        sector_signals = [s for s in signals if s.sector == selected_sector]

        if sector_signals:
            # Sector summary metrics
            s_buy = [s for s in sector_signals if s.action == "BUY"]
            s_sell = [s for s in sector_signals if s.action == "SELL"]
            s_hold = [s for s in sector_signals if s.action == "HOLD"]

            m_cols = st.columns(4)
            m_cols[0].metric(f"📊 Total in {selected_sector}", len(sector_signals))
            m_cols[1].metric("🟢 BUY", len(s_buy))
            m_cols[2].metric("🔴 SELL", len(s_sell))
            m_cols[3].metric("⚪ HOLD", len(s_hold))

            # ─── Sector sentiment ────────────────────────────────────────
            with st.expander("📰 Sector Sentiment & Dependent Analysis", expanded=True):
                with st.spinner("Fetching sector news..."):
                    sector_sent = run_sentiment_sector(selected_sector)

                if sector_sent:
                    s_summary = sentiment_summary(sector_sent)
                    ss_cols = st.columns(4)
                    ss_cols[0].metric("Overall Sentiment", s_summary["overall"].upper())
                    ss_cols[1].metric("🟢 Positive", s_summary["positive"])
                    ss_cols[2].metric("🔴 Negative", s_summary["negative"])
                    ss_cols[3].metric("⚪ Neutral", s_summary["neutral"])

                    for item in sector_sent:
                        emoji = "🟢" if item.sentiment == "positive" else ("🔴" if item.sentiment == "negative" else "⚪")
                        color = "#22c55e" if item.sentiment == "positive" else ("#ef4444" if item.sentiment == "negative" else "#94a3b8")
                        st.markdown(
                            f'{emoji} <span style="color: {color};">{item.headline}</span> '
                            f'<span style="color: #64748b; font-size: 0.85em;">({item.date}) [{item.category}]</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No recent sector news found.")

            # ─── Sector stocks table ─────────────────────────────────────
            st.markdown("### 📋 Stocks in this Sector")

            # Build HTML table with color coding and autofit
            rows_html = []
            for s in sorted(sector_signals, key=lambda x: (-x.confidence, x.ticker)):
                # Action color
                action_color = "#22c55e" if s.action == "BUY" else ("#ef4444" if s.action == "SELL" else "#94a3b8")

                # Valuation color
                val_text = s.price_vs_valuation or s.valuation_label or "-"
                val_color = "#22c55e" if val_text in ("CHEAP", "undervalued") else ("#ef4444" if val_text in ("EXPENSIVE", "expensive") else "#94a3b8")

                # Format reasons as separate lines with color
                reason_html = _reason_bullets(s.reason)

                # Ownership
                holders = s.major_holders or "-"

                # Confidence color
                conf_pct = s.confidence * 100
                conf_color = "#22c55e" if conf_pct >= 70 else ("#f59e0b" if conf_pct >= 40 else "#ef4444")

                rows_html.append(f"""
                <tr>
                    <td style="white-space: nowrap; font-weight: 600;">{s.ticker}</td>
                    <td style="white-space: nowrap;">{s.company_name or '-'}</td>
                    <td style="color: {action_color}; font-weight: 700; text-align: center;">{s.action}</td>
                    <td style="color: {conf_color}; text-align: center; font-weight: 600;">{s.confidence:.0%}</td>
                    <td style="color: {val_color}; font-weight: 600; text-align: center;">{val_text}</td>
                    <td style="text-align: right; white-space: nowrap;">{_rp(s.price)}</td>
                    <td style="text-align: right;">{_fmt(s.pbv, '.2f', 'x') if s.pbv else '-'}</td>
                    <td style="text-align: right;">{_fmt(s.per, '.1f', 'x') if s.per else '-'}</td>
                    <td style="text-align: right;">{_rp(s.valuation_price)}</td>
                    <td style="text-align: right;">{_pct(s.roe)}</td>
                    <td style="text-align: right;">{_pct(s.dividend_yield)}</td>
                    <td style="font-size: 0.85em; max-width: 200px;">{holders}</td>
                    <td style="font-size: 0.85em; max-width: 400px;">{reason_html}</td>
                </tr>
                """)

            table_html = f"""
            <style>
                .sector-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 0.9em;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }}
                .sector-table th {{
                    background: #1e293b;
                    color: #e2e8f0;
                    padding: 10px 8px;
                    text-align: left;
                    position: sticky;
                    top: 0;
                    font-weight: 600;
                    white-space: nowrap;
                    border-bottom: 2px solid #334155;
                }}
                .sector-table td {{
                    padding: 8px;
                    border-bottom: 1px solid #334155;
                    vertical-align: top;
                }}
                .sector-table tr:hover {{
                    background: #1e293b40;
                }}
            </style>
            <div style="overflow-x: auto; max-height: 600px; overflow-y: auto;">
            <table class="sector-table">
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Company</th>
                        <th>Action</th>
                        <th>Conf</th>
                        <th>Valuation</th>
                        <th>Price</th>
                        <th>PBV</th>
                        <th>PER</th>
                        <th>Val. Price</th>
                        <th>ROE</th>
                        <th>Div Yield</th>
                        <th>Major Holders</th>
                        <th>Analysis Reasons</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
            </div>
            """
            st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.info(f"No stocks found in the '{selected_sector}' sector from the current scan. Try scanning more stocks.")


# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(120)
    st.cache_data.clear()
    st.rerun()


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

        **Valuation Price**
        - **Valuation Price** = PER × EPS (TTM)
        - If **Current Price < Valuation Price** → CHEAP
        - If **Current Price > Valuation Price** → EXPENSIVE

        **Confidence Score**
        - Technical signals (0–100%)
        - +25% if undervalued
        - ±10% for free float
        - +5% each: high ROE, ROIC, positive FCF, EPS growth
        """)

with st.expander("⚠️ Disclaimer"):
    st.warning(
        "This tool is for **educational and research purposes only**. "
        "Data sourced from yfinance, Google Finance, and public news RSS feeds. "
        "Past performance does not guarantee future results. "
        "Always do your own research (DYOR) before making any investment decisions. "
        "This is not financial advice."
    )
