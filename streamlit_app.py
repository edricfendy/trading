"""
Indonesia Stock Trading AI - Streamlit App (Two-Tab Layout)
Optimizations:
- Longer cache TTLs (120s → 5min for bulk, 300s → 10min for single ticker)
- Progress bar during bulk scan
- Fundamentals loaded lazily only when ticker selected (not during bulk scan)
- Universe list cached 24 h (no re-scraping on every reload)
- st.cache_data keyed on refresh_token so manual refresh always works
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

from data_fetcher import (
    DataProviderError,
    DataRateLimitError,
    fetch_stock_data,
    fetch_realtime_prices,
    update_last_candle_with_realtime,
    clear_cache,
)
from universe import get_universe
from sentiment import fetch_ticker_sentiment, fetch_sector_sentiment, sentiment_summary

st.set_page_config(
    page_title="Indonesia Stock Trading AI",
    page_icon="📈",
    layout="wide",
)

try:
    from analyzer import (
        get_all_signals,
        get_all_signals_bulk,
        analyze_buy_sell_timing,
        screen_rebound_candidates,
        TimingSignal,
        ReboundCandidate,
    )
except Exception as exc:
    st.error("Failed to import analysis modules.")
    st.exception(exc)
    st.stop()

st.title("📈 Indonesia Stock Trading AI")
st.caption(
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
    "Stochastic RSI | MACD | Bandar Volume | Smart Money | Fundamentals | Sentiment"
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    ta_interval = st.selectbox(
        "TA Interval",
        ["1d", "1h", "30m", "15m"],
        index=0,
        help="Intraday intervals use ~5d of data; daily uses ~3mo.",
    )
    batch_limit = st.slider(
        "Max stocks to scan",
        min_value=10, max_value=500,
        value=50, step=10,
        help="Higher = more comprehensive but slower.",
    )
    min_rebound = st.slider("Rebound Min Score", 20, 80, 40)

    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 5 min)", value=False)
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

    if "refresh_token" not in st.session_state:
        st.session_state["refresh_token"] = 0.0
    if "force_refresh" not in st.session_state:
        st.session_state["force_refresh"] = False

    if st.button("🔄 Refresh Analysis (bypass cache)"):
        st.session_state["refresh_token"] = time.time()
        st.session_state["force_refresh"] = True
        st.cache_data.clear()
        clear_cache()
        st.rerun()


# ─── Universe ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)   # 24 h — universe barely changes
def load_all_tickers() -> list[str]:
    return get_universe(all_idx=True)

all_tickers = load_all_tickers()
tickers = all_tickers[:batch_limit]
st.sidebar.write(f"Scanning {len(tickers)} of {len(all_tickers)} IDX stocks.")


# ─── Cached analysis helpers ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)      # 5-min cache
def run_signals(ticker_tuple, interval, refresh_token, force_refresh):
    """Bulk TA-only scan — no fundamentals (fast)."""
    return get_all_signals_bulk(
        tickers=list(ticker_tuple),
        interval=interval,
        period=None,
        bypass_cache=force_refresh,
        return_data=True,
    )


@st.cache_data(ttl=600, show_spinner=False)      # 10-min cache
def run_single_ticker_analysis(ticker, interval, refresh_token):
    """Full analysis with fundamentals for a single selected ticker.
    Note: ticker is part of the cache key, so switching tickers always runs fresh.
    """
    df = fetch_stock_data(ticker, period=None, interval=interval)
    if df is None or df.empty:
        return None, None, ""
    rt_prices, rt_source = {}, ""
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


# ─── Run bulk scan ────────────────────────────────────────────────────────────
refresh_token = st.session_state.get("refresh_token", 0.0)
force_refresh  = st.session_state.get("force_refresh", False)

progress_placeholder = st.empty()
try:
    with progress_placeholder.container():
        prog = st.progress(0, text=f"⏳ Bulk downloading {len(tickers)} stocks…")
        signals, rt_prices, rt_source_label, bulk_data = run_signals(
            tuple(tickers), ta_interval, refresh_token, force_refresh
        )
        prog.progress(100, text="✅ Analysis complete")
        time.sleep(0.3)

    progress_placeholder.empty()

    buy_signals  = [s for s in signals if s.action == "BUY"]
    sell_signals = [s for s in signals if s.action == "SELL"]
    hold_signals = [s for s in signals if s.action == "HOLD"]
    rebounds = screen_rebound_candidates(
        min_score=min_rebound, rt_prices=rt_prices, data=bulk_data
    )
    if force_refresh:
        st.session_state["force_refresh"] = False

except DataRateLimitError:
    st.error("⚠️ Data provider rate limit. Reduce 'Max stocks to scan' or wait a minute.")
    st.stop()
except DataProviderError as exc:
    st.error(f"Data provider error: {exc}")
    st.stop()


# ─── Formatters ───────────────────────────────────────────────────────────────
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

def _color_valuation(label: str) -> str:
    if label == "CHEAP":
        return '<span style="color:#22c55e;font-weight:700;">✅ CHEAP (Below Fair Value)</span>'
    elif label == "EXPENSIVE":
        return '<span style="color:#ef4444;font-weight:700;">⚠️ EXPENSIVE (Above Fair Value)</span>'
    return '<span style="color:#94a3b8;">N/A</span>'

def _reason_bullets(reason_str: str) -> str:
    if not reason_str:
        return "-"
    parts = [r.strip() for r in reason_str.split("|") if r.strip()]
    lines = []
    pos_kw = ["oversold","bullish","accumulating","buying","undervalued",
              "positive","inflow","high capital","strong","good",
              "healthy","low debt","growth","bounce","golden cross"]
    neg_kw = ["overbought","bearish","distributing","selling","overvalued",
              "negative","outflow","high leverage","concern","risk",
              "dead cross","burn rate","expensive","rich valuation"]
    for p in parts:
        lower = p.lower()
        if any(k in lower for k in pos_kw) and not any(k in lower for k in neg_kw):
            lines.append(f'<span style="color:#22c55e;">• {p}</span>')
        elif any(k in lower for k in neg_kw) and not any(k in lower for k in pos_kw):
            lines.append(f'<span style="color:#ef4444;">• {p}</span>')
        else:
            lines.append(f"• {p}")
    return "<br>".join(lines)


# ─── TAB LAYOUT ───────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Ticker Search & Analysis", "🗂️ Sector Overview"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🔍 Ticker Search & Deep Analysis")
    st.caption("💡 Fundamentals (PER, ROE, etc.) are fetched on-demand when you select a ticker. The market overview uses TA-only for speed.")

    ticker_options = sorted(set(tickers))
    ticker_name_map = {s.ticker: s.company_name for s in signals if s.company_name}
    display_options = [
        f"{t} — {ticker_name_map[t]}" if t in ticker_name_map else t
        for t in ticker_options
    ]

    col_search, col_extra = st.columns([3, 1])
    with col_search:
        selected_display = st.selectbox(
            "Search Ticker (Emiten)",
            options=display_options,
            index=None,
            placeholder="Type or select a ticker (e.g. BBCA.JK)…",
        )
    with col_extra:
        manual_ticker = st.text_input("Or enter ticker manually", placeholder="e.g. BBCA.JK")

    selected_ticker = None
    if selected_display:
        selected_ticker = selected_display.split(" — ")[0].strip()
    elif manual_ticker:
        t = manual_ticker.strip().upper()
        if not t.endswith(".JK"):
            t += ".JK"
        selected_ticker = t

    if selected_ticker:
        # Prefer pre-scanned TA result but always re-run for fundamentals
        with st.spinner(f"⏳ Loading full analysis for {selected_ticker} (including fundamentals)…"):
            try:
                sig, rt_price, rt_src = run_single_ticker_analysis(
                    selected_ticker,
                    ta_interval,
                    st.session_state.get("refresh_token", 0.0),
                )
            except Exception as e:
                st.error(f"Error: {e}")
                sig = None

        if sig and sig.price > 0:
            st.markdown("---")
            name_label = f" — {sig.company_name}" if sig.company_name else ""
            st.subheader(f"📊 {sig.ticker}{name_label}")

            # Valuation panel
            st.markdown("### 💰 Valuation Analysis")
            v = st.columns(5)
            bm_per_val = getattr(sig, "benchmark_per", None)
            v[0].metric("Current Price",    _rp(sig.price))
            v[1].metric("Trailing PER",     _fmt(sig.per, ".1f", "x") if sig.per else "-",
                        help="Stock's own trailing P/E (price ÷ EPS). NOT used in fair value.")
            v[2].metric("EPS (TTM)",        _fmt(sig.eps, ",.0f", " Rp") if sig.eps else "-")
            v[3].metric("Fair Value (BM PER×EPS)", _rp(sig.valuation_price),
                        help=f"Benchmark PER {bm_per_val:.1f}x × EPS(TTM) = Fair Value" if bm_per_val else "N/A")
            v[4].metric("PBV",              _fmt(sig.pbv, ".2f", "x") if sig.pbv else "-")

            if sig.price_vs_valuation:
                st.markdown(_color_valuation(sig.price_vs_valuation), unsafe_allow_html=True)
                if sig.valuation_price:
                    diff = sig.price - sig.valuation_price
                    diff_pct = (diff / sig.valuation_price) * 100
                    direction = "above" if diff > 0 else "below"
                    # Show the benchmark PER used and the formula
                    bm = getattr(sig, "benchmark_per", None)
                    method = getattr(sig, "valuation_method", "")
                    formula_note = f" | Formula: {method}" if method else ""
                    st.caption(
                        f"Current price is Rp {abs(diff):,.0f} ({abs(diff_pct):.1f}%) {direction} "
                        f"the fair value estimate{formula_note}"
                    )
                    if bm:
                        st.info(
                            f"📐 **Valuation Method**: Benchmark PER **{bm:.1f}×** (IDX {sig.sector or 'market'} "
                            f"sector average) × EPS(TTM) **Rp {sig.eps:,.0f}** = "
                            f"Fair Value **{_rp(sig.valuation_price)}**  \n"
                            f"*Note: Benchmark PER is the sector fair multiple, NOT the stock's own trailing PER "
                            f"({_fmt(sig.per, '.1f', 'x') if sig.per else 'N/A'}). "
                            f"Using the stock's own PER × EPS always equals the current price.*"
                        )
            else:
                st.caption("Valuation price unavailable (missing PER or EPS data)")

            # Signal
            st.markdown("### 🎯 Signal")
            a = st.columns(4)
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪"}.get(sig.action, "")
            a[0].metric(f"{emoji} Action",  sig.action)
            a[1].metric("Confidence",       f"{sig.confidence:.0%}")
            a[2].metric("Horizon",          sig.horizon.title())
            a[3].metric("Valuation",        sig.valuation_label.title())

            # Technical
            st.markdown("### 📈 Technical Analysis")
            t1 = st.columns(6)
            t1[0].metric("StochRSI", _fmt(sig.stoch_rsi_k, ".1f"))
            t1[1].metric("RSI",      _fmt(sig.rsi, ".1f"))
            t1[2].metric("SMI",      _fmt(sig.smi, ".2f"))
            macd_lbl = (sig.macd_trend or "-") + (f" + {sig.macd_divergence} div" if sig.macd_divergence else "")
            t1[3].metric("MACD",         macd_lbl)
            t1[4].metric("Bandar Score", _fmt(sig.bandar_score, ".1f"))
            t1[5].metric("OBV Flow",     sig.obv_momentum or "-")

            t2 = st.columns(6)
            t2[0].metric("Support",     _rp(sig.support_20))
            t2[1].metric("Resistance",  _rp(sig.resistance_20))
            t2[2].metric("VWAP",        _rp(sig.vwap))
            t2[3].metric("ATR",         _rp(sig.atr))
            t2[4].metric("Take Profit", _rp(sig.take_profit))
            t2[5].metric("Stop Loss",   _rp(sig.stop_loss))

            # Fundamentals
            st.markdown("### 📋 Fundamentals")
            f1 = st.columns(6)
            f1[0].metric("ROE",        _pct(sig.roe))
            f1[1].metric("ROA",        _pct(sig.roa))
            f1[2].metric("ROIC",       _pct(sig.roic))
            f1[3].metric("EPS",        _fmt(sig.eps, ",.0f", " Rp") if sig.eps else "-")
            f1[4].metric("EPS Growth", _pct(sig.eps_growth))
            f1[5].metric("D/E Ratio",  _fmt(sig.debt_to_equity, ".1f", "x") if sig.debt_to_equity else "-")

            f2 = st.columns(6)
            f2[0].metric("FCF",          _billions(sig.free_cashflow))
            f2[1].metric("Op Cash Flow", _billions(sig.operating_cashflow))
            f2[2].metric("Revenue",      _billions(sig.revenue))
            f2[3].metric("Current Ratio",_fmt(sig.current_ratio, ".1f", "x") if sig.current_ratio else "-")
            f2[4].metric("Market Cap",   _billions(sig.market_cap))
            f2[5].metric("Div Yield",    _pct(sig.dividend_yield))

            ff_pct = f"{sig.free_float_ratio*100:.1f}%" if sig.free_float_ratio else "-"
            if sig.free_float_ratio and sig.free_float_ratio < 0.15:
                st.warning(f"⚠️ LOW Free Float: {ff_pct} — illiquid, high manipulation risk")
            elif sig.free_float_ratio and sig.free_float_ratio > 0.40:
                st.success(f"✅ HIGH Free Float: {ff_pct} — good market liquidity")
            else:
                st.info(f"Free Float: {ff_pct}")

            if sig.major_holders:
                st.markdown("### 👥 Major Holders / Ownership")
                st.markdown(f"**{sig.major_holders}**")

            st.markdown("### 📝 Analysis Reasons")
            st.markdown(_reason_bullets(sig.reason), unsafe_allow_html=True)

            # Sentiment
            st.markdown("### 📰 Sentiment & Catalyst Analysis")
            with st.spinner("Fetching news…"):
                sent_items = run_sentiment_ticker(selected_ticker, sig.company_name)
            if sent_items:
                summary = sentiment_summary(sent_items)
                sc = st.columns(4)
                sc[0].metric("Overall",      summary["overall"].upper())
                sc[1].metric("🟢 Positive",  summary["positive"])
                sc[2].metric("🔴 Negative",  summary["negative"])
                sc[3].metric("⚪ Neutral",   summary["neutral"])
                st.markdown("#### Recent News & Catalysts")
                for item in sent_items:
                    emoji = "🟢" if item.sentiment == "positive" else ("🔴" if item.sentiment == "negative" else "⚪")
                    cat_emoji = {"company_action":"🏢","sector_demand":"📊","commodity":"🪨",
                                 "ownership_change":"👥","macro":"🌍","regulatory":"📜"}.get(item.category, "📰")
                    color = "#22c55e" if item.sentiment == "positive" else ("#ef4444" if item.sentiment == "negative" else "#94a3b8")
                    st.markdown(
                        f'{emoji} {cat_emoji} <span style="color:{color};">{item.headline}</span> '
                        f'<span style="color:#64748b;font-size:0.85em;">({item.date}) [{item.category}]</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No recent news found.")
        elif selected_ticker:
            st.warning(f"No data available for {selected_ticker}.")

    else:
        # ── Market overview ────────────────────────────────────────────────
        st.markdown("### 📊 Market Overview")
        if rt_source_label:
            st.caption(f"📡 Live prices: {rt_source_label} | Bulk download mode")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY",  len(buy_signals))
        c2.metric("🔴 SELL", len(sell_signals))
        c3.metric("⚪ HOLD", len(hold_signals))
        c4.metric("🔁 Rebound", len(rebounds))

        def _full_signal_row(s: TimingSignal) -> dict:
            horizon_emoji = {"long-term":"🏦","balanced":"⚖️","short-term":"⚡","speculative":"🎯","neutral":"—"}.get(s.horizon,"—")
            macd_lbl = (s.macd_trend or "-") + (f" + {s.macd_divergence} div" if s.macd_divergence else "")
            ff_pct = f"{s.free_float_ratio*100:.1f}%" if s.free_float_ratio else "-"
            ff_flag = "⚠️" if s.free_float_ratio and s.free_float_ratio < 0.15 else ("✅" if s.free_float_ratio and s.free_float_ratio > 0.40 else "")
            val_label = ""
            if s.price_vs_valuation:
                val_label = f"{'🟢' if s.price_vs_valuation == 'CHEAP' else '🔴'} {s.price_vs_valuation}"
            else:
                val_label = s.valuation_label or "-"
            return {
                "Ticker":  s.ticker,
                "Action":  s.action,
                "Conf":    f"{s.confidence:.0%}",
                "Horizon": f"{horizon_emoji} {s.horizon}",
                "Valuation": val_label,
                "Sector":  s.sector or "-",
                "StochRSI":_fmt(s.stoch_rsi_k, ".1f"),
                "RSI":     _fmt(s.rsi, ".1f"),
                "SMI":     _fmt(s.smi, ".2f"),
                "MACD":    macd_lbl,
                "Bandar":  _fmt(s.bandar_score, ".1f"),
                "OBV":     s.obv_momentum or "-",
                "Price":   _rp(s.price),
                "VWAP":    _rp(s.vwap),
                "Support": _rp(s.support_20),
                "Resist":  _rp(s.resistance_20),
                "ATR":     _rp(s.atr),
                "TP":      _rp(s.take_profit),
                "SL":      _rp(s.stop_loss),
                "PBV":     _fmt(s.pbv, ".2f", "x"),
                "PER":     _fmt(s.per, ".1f", "x"),
                "ROE":     _pct(s.roe),
                "EPS":     _fmt(s.eps, ",.0f", " Rp") if s.eps else "-",
                f"FF{ff_flag}": ff_pct,
                "MktCap":  _billions(s.market_cap),
                "DivYld":  _pct(s.dividend_yield),
                "Reason":  s.reason[:200] + "…" if len(s.reason) > 200 else s.reason,
            }

        if buy_signals:
            st.header("🟢 Buy Opportunities")
            with st.expander("🏦 Long-Term (Undervalued)", expanded=True):
                lt = [s for s in buy_signals if s.horizon == "long-term"]
                if lt:
                    st.dataframe(pd.DataFrame([_full_signal_row(s) for s in lt]),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No long-term undervalued buys at this time.")
            with st.expander("⚡ Short-Term / Momentum", expanded=True):
                st_ = [s for s in buy_signals if s.horizon != "long-term"]
                if st_:
                    st.dataframe(pd.DataFrame([_full_signal_row(s) for s in st_]),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No short-term momentum buys.")

        if sell_signals:
            st.header("🔴 Sell Signals")
            st.dataframe(pd.DataFrame([_full_signal_row(s) for s in sell_signals]),
                         use_container_width=True, hide_index=True)

        if hold_signals:
            with st.expander(f"⚪ Hold / No Strong Signal ({len(hold_signals)} stocks)", expanded=False):
                st.dataframe(pd.DataFrame([_full_signal_row(s) for s in hold_signals[:20]]),
                             use_container_width=True, hide_index=True)

        st.header("🔁 Potential Rebound Candidates")
        st.caption("Oversold Stoch RSI + Smart Money + Bandar accumulation")
        if rebounds:
            rb_data = [{
                "Ticker":       c.ticker,
                "Rebound Score":f"{c.rebound_score:.1f}",
                "Stoch RSI":    f"{c.stoch_rsi_k:.1f}",
                "SMI Trend":    c.smi_trend,
                "Bandar":       c.bandar_trend,
                "5d Change":    f"{c.recent_change_pct:+.1f}%",
                "Price (Rp)":   f"{c.price:,.0f}",
                "Reasons":      " | ".join(c.reasons[:3]),
            } for c in rebounds[:20]]
            st.dataframe(pd.DataFrame(rb_data), use_container_width=True, hide_index=True)
        else:
            st.info("No strong rebound candidates. Try lowering the min score in the sidebar.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SECTOR OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🗂️ Sector Overview")

    all_sectors = sorted(set(s.sector for s in signals if s.sector))
    selected_sector = None
    if all_sectors:
        selected_sector = st.selectbox(
            "Select Sector",
            options=all_sectors,
            index=0,
            placeholder="Choose a sector...",
        )
    else:
        st.warning(
            "Sector data is not available in this fast bulk scan. "
            "Current mode runs TA-only analysis and does not fetch fundamentals "
            "(sector/industry)."
        )
        st.caption(
            "Tip: Use ticker deep-dive in Tab 1 for full fundamentals, or switch "
            "to a slower full-fundamental scan mode if you add one."
        )

    if selected_sector:
        sector_signals = [s for s in signals if s.sector == selected_sector]

        if sector_signals:
            s_buy  = [s for s in sector_signals if s.action == "BUY"]
            s_sell = [s for s in sector_signals if s.action == "SELL"]
            s_hold = [s for s in sector_signals if s.action == "HOLD"]

            m = st.columns(4)
            m[0].metric(f"📊 Total",  len(sector_signals))
            m[1].metric("🟢 BUY",    len(s_buy))
            m[2].metric("🔴 SELL",   len(s_sell))
            m[3].metric("⚪ HOLD",   len(s_hold))

            with st.expander("📰 Sector Sentiment", expanded=True):
                with st.spinner("Fetching sector news…"):
                    sector_sent = run_sentiment_sector(selected_sector)
                if sector_sent:
                    ss = sentiment_summary(sector_sent)
                    sc = st.columns(4)
                    sc[0].metric("Overall",     ss["overall"].upper())
                    sc[1].metric("🟢 Positive", ss["positive"])
                    sc[2].metric("🔴 Negative", ss["negative"])
                    sc[3].metric("⚪ Neutral",  ss["neutral"])
                    for item in sector_sent:
                        emoji = "🟢" if item.sentiment=="positive" else ("🔴" if item.sentiment=="negative" else "⚪")
                        color = "#22c55e" if item.sentiment=="positive" else ("#ef4444" if item.sentiment=="negative" else "#94a3b8")
                        st.markdown(
                            f'{emoji} <span style="color:{color};">{item.headline}</span> '
                            f'<span style="color:#64748b;font-size:0.85em;">({item.date}) [{item.category}]</span>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No recent sector news found.")

            st.markdown("### 📋 Stocks in this Sector")
            rows_html = []
            for s in sorted(sector_signals, key=lambda x: (-x.confidence, x.ticker)):
                action_color = "#22c55e" if s.action=="BUY" else ("#ef4444" if s.action=="SELL" else "#94a3b8")
                val_text  = s.price_vs_valuation or s.valuation_label or "-"
                val_color = "#22c55e" if val_text in ("CHEAP","undervalued") else ("#ef4444" if val_text in ("EXPENSIVE","expensive") else "#94a3b8")
                conf_pct  = s.confidence * 100
                conf_color = "#22c55e" if conf_pct >= 70 else ("#f59e0b" if conf_pct >= 40 else "#ef4444")
                rows_html.append(f"""
                <tr>
                    <td style="white-space:nowrap;font-weight:600;">{s.ticker}</td>
                    <td style="white-space:nowrap;">{s.company_name or '-'}</td>
                    <td style="color:{action_color};font-weight:700;text-align:center;">{s.action}</td>
                    <td style="color:{conf_color};text-align:center;font-weight:600;">{s.confidence:.0%}</td>
                    <td style="color:{val_color};font-weight:600;text-align:center;">{val_text}</td>
                    <td style="text-align:right;white-space:nowrap;">{_rp(s.price)}</td>
                    <td style="text-align:right;">{_fmt(s.pbv,'.2f','x') if s.pbv else '-'}</td>
                    <td style="text-align:right;">{_fmt(s.per,'.1f','x') if s.per else '-'}</td>
                    <td style="text-align:right;">{_rp(s.valuation_price)}</td>
                    <td style="text-align:right;">{_pct(s.roe)}</td>
                    <td style="text-align:right;">{_pct(s.dividend_yield)}</td>
                    <td style="font-size:0.85em;max-width:200px;">{s.major_holders or '-'}</td>
                    <td style="font-size:0.85em;max-width:400px;">{_reason_bullets(s.reason)}</td>
                </tr>""")

            st.markdown(f"""
            <style>
            .sector-table{{width:100%;border-collapse:collapse;font-size:0.9em;font-family:-apple-system,sans-serif;}}
            .sector-table th{{background:#1e293b;color:#e2e8f0;padding:10px 8px;text-align:left;
                              position:sticky;top:0;font-weight:600;white-space:nowrap;border-bottom:2px solid #334155;}}
            .sector-table td{{padding:8px;border-bottom:1px solid #334155;vertical-align:top;}}
            .sector-table tr:hover{{background:#1e293b40;}}
            </style>
            <div style="overflow-x:auto;max-height:600px;overflow-y:auto;">
            <table class="sector-table"><thead><tr>
                <th>Ticker</th><th>Company</th><th>Action</th><th>Conf</th>
                <th>Valuation</th><th>Price</th><th>PBV</th><th>PER</th>
                <th>Val. Price</th><th>ROE</th><th>Div Yield</th>
                <th>Major Holders</th><th>Analysis Reasons</th>
            </tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>
            """, unsafe_allow_html=True)
        else:
            st.info(f"No stocks found in '{selected_sector}' from the current scan. Try scanning more stocks.")


# ─── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(300)   # 5 minutes
    st.cache_data.clear()
    clear_cache()
    st.rerun()


# ─── Guide ─────────────────────────────────────────────────────────────────────
with st.expander("📖 Indicator & Signal Guide"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Technical Signals**
        - **Stoch RSI < 20** → oversold → potential buy
        - **Stoch RSI > 80** → overbought → potential sell
        - **RSI < 30** → deeply oversold
        - **MACD golden cross** → bullish momentum
        - **MACD dead cross** → bearish momentum
        - **Bandar > 3** → big player buying
        - **Bandar < -3** → big player selling
        - **OBV INFLOW** → institutional accumulation proxy
        - **BB BELOW** → price below Bollinger lower band
        """)
    with c2:
        st.markdown("""
        **Fundamental Signals**
        - **PBV < 1.5 + PER < 15** → undervalued
        - **ROE > 15%** → high capital efficiency
        - **D/E > 2x** → high leverage risk
        - **Current Ratio < 1** → liquidity risk
        - **Positive FCF** → healthy cash
        - **Free Float < 15%** → ⚠️ manipulation risk

        **Valuation Price = Benchmark PER × EPS(TTM)**
        - **Benchmark PER** = IDX sector-average fair multiple (e.g. 12× for banks, 22× for tech)
        - **NOT** the stock's own trailing PER (that gives a tautology: price = price)
        - **CHEAP** = current price < fair value estimate
        - **EXPENSIVE** = current price > fair value estimate

        **Confidence Score**
        - Technical signals (0–100%)
        - +25% undervalued
        - ±10% free float
        - +5% each: ROE, ROIC, FCF, EPS growth
        """)

with st.expander("⚠️ Disclaimer"):
    st.warning(
        "For educational/research use only. Past performance does not guarantee future results. "
        "Always DYOR before investing. This is not financial advice."
    )
