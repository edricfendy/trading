"""
Indonesia Stock Trading AI - Main Entry Point
Real-time tracking with Stochastic RSI and Smart Money accumulation analysis
"""
from __future__ import annotations
import sys
import io
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Fix Windows cp1252 encoding for console output
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except (AttributeError, io.UnsupportedOperation):
        pass

from tabulate import tabulate
from datetime import datetime
from analyzer import get_all_signals, screen_rebound_candidates, TimingSignal, ReboundCandidate
from data_fetcher import DataProviderError, DataRateLimitError


def format_signals(signals: list[TimingSignal]) -> str:
    """Format timing signals as table"""
    rows = []
    for s in signals:
        action_emoji = {"BUY": "[+]", "SELL": "[-]", "HOLD": "[.]"}.get(s.action, "[.]")
        stoch = f"{s.stoch_rsi_k:.1f}" if s.stoch_rsi_k is not None else "-"
        smi = f"{s.smi:.2f}" if s.smi is not None else "-"
        rows.append([
            f"{action_emoji} {s.ticker}",
            s.action,
            f"{s.confidence:.0%}",
            stoch,
            smi,
            f"Rp {s.price:,.0f}",
            s.reason[:50] + "..." if len(s.reason) > 50 else s.reason,
        ])
    return tabulate(
        rows,
        headers=["Ticker", "Action", "Confidence", "Stoch RSI", "SMI", "Price", "Reason"],
        tablefmt="simple",
    )


def format_rebounds(candidates: list[ReboundCandidate]) -> str:
    """Format rebound candidates as table"""
    rows = []
    for c in candidates[:15]:  # Top 15
        rows.append([
            c.ticker,
            f"{c.rebound_score:.1f}",
            f"{c.stoch_rsi_k:.1f}",
            c.smi_trend,
            f"{c.recent_change_pct:+.1f}%",
            f"Rp {c.price:,.0f}",
            "; ".join(c.reasons[:2]),
        ])
    return tabulate(
        rows,
        headers=["Ticker", "Rebound Score", "Stoch RSI", "SMI Trend", "5d Change", "Price", "Reasons"],
        tablefmt="simple",
    )


def run_analysis():
    """Run full analysis and print results"""
    print("=" * 70)
    print("INDONESIA STOCK TRADING AI")
    print(f"   Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print("\n>> BUY/SELL TIMING SIGNALS (Stochastic RSI + Smart Money)")
    print("-" * 70)
    signals, rt_prices, rt_source_label = get_all_signals()
    buy_signals = [s for s in signals if s.action == "BUY"]
    sell_signals = [s for s in signals if s.action == "SELL"]
    hold_signals = [s for s in signals if s.action == "HOLD"]

    if rt_source_label:
        print(f"\n   📡 Live prices: {rt_source_label}")

    if buy_signals:
        print("\n[BUY] OPPORTUNITIES:")
        print(format_signals(buy_signals))
    if sell_signals:
        print("\n[SELL] SIGNALS:")
        print(format_signals(sell_signals))
    if hold_signals:
        print("\n[HOLD] (no strong signal):")
        print(format_signals(hold_signals[:10]))  # First 10 holds

    print("\n" + "=" * 70)
    print(">> POTENTIAL REBOUND CANDIDATES")
    print("   (Oversold Stoch RSI + Smart Money accumulation)")
    print("-" * 70)
    rebounds = screen_rebound_candidates(min_score=40, rt_prices=rt_prices)
    if rebounds:
        print(format_rebounds(rebounds))
    else:
        print("No strong rebound candidates at this time. Try lowering min_score.")

    print("\n" + "=" * 70)
    print("Tips:")
    print("   - Stoch RSI < 20 = oversold (potential buy)")
    print("   - Stoch RSI > 80 = overbought (potential sell)")
    print("   - SMI > 0 = smart money accumulating")
    print("   - Run regularly for fresh updates")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_analysis()
    except DataRateLimitError:
        print("Data provider rate limit reached. Reduce tickers or retry in a few minutes.")
    except DataProviderError as exc:
        print(f"Data provider error: {exc}")
