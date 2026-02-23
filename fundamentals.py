"""
Fundamental and liquidity data via yfinance.
PBV, PER, free float, profitability, cash flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import yfinance as yf


@dataclass
class FundamentalSnapshot:
    pbv: Optional[float] = None  # Price / Book
    per: Optional[float] = None  # Price / Earnings
    revenue: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    eps: Optional[float] = None
    free_cashflow: Optional[float] = None
    debt_to_equity: Optional[float] = None
    free_float_ratio: Optional[float] = None  # 0-1


def _safe_ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den in (None, 0):
        return None
    return num / den


def fetch_fundamentals(ticker: str) -> FundamentalSnapshot:
    """
    Fetch key fundamentals from Yahoo Finance via yfinance.
    Fields are best-effort (some tickers may miss data).
    """
    t = yf.Ticker(ticker)
    try:
        info = t.get_info()
    except Exception:
        # Fall back to deprecated .info if necessary
        try:
            info = t.info
        except Exception:
            info = {}

    # yfinance can sometimes return None here; normalise to dict
    if not isinstance(info, dict):
        info = {}

    pbv = info.get("priceToBook")
    per = info.get("trailingPE") or info.get("forwardPE")
    revenue = info.get("totalRevenue")
    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")
    eps = info.get("trailingEps") or info.get("forwardEps")
    free_cashflow = info.get("freeCashflow") or info.get("operatingCashflow")

    total_debt = info.get("totalDebt")
    equity = info.get("totalStockholderEquity")
    debt_to_equity = _safe_ratio(total_debt, equity)

    float_shares = info.get("floatShares")
    shares_out = info.get("sharesOutstanding")
    free_float_ratio = _safe_ratio(float_shares, shares_out)

    return FundamentalSnapshot(
        pbv=pbv,
        per=per,
        revenue=revenue,
        roe=roe,
        roa=roa,
        eps=eps,
        free_cashflow=free_cashflow,
        debt_to_equity=debt_to_equity,
        free_float_ratio=free_float_ratio,
    )

