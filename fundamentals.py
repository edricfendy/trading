"""
Fundamental and liquidity data via yfinance.
Optimizations vs original:
- In-process LRU cache (default 30-min TTL) — yfinance .info is ~1-2 s per ticker
- Parallel batch fetching via ThreadPoolExecutor
- Graceful degradation: returns empty snapshot on any error
"""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

try:
    import yfinance as yf
    _YF_OK = True
except Exception:
    yf = None  # type: ignore[assignment]
    _YF_OK = False

# ─── Cache ─────────────────────────────────────────────────────────────────────
_FUND_CACHE: dict[str, tuple[float, "FundamentalSnapshot"]] = {}
_FUND_LOCK = threading.Lock()
FUND_CACHE_TTL_SEC = int(__import__("os").getenv("FUND_CACHE_TTL_SEC", str(30 * 60)))  # 30 min


@dataclass
class FundamentalSnapshot:
    # Valuation
    pbv: Optional[float] = None
    per: Optional[float] = None
    per_forward: Optional[float] = None
    peg_ratio: Optional[float] = None
    book_value_per_share: Optional[float] = None
    valuation_price: Optional[float] = None
    # Profitability
    revenue: Optional[float] = None
    gross_margins: Optional[float] = None
    operating_margins: Optional[float] = None
    profit_margins: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    roic: Optional[float] = None
    eps: Optional[float] = None
    eps_forward: Optional[float] = None
    eps_growth: Optional[float] = None
    # Cash Flow
    free_cashflow: Optional[float] = None
    operating_cashflow: Optional[float] = None
    capex: Optional[float] = None
    # Solvency
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    # Liquidity
    free_float_ratio: Optional[float] = None
    avg_volume_10d: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    # Dividend
    dividend_yield: Optional[float] = None
    # Classification
    sector: Optional[str] = None
    industry: Optional[str] = None
    # Identity
    company_name: Optional[str] = None
    major_holders: Optional[str] = None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b in (None, 0):
        return None
    return a / b


def _fetch_ownership(t) -> str:
    try:
        holders = t.institutional_holders
        if holders is not None and not holders.empty:
            names = holders["Holder"].head(5).tolist()
            return ", ".join(str(n) for n in names if n)
    except Exception:
        pass
    try:
        mh = t.major_holders
        if mh is not None and not mh.empty:
            return ", ".join(str(n) for n in mh.iloc[:, 1].tolist() if n)
    except Exception:
        pass
    return ""


def _build_snapshot(info: dict, t) -> FundamentalSnapshot:
    """Construct a FundamentalSnapshot from yfinance info dict."""
    per = info.get("trailingPE")
    eps = info.get("trailingEps")
    valuation_price: Optional[float] = None
    if per is not None and eps is not None and eps > 0:
        valuation_price = per * eps

    ebit = info.get("ebit")
    tax_rate = info.get("effectiveTaxRate") or 0.22
    invested_capital = (info.get("totalDebt") or 0) + (info.get("totalStockholderEquity") or 0)
    roic: Optional[float] = None
    if ebit and invested_capital:
        roic = ebit * (1 - tax_rate) / invested_capital

    eps_fwd = info.get("forwardEps")
    eps_growth: Optional[float] = None
    if eps and eps_fwd and eps != 0:
        eps_growth = (eps_fwd - eps) / abs(eps)

    fcf = info.get("freeCashflow")
    opcf = info.get("operatingCashflow")
    capex = (opcf - fcf) if fcf is not None and opcf is not None else None

    total_debt = info.get("totalDebt")
    equity = info.get("totalStockholderEquity")

    float_shares = info.get("floatShares")
    shares_out = info.get("sharesOutstanding")

    return FundamentalSnapshot(
        pbv=info.get("priceToBook"),
        per=per,
        per_forward=info.get("forwardPE"),
        peg_ratio=info.get("pegRatio") or info.get("trailingPegRatio"),
        book_value_per_share=info.get("bookValue"),
        valuation_price=valuation_price,
        revenue=info.get("totalRevenue"),
        gross_margins=info.get("grossMargins"),
        operating_margins=info.get("operatingMargins"),
        profit_margins=info.get("profitMargins"),
        roe=info.get("returnOnEquity"),
        roa=info.get("returnOnAssets"),
        roic=roic,
        eps=eps,
        eps_forward=eps_fwd,
        eps_growth=eps_growth,
        free_cashflow=fcf,
        operating_cashflow=opcf,
        capex=capex,
        total_debt=total_debt,
        total_cash=info.get("totalCash"),
        debt_to_equity=_safe_div(total_debt, equity),
        current_ratio=info.get("currentRatio"),
        quick_ratio=info.get("quickRatio"),
        free_float_ratio=_safe_div(float_shares, shares_out),
        avg_volume_10d=info.get("averageVolume10days") or info.get("averageVolume"),
        market_cap=info.get("marketCap"),
        enterprise_value=info.get("enterpriseValue"),
        dividend_yield=info.get("dividendYield"),
        sector=info.get("sector") or info.get("sectorDisp"),
        industry=info.get("industry") or info.get("industryDisp"),
        company_name=info.get("longName") or info.get("shortName") or "",
        major_holders=_fetch_ownership(t),
    )


def fetch_fundamentals(ticker: str) -> FundamentalSnapshot:
    """
    Fetch fundamentals for a single ticker, with in-process cache.
    Safe: always returns a FundamentalSnapshot (never raises).
    """
    # Cache check
    with _FUND_LOCK:
        entry = _FUND_CACHE.get(ticker)
        if entry is not None:
            ts, snap = entry
            if time.time() - ts < FUND_CACHE_TTL_SEC:
                return snap

    if not _YF_OK:
        return FundamentalSnapshot()

    try:
        t = yf.Ticker(ticker)
        try:
            info = t.get_info()
        except Exception:
            info = t.info
        if not isinstance(info, dict):
            info = {}
        snap = _build_snapshot(info, t)
    except Exception:
        snap = FundamentalSnapshot()

    with _FUND_LOCK:
        _FUND_CACHE[ticker] = (time.time(), snap)

    return snap


def fetch_fundamentals_batch(
    tickers: list[str],
    max_workers: int = 8,
) -> dict[str, FundamentalSnapshot]:
    """
    Parallel fundamentals fetch.  Uses cache so repeated calls are free.
    """
    # Separate cached vs uncached
    results: dict[str, FundamentalSnapshot] = {}
    to_fetch: list[str] = []

    with _FUND_LOCK:
        for ticker in tickers:
            entry = _FUND_CACHE.get(ticker)
            if entry is not None and (time.time() - entry[0]) < FUND_CACHE_TTL_SEC:
                results[ticker] = entry[1]
            else:
                to_fetch.append(ticker)

    if not to_fetch or not _YF_OK:
        for ticker in to_fetch:
            results[ticker] = FundamentalSnapshot()
        return results

    def _fetch(ticker: str) -> tuple[str, FundamentalSnapshot]:
        return ticker, fetch_fundamentals(ticker)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ticker, snap in ex.map(_fetch, to_fetch):
            results[ticker] = snap

    return results
