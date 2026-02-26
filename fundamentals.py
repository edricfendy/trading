"""
Fundamental and liquidity data via yfinance.
Covers: PBV, PER, ROE, ROA, ROIC, EPS, Free Cashflow,
Solvency (D/E, Current Ratio), Revenue, Free Float.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf


@dataclass
class FundamentalSnapshot:
    # Valuation
    pbv: Optional[float] = None            # Price / Book Value
    per: Optional[float] = None            # Price / Earnings (trailing)
    per_forward: Optional[float] = None    # Forward PER
    peg_ratio: Optional[float] = None      # PEG ratio
    book_value_per_share: Optional[float] = None  # Book value per share
    valuation_price: Optional[float] = None       # PBV × BV/share (fair value estimate)

    # Profitability
    revenue: Optional[float] = None
    gross_margins: Optional[float] = None  # 0-1
    operating_margins: Optional[float] = None
    profit_margins: Optional[float] = None
    roe: Optional[float] = None            # Return on Equity 0-1
    roa: Optional[float] = None            # Return on Assets 0-1
    roic: Optional[float] = None           # Return on Invested Capital (computed)
    eps: Optional[float] = None            # Trailing EPS
    eps_forward: Optional[float] = None    # Forward EPS
    eps_growth: Optional[float] = None     # YoY EPS growth

    # Cash Flow
    free_cashflow: Optional[float] = None
    operating_cashflow: Optional[float] = None
    capex: Optional[float] = None

    # Solvency / Leverage
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

    # Liquidity
    free_float_ratio: Optional[float] = None   # 0-1
    avg_volume_10d: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None

    # Dividend
    dividend_yield: Optional[float] = None
    # Classification
    sector: Optional[str] = None
    industry: Optional[str] = None
    # Identity / Ownership
    company_name: Optional[str] = None
    major_holders: Optional[str] = None    # comma-separated major holder names


def _safe_div(num: Optional[float], den: Optional[float]) -> Optional[float]:
    if num is None or den in (None, 0):
        return None
    return num / den


def _fetch_ownership(t: yf.Ticker) -> str:
    """Extract major holder names from yfinance, return comma-separated."""
    try:
        holders = t.institutional_holders
        if holders is not None and not holders.empty:
            names = holders["Holder"].head(5).tolist()
            return ", ".join(str(n) for n in names if n)
    except Exception:
        pass
    try:
        holders = t.major_holders
        if holders is not None and not holders.empty:
            # major_holders is a 2-col DataFrame; extract values
            names = holders.iloc[:, 1].tolist()
            return ", ".join(str(n) for n in names if n)
    except Exception:
        pass
    return ""


def fetch_fundamentals(ticker: str) -> FundamentalSnapshot:
    t = yf.Ticker(ticker)
    try:
        info = t.get_info()
    except Exception:
        try:
            info = t.info
        except Exception:
            info = {}

    if not isinstance(info, dict):
        info = {}

    # Valuation
    pbv = info.get("priceToBook")
    per = info.get("trailingPE")
    per_forward = info.get("forwardPE")
    peg = info.get("pegRatio") or info.get("trailingPegRatio")

    # Book value per share and valuation price
    bvps = info.get("bookValue")
    valuation_price: Optional[float] = None
    if pbv is not None and bvps is not None and bvps > 0:
        valuation_price = pbv * bvps

    # Profitability
    revenue = info.get("totalRevenue")
    gross_margins = info.get("grossMargins")
    op_margins = info.get("operatingMargins")
    profit_margins = info.get("profitMargins")
    roe = info.get("returnOnEquity")
    roa = info.get("returnOnAssets")
    eps = info.get("trailingEps")
    eps_fwd = info.get("forwardEps")

    # Rough ROIC = EBIT*(1-tax) / (Debt + Equity)
    ebit = info.get("ebit")
    tax_rate = info.get("effectiveTaxRate") or 0.22
    invested_capital = (info.get("totalDebt") or 0) + (info.get("totalStockholderEquity") or 0)
    roic: Optional[float] = None
    if ebit and invested_capital:
        roic = ebit * (1 - tax_rate) / invested_capital

    # EPS Growth (yoy)
    eps_growth: Optional[float] = None
    if eps and eps_fwd and eps != 0:
        eps_growth = (eps_fwd - eps) / abs(eps)

    # Cash flow
    fcf = info.get("freeCashflow")
    opcf = info.get("operatingCashflow")
    capex: Optional[float] = None
    if fcf is not None and opcf is not None:
        capex = opcf - fcf

    # Solvency
    total_debt = info.get("totalDebt")
    total_cash = info.get("totalCash")
    equity = info.get("totalStockholderEquity")
    de = _safe_div(total_debt, equity)
    current_ratio = info.get("currentRatio")
    quick_ratio = info.get("quickRatio")

    # Liquidity
    float_shares = info.get("floatShares")
    shares_out = info.get("sharesOutstanding")
    free_float_ratio = _safe_div(float_shares, shares_out)
    avg_vol = info.get("averageVolume10days") or info.get("averageVolume")
    mkt_cap = info.get("marketCap")
    ev = info.get("enterpriseValue")
    div_yield = info.get("dividendYield")

    # Classification
    sector = info.get("sector") or info.get("sectorDisp")
    industry = info.get("industry") or info.get("industryDisp")

    # Identity / Ownership
    company_name = info.get("longName") or info.get("shortName") or ""
    major_holders = _fetch_ownership(t)

    return FundamentalSnapshot(
        pbv=pbv,
        per=per,
        per_forward=per_forward,
        peg_ratio=peg,
        book_value_per_share=bvps,
        valuation_price=valuation_price,
        revenue=revenue,
        gross_margins=gross_margins,
        operating_margins=op_margins,
        profit_margins=profit_margins,
        roe=roe,
        roa=roa,
        roic=roic,
        eps=eps,
        eps_forward=eps_fwd,
        eps_growth=eps_growth,
        free_cashflow=fcf,
        operating_cashflow=opcf,
        capex=capex,
        total_debt=total_debt,
        total_cash=total_cash,
        debt_to_equity=de,
        current_ratio=current_ratio,
        quick_ratio=quick_ratio,
        free_float_ratio=free_float_ratio,
        avg_volume_10d=avg_vol,
        market_cap=mkt_cap,
        enterprise_value=ev,
        dividend_yield=div_yield,
        sector=sector,
        industry=industry,
        company_name=company_name,
        major_holders=major_holders,
    )
