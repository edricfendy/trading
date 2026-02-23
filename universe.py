"""
IDX universe helpers.

By default we try to fetch ALL listed Indonesia stocks (Daftar Saham) from
the official IDX website. If that fails, we fall back to the static LQ45‑style
list in config.IDX_STOCKS.
"""
from __future__ import annotations

from typing import List, Optional, Set

import requests
from bs4 import BeautifulSoup

from config import IDX_STOCKS

IDX_DAFTAR_SAHAM_URL = "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham"


def _extract_tickers_from_html(html: str) -> list[str]:
    """
    Best-effort parser for the Daftar Saham page.
    Looks for any table with a header column containing 'kode' or 'code'.
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    tickers: Set[str] = set()

    for table in tables:
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers:
            continue

        try:
            code_idx = next(
                i for i, h in enumerate(headers) if "kode" in h or "code" in h
            )
        except StopIteration:
            continue

        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) <= code_idx:
                continue
            code = cells[code_idx].strip()
            if not code or code.lower() == "kode":
                continue
            # Basic clean-up
            code = code.replace(" ", "").upper()
            if "." not in code:
                code = f"{code}.JK"
            tickers.add(code)

    return sorted(tickers)


def fetch_all_idx_tickers(max_count: Optional[int] = None) -> list[str]:
    """
    Try to fetch the current full IDX stock list from IDX.
    Falls back to IDX_STOCKS if anything goes wrong.
    """
    try:
        resp = requests.get(
            IDX_DAFTAR_SAHAM_URL,
            timeout=20,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; IDXScanner/1.0)",
            },
        )
        resp.raise_for_status()
        tickers = _extract_tickers_from_html(resp.text)
    except Exception:
        tickers = []

    if not tickers:
        # Fallback to static list
        tickers = list(IDX_STOCKS)

    if max_count is not None and max_count > 0:
        return tickers[:max_count]
    return tickers


def get_universe(all_idx: bool = True, limit: Optional[int] = None) -> list[str]:
    """
    Returns the list of tickers to scan.
    - all_idx=True: try full IDX list, fallback to IDX_STOCKS
    - all_idx=False: use static IDX_STOCKS list only
    """
    if all_idx:
        return fetch_all_idx_tickers(max_count=limit)
    return list(IDX_STOCKS)

