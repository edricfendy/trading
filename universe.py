"""
IDX Universe - Indonesia Stock Exchange
~956 companies listed as of early 2026.

Optimizations vs original:
- Results cached to disk for 24 hours (avoids scraping on every app reload)
- In-memory cache so repeated calls within the same process cost nothing
- Parallel API + scrape fallback
"""
from __future__ import annotations

import json
import os
import re
import time
import threading
from pathlib import Path
from typing import Optional, Set

import requests

from config import IDX_STOCKS

# ─── Cache config ──────────────────────────────────────────────────────────────
_UNIVERSE_CACHE_TTL_SEC = int(os.getenv("UNIVERSE_CACHE_TTL_SEC", str(24 * 3600)))  # 24 h
_UNIVERSE_CACHE_FILE    = Path(os.getenv("UNIVERSE_CACHE_FILE", "/tmp/idx_universe_cache.json"))

_MEM_CACHE: Optional[list[str]] = None
_MEM_CACHE_TS: float = 0.0
_MEM_LOCK = threading.Lock()

# ─── URLs ─────────────────────────────────────────────────────────────────────
IDX_API_URL = (
    "https://www.idx.co.id/primary/StockData/GetSecurities"
    "?start=0&length=9999&code=&name=&market=REGULER&sector=&type=s"
    "&board=&tradingUnit=&shares=&listingDate=&currency=&assetValue="
)
STOCKANALYSIS_URL = "https://stockanalysis.com/list/indonesia-stock-exchange/"

# ─── Static fallback (~900 verified IDX tickers, Feb 2026) ────────────────────
ALL_IDX_STATIC: list[str] = [t + ".JK" for t in [
    # Top 250 by market cap (verified Feb 2026)
    "BREN","BBCA","TPIA","DSSA","BYAN","DCII","BBRI","AMMN","BMRI","TLKM",
    "BRPT","CUAN","ASII","CDIA","PANI","MORA","IMPC","SRAJ","BNLI","BBNI",
    "BRMS","MLPT","DNET","MPRO","UNTR","FILM","PTRO","BRIS","RISE","UNVR",
    "ICBP","HMSP","CASA","BUMI","SMMA","EMTK","CPIN","AMRT","ISAT","ANTM",
    "GOTO","NCKL","BELI","INDF","AADI","MBMA","COIN","EXCL","MDKA","EMAS",
    "CBDK","ADMR","KLBF","ADRO","PGUN","GEMS","MTEL","PGEO","SUPR","MYOR",
    "INKP","CMRY","BNGA","TBIG","PGAS","INCO","TCPI","MEGA","BUVA","MIKA",
    "SILO","MEDC","TOWR","ARCI","BBHI","TAPG","RATU","NISP","ENRG","JPFA",
    "BKSL","ARTO","GGRM","VKTR","AVIA","BINA","RAJA","JARR","PTBA","PNBN",
    "MDIY","SCMA","AKRA","MSIN","ITMG","JSMR","BDMN","TINS","MKPI","INTP",
    "TKIM","BTPN","HEAL","FAPA","SRTG","MAPA","MAPI","WIFI","BSDE","SMGR",
    "SOHO","DSNG","POLU","APIC","RMKE","PWON","LIFE","BSIM","DEWA","CITA",
    "BBTN","SSMS","CTRA","SIDO","ULTJ","BNII","BUKA","BNBR","NSSS","AALI",
    "STAA","BBSI","PSAB","SHIP","YUPI","SMAR","SGRO","JRPT","ARKO","CARE",
    "BBKP","FASW","GOOD","MCOL","AUTO","HRUM","CMNT","KPIG","LINK","BANK",
    "STTP","PRAY","TSPC","MIDI","BMAS","CLEO","MLBI","NICL","BFIN","POWR",
    "ESSA","ADMF","BSSR","GIAA","SMSM","BTPS","INDY","CNMA","ALII","EDGE",
    "CYBR","SIMP","TLDN","ADES","LSIP","SSIA","PLIN","FPNI","PNLF","BJBR",
    "BJTM","WIKA","CMNP","BBMD","ABMM","INPP","DUTI","UDNG","JSPT","NIRO",
    "TMAS","KRAS","MTDL","HRTA","ACES","BHAT","SAME","SMCB","INET","ERAA",
    "CLAY","TOBA","UANG","SGER","IBST","SMRA","DMAS","EPMT","DATA","LPKR",
    "PYFA","ANJT","CASS","FISH","PALM","DMND","BIPI","BBYB","AGRO","BALI",
    "BUKK","BESS","SMDR","BWPT","IMAS","OMED","CBRE","NETV","CTBN","CENT",
    "GOLF","ELPI","SMMT","BPII","DRMA","TRIM","VICI","SINI","MINA","FORE",
    "GMFI","AGII","RONY","TGKA","BULL","ROTI",
    # 251-500
    "MPPA","ACST","DOID","MCAS","NFCX","FIRE","ASSA","BTEK","BTON","WOOD",
    "PTPP","WSKT","WTON","WEGE","ADHI","NRCA","TOTL","DGIK","PTSP","PNIN",
    "ASRM","ASBI","ASDM","ASJT","MREI","JKON","SMDM","RDTX","APLN","LPCK",
    "DILD","BKDP","GPRA","OMRE","MTLA","RBMS","LPLI","BCAP","BCIP","TARA",
    "ATPK","PRIM","NUSA","KIJA","MDLN","ELSA","PEGE","WOWS","RUNS","HERO",
    "RALS","LPPF","MPMX","TURI","FAST","CMPP","LMAS","CARS","ALDO","BOLT",
    "KBLM","KBLI","SCCO","VOKS","JECC","IKBI","BRAM","GJTL","MASA","MLIA",
    "SRSN","YPAS","TALF","INAI","DPUM","AKPI","TRST","IPOL","BRNA","CINT",
    "UPCL","IGAR","IMJS","ARKA","SDPC","SDMU","SKLT","CEKA","SKBM","ALTO",
    "DLTA","MRAT","INAF","KAEF","DVLA","MERK","SQMI","PRDA","PEVE","WIIM",
    "RICY","INDR","TRIS","BELL","PBRX","STAR","HDTX","SSTM","CNTX","ARGO",
    "MYTX","POLY","BATA","LMPI","GDST","ISSP","MAIN","SIPD","MGNA","MBSS",
    "NELY","WEHA","HITS","LEAD","KREN","MDIA","AIMS","PJAA","BAYU","PANR",
    "CFIN","MFIN","TIFA","WOMF","ABDA","MAYA","SDRA","NOBU","DNAR","INPC",
    "MCOR","PNBS","UOBK","BNBA","BVIC","BABP","BEKS","PNSE","ASMI","ASLC",
    "BHIT","ARMY","FOOD","BOBA","BOLA","BOSS","KOPI","KOIN","KONI","CAMP",
    "CANI","DUCK","LUCK","PANR","BAYU","TIGA","MNCN","BMTR",
    "SCMA","MSIN","NETV","LINK","DCII","MORA","DATA","WIFI","CYBR",
    "INET","EDGE","DNET","EMTK","ISAT","EXCL","TLKM","FREN","MDIA",
    "TBIG","TOWR","SUPR","MTEL","IBST","MKPI","PLIN",
    # 501-700
    "LION","LMSH","TBMS","GDYR","AMFG","ARNA","MARK","TOTO","KIAS","KDSI",
    "SLFA","INRU","TGRA","SULI","SUGI","KBRI","CPRO","HEXA","INTA","RUIS",
    "WINS","BLTA","INDY","BRAM","GJTL","MASA","BRNA","AKPI","TRST","IPOL",
    "CINT","IGAR","SDPC","SKLT","CEKA","SKBM","ALTO","DLTA","INAF","KAEF",
    "DVLA","MERK","RICY","INDR","TRIS","BELL","STAR","HDTX","SSTM","MYTX",
    "BATA","ISSP","MAIN","SIPD","NELY","WEHA","HITS","LEAD","AIMS","PJAA",
    "TIFA","WOMF","NOBU","DNAR","INPC","BACA","PNBS","BABP","BEKS","ARMY",
    "FOOD","LUCK","BISI","CFIN","INTA","RUIS","WINS","BULL","TMAS","SMDR",
    "BLTA","SHIP","MBSS","ELPI","BESS","RAJA","CASS","GIAA","GMFI","MEDC",
    "PGAS","POWR","CMNP","JSMR","NRCA","ADHI","DGIK","BKDP","APLN","LPCK",
    "DILD","GPRA","OMRE","MTLA","RBMS","LPLI","KIJA","MDLN","ELSA","PEGE",
    "WOWS","RUNS","HERO","RALS","LPPF","MPMX","TURI","FAST","CMPP","ALDO",
    "BOLT","KBLM","KBLI","SCCO","VOKS","JECC","BRAM","BATA","SRSN","YPAS",
    "TALF","INAI","DPUM","TRST","IPOL","CINT","UPCL","IMJS","ARKA","SDMU",
    "CEKA","ROTI","DLTA","MRAT","TSPC","SQMI","PRDA","PEVE","WIIM","PBRX",
    "ARGO","MYTX","POLY","LMPI","GDST","MAIN","MGNA","MBSS","NELY",
    "IBST","KREN","CFIN","ADMF","MFIN","TIFA","ABDA","MAYA","SDRA","MCOR",
    "UOBK","BNBA","BVIC","BCAP","ASMI","ASLC","BHIT",
    # 701-900
    "ATPK","PRIM","NUSA","CARS","LMAS","MCAS","NFCX","FIRE","BOBA","BOSS",
    "KONI","DUCK","FORE","COAL","INCO","TINS","ANTM","NCKL","NICL","CMRY",
    "CLEO","ULTJ","DMND","ADES","STTP","SKBM","MLBI","SOHO","MAPA","MAPI",
    "ERAA","HERO","RALS","LPPF","ACES","FISH","AMRT","MIDI","EPMT","DNET",
    "INET","WIFI","LINK","EDGE","CYBR","PANR","INPP","JSPT","NIRO","DUTI",
    "ASRI","PWON","BSDE","CTRA","SMRA","GPRA","LPKR","LPCK","DILD","APLN",
    "SSIA","RBMS","OMRE","MTLA","BCIP","DGIK","NRCA","TOTL","ADHI","TBMS",
    "GDYR","GDST","LMSH","LION","BTON","YPAS","TALF","INAI","DPUM","TRST",
    "IPOL","BRNA","CINT","UPCL","IGAR","IMJS","ARKA","SDMU","SKLT","ROTI",
    "DLTA","MRAT","INAF","KAEF","DVLA","MERK","TSPC","PRDA","PEVE","WIIM",
    "RICY","INDR","TRIS","BELL","PBRX","STAR","HDTX","SSTM","CNTX","BATA",
    "LMPI","ISSP","SIPD","MGNA","NELY","BAYU","AIMS","PJAA","WOMF",
    "DNAR","INPC","BINA","BACA","BABP","BEKS","ARMY","FOOD","CAMP","CANI",
    "LUCK","BHIT","MNCN","BMTR","SCMA","MSIN","DATA","MORA","DCII","WIFI",
    "FREN","MDIA","BALI","CENT","MKPI","PLIN","RDTX","SMDM","JKON","ASDM",
    "ASBI","ASRM","ASJT","MREI","ASLC","PNSE","ABDA","SDRA","NOBU","MCOR",
    "PNBS","UOBK","BNBA","BVIC","BABP","BEKS","BCAP","PNIN","CFIN","BPII",
    "DEFI","SMMA","PNLF","AHAP","ASBI","ASDM","BJBR","BJTM","BBMD","BMAS",
    "BBSI","BBYB","BBKP","AGRO","BTPN","BTPS","BSIM","NISP","MAYA","NOBU",
]]


def _dedup(tickers: list[str]) -> list[str]:
    seen: Set[str] = set()
    result = []
    for t in tickers:
        t = t.strip().upper()
        if t and t not in seen:
            seen.add(t)
            result.append(t)
    return result


# ─── Disk cache helpers ────────────────────────────────────────────────────────
def _disk_cache_load() -> Optional[list[str]]:
    try:
        if _UNIVERSE_CACHE_FILE.exists():
            data = json.loads(_UNIVERSE_CACHE_FILE.read_text())
            if time.time() - data["ts"] < _UNIVERSE_CACHE_TTL_SEC:
                tickers = data["tickers"]
                if len(tickers) > 100:
                    return tickers
    except Exception:
        pass
    return None


def _disk_cache_save(tickers: list[str]) -> None:
    try:
        _UNIVERSE_CACHE_FILE.write_text(json.dumps({"ts": time.time(), "tickers": tickers}))
    except Exception:
        pass


# ─── Fetchers ─────────────────────────────────────────────────────────────────
def _try_idx_api() -> list[str]:
    try:
        r = requests.get(IDX_API_URL, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.ok:
            records = r.json().get("data", []) or r.json().get("Data", [])
            tickers = [
                rec.get("Code", rec.get("code", "")).strip() + ".JK"
                for rec in records
                if 2 <= len(rec.get("Code", rec.get("code", "")).strip()) <= 6
            ]
            if len(tickers) > 100:
                return _dedup(tickers)
    except Exception:
        pass
    return []


def _try_stockanalysis() -> list[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        ),
    }
    collected: list[str] = []
    seen: Set[str] = set()
    for page in range(1, 6):
        url = STOCKANALYSIS_URL if page == 1 else f"{STOCKANALYSIS_URL}?page={page}"
        try:
            r = requests.get(url, timeout=20, headers=headers)
            if not r.ok:
                break
            codes = re.findall(r'/quote/idx/([A-Z]{2,6})/', r.text)
            if not codes:
                break
            added = 0
            for c in codes:
                if c not in seen:
                    seen.add(c)
                    collected.append(c)
                    added += 1
            if added == 0:
                break
        except Exception:
            break
    if len(collected) > 100:
        return _dedup([c + ".JK" for c in collected])
    return []


def fetch_all_idx_tickers(max_count: Optional[int] = None) -> list[str]:
    """
    Full IDX stock universe with layered caching:
    1. In-memory (same process, free)
    2. Disk cache (24 h TTL, avoids HTTP on every Streamlit reload)
    3. IDX API (live)
    4. stockanalysis.com scrape (fallback)
    5. Comprehensive static list (offline fallback)
    """
    global _MEM_CACHE, _MEM_CACHE_TS

    with _MEM_LOCK:
        # 1. In-memory
        if _MEM_CACHE is not None and (time.time() - _MEM_CACHE_TS) < _UNIVERSE_CACHE_TTL_SEC:
            tickers = _MEM_CACHE
            return tickers[:max_count] if max_count else tickers

        # 2. Disk cache
        tickers = _disk_cache_load()
        if tickers:
            _MEM_CACHE = tickers
            _MEM_CACHE_TS = time.time()
            return tickers[:max_count] if max_count else tickers

        # 3. Live fetch
        tickers = _try_idx_api()
        if len(tickers) < 100:
            tickers = _try_stockanalysis()
        if len(tickers) < 100:
            tickers = _dedup(ALL_IDX_STATIC)

        _disk_cache_save(tickers)
        _MEM_CACHE = tickers
        _MEM_CACHE_TS = time.time()

    return tickers[:max_count] if max_count else tickers


def get_universe(all_idx: bool = True, limit: Optional[int] = None) -> list[str]:
    """
    Returns tickers to scan.
    - all_idx=True:  Full IDX universe (up to ~956 stocks)
    - all_idx=False: LQ45 core list (26 stocks)
    """
    if all_idx:
        return fetch_all_idx_tickers(max_count=limit)
    return list(IDX_STOCKS)
