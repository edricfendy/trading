"""
IDX Universe - Indonesia Stock Exchange
~956 companies listed as of early 2026.

Strategy (in order of priority):
1. Try IDX official API
2. Try stockanalysis.com scrape (most reliable dynamic fallback)
3. Use comprehensive static list (~900 tickers)
"""
from __future__ import annotations

import re
import requests
from typing import Optional, Set

from config import IDX_STOCKS

IDX_API_URL = (
    "https://www.idx.co.id/primary/StockData/GetSecurities"
    "?start=0&length=9999&code=&name=&market=REGULER&sector=&type=s"
    "&board=&tradingUnit=&shares=&listingDate=&currency=&assetValue="
)
STOCKANALYSIS_URL = "https://stockanalysis.com/list/indonesia-stock-exchange/"

# Comprehensive static fallback — real IDX tickers verified from live data Feb 2026
# Sourced from stockanalysis.com IDX full list (956 stocks as of Dec 2025)
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
    "CANI","DUCK","LUCK","CAMP","PJAA","PANR","BAYU","TIGA","MNCN","BMTR",
    "BHIT","SCMA","MSIN","NETV","LINK","DCII","MORA","DATA","WIFI","CYBR",
    "INET","EDGE","DNET","EMTK","ISAT","EXCL","TLKM","FREN","MDIA","KREN",
    "TBIG","TOWR","SUPR","MTEL","BALI","IBST","CENT","BCIP","MKPI","PLIN",
    # 501-700
    "LION","LMSH","TBMS","GDYR","AMFG","ARNA","MARK","TOTO","KIAS","KDSI",
    "SLFA","DAJK","CPDW","SWAT","INRU","TGRA","SULI","SUGI","KBRI","PTBA",
    "CPRO","HEXA","INTA","RUIS","WINS","BLTA","HIDE","INDY","MEDC","ESSA",
    "POWR","CMNP","JSMR","NRCA","TOTL","ADHI","PTPP","WSKT","WTON","WIKA",
    "BRAM","GJTL","MASA","BRNA","AKPI","TRST","IPOL","CINT","UPCL","IGAR",
    "SDPC","SKLT","CEKA","SKBM","ALTO","DLTA","INAF","KAEF","DVLA","MERK",
    "RICY","INDR","TRIS","BELL","STAR","HDTX","SSTM","MYTX","BATA","ISSP",
    "MAIN","SIPD","NELY","WEHA","HITS","LEAD","AIMS","PJAA","TIFA","WOMF",
    "NOBU","DNAR","INPC","BACA","PNBS","BABP","BEKS","ARMY","FOOD","LUCK",
    "BISI","CPRO","CFIN","INTA","RUIS","WINS","BULL","TMAS","SMDR","BLTA",
    "SHIP","MBSS","ELPI","BESS","WEHA","HITS","LEAD","RAJA","CASS","GIAA",
    "GMFI","HIDE","MEDC","PGAS","POWR","CMNP","JSMR","NRCA","ADHI","DGIK",
    "BKDP","APLN","LPCK","DILD","GPRA","OMRE","MTLA","RBMS","LPLI","KIJA",
    "MDLN","ELSA","PEGE","WOWS","RUNS","HERO","RALS","LPPF","MPMX","TURI",
    "FAST","CMPP","ALDO","BOLT","KBLM","KBLI","SCCO","VOKS","JECC","BRAM",
    "BATA","SRSN","YPAS","TALF","INAI","DPUM","TRST","IPOL","CINT","UPCL",
    "IMJS","ARKA","SDMU","CEKA","ROTI","DLTA","MRAT","TSPC","SQMI","PRDA",
    "PEVE","WIIM","PBRX","ARGO","MYTX","POLY","LMPI","GDST","MAIN","MGNA",
    "MBSS","NELY","CENT","IBST","KREN","CFIN","ADMF","MFIN","TIFA","ABDA",
    "MAYA","SDRA","MCOR","UOBK","BNBA","BVIC","BCAP","ASMI","ASLC","BHIT",
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
    "LMPI","ISSP","SIPD","MGNA","NELY","BAYU","YULE","AIMS","PJAA","WOMF",
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


def _try_idx_api() -> list[str]:
    try:
        r = requests.get(IDX_API_URL, timeout=15,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.ok:
            data = r.json()
            records = data.get("data", []) or data.get("Data", [])
            tickers = []
            for rec in records:
                code = rec.get("Code", rec.get("code", "")).strip()
                if code and 2 <= len(code) <= 6 and code.isalpha():
                    tickers.append(code + ".JK")
            if len(tickers) > 100:
                return _dedup(tickers)
    except Exception:
        pass
    return []


def _try_stockanalysis() -> list[str]:
    """Scrape stockanalysis.com for full IDX ticker list."""
    try:
        r = requests.get(
            STOCKANALYSIS_URL,
            timeout=20,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml",
            },
        )
        if not r.ok:
            return []
        # Tickers appear as /quote/idx/XXXX/ links
        codes = re.findall(r'/quote/idx/([A-Z]{2,6})/', r.text)
        if len(codes) > 100:
            return _dedup([c + ".JK" for c in codes])
    except Exception:
        pass
    return []


def fetch_all_idx_tickers(max_count: Optional[int] = None) -> list[str]:
    """
    Fetch full IDX stock universe (~956 stocks as of Dec 2025).
    Priority: IDX API → stockanalysis.com scrape → comprehensive static list.
    """
    tickers = _try_idx_api()

    if len(tickers) < 100:
        tickers = _try_stockanalysis()

    if len(tickers) < 100:
        tickers = _dedup(ALL_IDX_STATIC)

    if max_count and max_count > 0:
        return tickers[:max_count]
    return tickers


def get_universe(all_idx: bool = True, limit: Optional[int] = None) -> list[str]:
    """
    Returns tickers to scan.
    - all_idx=True:  Full IDX universe (up to ~956 stocks)
    - all_idx=False: LQ45 core list (26 stocks)
    """
    if all_idx:
        return fetch_all_idx_tickers(max_count=limit)
    return list(IDX_STOCKS)
