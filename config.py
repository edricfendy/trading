"""
IDX universe helpers.

Fetches ALL listed Indonesia stocks from IDX API endpoint, with fallback to
a comprehensive static list of ~800+ IDX stocks.
"""
from __future__ import annotations

import json
from typing import List, Optional, Set

import requests

from config import IDX_STOCKS

# IDX API endpoint (more reliable than scraping)
IDX_API_URL = "https://www.idx.co.id/primary/StockData/GetSecurities?start=0&length=9999&code=&name=&market=REGULER&sector=&type=s&board=&tradingUnit=&shares=&listingDate=&currency=&assetValue="
IDX_DAFTAR_SAHAM_URL = "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham"

# Comprehensive static fallback list of IDX stocks
ALL_IDX_STATIC = [
    # LQ45 + IHSG major
    "AALI.JK","ABBA.JK","ABDA.JK","ADMF.JK","ADMG.JK","ADRO.JK","AGII.JK","AGRO.JK",
    "AHAP.JK","AIMS.JK","AISA.JK","AKPI.JK","AKRA.JK","ALDO.JK","ALII.JK","ALKA.JK",
    "ALMI.JK","ALTO.JK","AMAG.JK","AMFG.JK","AMMN.JK","AMRT.JK","ANDI.JK","ANJT.JK",
    "ANTM.JK","APII.JK","APLN.JK","ARCI.JK","ARGO.JK","ARII.JK","ARKA.JK","ARMY.JK",
    "ARTA.JK","ARTL.JK","ASBI.JK","ASDM.JK","ASII.JK","ASJT.JK","ASLC.JK","ASMI.JK",
    "ASRI.JK","ASRM.JK","ASSA.JK","ATAP.JK","ATPK.JK","AUTO.JK","AYLS.JK",
    "BACA.JK","BALI.JK","BAPA.JK","BAPI.JK","BBCA.JK","BBHI.JK","BBKP.JK","BBMD.JK",
    "BBNI.JK","BBRI.JK","BBSI.JK","BBTN.JK","BBYB.JK","BCAP.JK","BCIC.JK","BCIP.JK",
    "BDMN.JK","BEKS.JK","BELI.JK","BELL.JK","BFIN.JK","BGTG.JK","BHIT.JK","BIKA.JK",
    "BIMA.JK","BINA.JK","BJBR.JK","BJTM.JK","BKDP.JK","BKSL.JK","BKSW.JK","BLTA.JK",
    "BLTZ.JK","BMRI.JK","BMSR.JK","BMTR.JK","BNBA.JK","BNGA.JK","BNII.JK","BNLI.JK",
    "BOBA.JK","BOGA.JK","BOLA.JK","BOSS.JK","BPII.JK","BPTR.JK","BRAM.JK","BREN.JK",
    "BRNA.JK","BRPT.JK","BSDE.JK","BSSR.JK","BTEK.JK","BTEL.JK","BTON.JK","BUDI.JK",
    "BUGO.JK","BUKA.JK","BULL.JK","BUMI.JK","BVIC.JK","BWPT.JK","BYMS.JK",
    "CAKK.JK","CAMP.JK","CANI.JK","CARE.JK","CARS.JK","CASA.JK","CASH.JK","CASS.JK",
    "CBPE.JK","CCSI.JK","CEKA.JK","CENT.JK","CFIN.JK","CINT.JK","CITY.JK","CLEO.JK",
    "CLPI.JK","CMPP.JK","CMRY.JK","CNKO.JK","CNTX.JK","COCO.JK","CPRO.JK","CPIN.JK",
    "CSIS.JK","CSMI.JK","CTRA.JK","CUAN.JK","DEAL.JK","DEFI.JK","DEWA.JK","DFAM.JK",
    "DGIK.JK","DILD.JK","DKFT.JK","DLTA.JK","DMAS.JK","DMMX.JK","DNAR.JK","DPUM.JK",
    "DRMA.JK","DSSA.JK","DUCK.JK","DUTI.JK",
    "ECII.JK","EDGE.JK","EKAD.JK","ELSA.JK","EMDE.JK","EMTK.JK","ENRG.JK","EPAC.JK",
    "ERAA.JK","ESSA.JK","EURO.JK","EXCL.JK",
    "FAST.JK","FASW.JK","FILM.JK","FITT.JK","FLMC.JK","FMII.JK","FOOD.JK","FORE.JK",
    "FORZ.JK","FPNI.JK","FREN.JK","FUJI.JK","GAMA.JK","GDST.JK","GDYR.JK","GEMA.JK",
    "GEMS.JK","GGRM.JK","GHON.JK","GJTL.JK","GLOB.JK","GLVA.JK","GOOD.JK","GOTO.JK",
    "GPRA.JK","GRIA.JK","GRMN.JK","GTSI.JK",
    "HADE.JK","HAIS.JK","HAKA.JK","HALO.JK","HDTX.JK","HEAL.JK","HERO.JK","HEXA.JK",
    "HITS.JK","HJKL.JK","HMSP.JK","HOME.JK","HOMI.JK","HRME.JK","HRTA.JK","HRUM.JK",
    "IATA.JK","IBST.JK","ICBP.JK","ICON.JK","IDEA.JK","IDPR.JK","IFII.JK","IFSH.JK",
    "IGAR.JK","IGBP.JK","IGTA.JK","IIKP.JK","IKAI.JK","IMAS.JK","IMJS.JK","IMPC.JK",
    "INAF.JK","INAI.JK","INCF.JK","INCO.JK","INDF.JK","INDIKA.JK","INDR.JK","INDS.JK",
    "INET.JK","INPC.JK","INPP.JK","INRU.JK","INTA.JK","INTD.JK","INTP.JK","IPPE.JK",
    "IPCC.JK","IPOL.JK","ISAT.JK","ISSP.JK","ITIC.JK","ITMA.JK","ITMG.JK","IVGN.JK",
    "JAWA.JK","JGLE.JK","JKON.JK","JKSW.JK","JPFA.JK","JRPT.JK","JSMR.JK","JTPE.JK",
    "KAEF.JK","KARW.JK","KBLI.JK","KBLM.JK","KBRI.JK","KDSI.JK","KIAS.JK","KICI.JK",
    "KIJA.JK","KINO.JK","KLBF.JK","KMTR.JK","KOBX.JK","KOIN.JK","KONI.JK","KOPI.JK",
    "KPAS.JK","KPIG.JK","KREN.JK","LAMI.JK","LAND.JK","LCKM.JK","LEAD.JK","LION.JK",
    "LMAS.JK","LMPI.JK","LMSH.JK","LPCK.JK","LPKR.JK","LPLI.JK","LPPF.JK","LPPS.JK",
    "LSIP.JK","MABA.JK","MAIN.JK","MAMI.JK","MAPA.JK","MAPI.JK","MARI.JK","MARK.JK",
    "MASA.JK","MАТА.JK","MBSS.JK","MCAS.JK","MCOL.JK","MDKA.JK","MDRN.JK","MEDC.JK",
    "MFIN.JK","MFMI.JK","MGNA.JK","MIKA.JK","MITI.JK","MKPI.JK","MLBI.JK","MLIA.JK",
    "MLPT.JK","MNCN.JK","MPMX.JK","MPOW.JK","MPRO.JK","MRAT.JK","MREI.JK","MTDL.JK",
    "MTEL.JK","MTLA.JK","MTPS.JK","MTRA.JK","MYOH.JK",
    "NASI.JK","NCKL.JK","NELY.JK","NETV.JK","NICL.JK","NIRO.JK","NISP.JK","NKLA.JK",
    "NOBU.JK","NRCA.JK","NTBK.JK","NUSA.JK",
    "OBMD.JK","OCAP.JK","OKAS.JK","OLIV.JK","OMRE.JK","OPMS.JK","OVIO.JK",
    "PALM.JK","PAMG.JK","PANI.JK","PANR.JK","PANS.JK","PBID.JK","PBRX.JK","PDPP.JK",
    "PEGE.JK","PEHA.JK","PGAS.JK","PGEO.JK","PGLI.JK","PICO.JK","PJAA.JK","PKPK.JK",
    "PLAN.JK","PLIN.JK","PMMP.JK","PNBN.JK","PNBS.JK","PNGO.JK","PNIN.JK","PNSE.JK",
    "POLI.JK","POLL.JK","POLY.JK","PORT.JK","POWR.JK","PPGL.JK","PPRE.JK","PPRO.JK",
    "PRAS.JK","PRDA.JK","PREY.JK","PRIM.JK","PROSPEK.JK","PSAB.JK","PSDN.JK","PSEI.JK",
    "PTBA.JK","PTIS.JK","PTPP.JK","PTPW.JK","PTRO.JK","PTSP.JK","PTSN.JK","PUBM.JK",
    "PURE.JK","PWON.JK","PYFA.JK",
    "RAJA.JK","RALS.JK","RBMS.JK","RDTX.JK","RELI.JK","RGAS.JK","RIGS.JK","RIMO.JK",
    "RODA.JK","ROTI.JK","RUIS.JK","RUNS.JK",
    "SAFE.JK","SAME.JK","SAMF.JK","SAPX.JK","SATU.JK","SCCO.JK","SCMA.JK","SCPI.JK",
    "SDMU.JK","SDPC.JK","SEKAR.JK","SFAN.JK","SGER.JK","SGRO.JK","SHIP.JK","SILO.JK",
    "SIMP.JK","SKBM.JK","SKLT.JK","SMAR.JK","SMCB.JK","SMDR.JK","SMDM.JK","SMGR.JK",
    "SMIL.JK","SMKL.JK","SMMA.JK","SMMT.JK","SMRA.JK","SMSM.JK","SOCI.JK","SOHO.JK",
    "SONA.JK","SOSS.JK","SPMA.JK","SRAJ.JK","SRTG.JK","SSIA.JK","SSMS.JK","SSTM.JK",
    "STAA.JK","STAR.JK","STEEL.JK","STTP.JK","SUGI.JK","SULI.JK","SUPR.JK","SURE.JK",
    "TALF.JK","TARA.JK","TAXI.JK","TBIG.JK","TBLA.JK","TCPI.JK","TDPM.JK","TELE.JK",
    "TFAS.JK","TGRA.JK","TIGA.JK","TINS.JK","TKIM.JK","TLKM.JK","TMAS.JK","TMPO.JK",
    "TOPS.JK","TOTL.JK","TOWR.JK","TPIA.JK","TPMA.JK","TRIM.JK","TRIS.JK","TRST.JK",
    "TRUS.JK","TSPC.JK","TUGU.JK","TURI.JK","UNIQ.JK",
    "UNSP.JK","UNTR.JK","UNVR.JK","UOBK.JK","UPCL.JK","VICI.JK","VINS.JK","VIVA.JK",
    "VKTR.JK","VOKS.JK","VSAT.JK",
    "WEGE.JK","WEHA.JK","WIFI.JK","WIIM.JK","WIKA.JK","WINS.JK","WMPP.JK","WOOD.JK",
    "WOWS.JK","WSKT.JK","WTON.JK",
    "YPAS.JK","YTKI.JK","YULE.JK",
    "ZBRA.JK",
]


def fetch_all_idx_tickers(max_count: Optional[int] = None) -> list[str]:
    """
    Try to fetch the current full IDX stock list from IDX API.
    Falls back to comprehensive static list if anything goes wrong.
    """
    tickers: list[str] = []
    try:
        resp = requests.get(
            IDX_API_URL,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 (compatible; IDXScanner/1.0)"},
        )
        if resp.ok:
            data = resp.json()
            records = data.get("data", []) or data.get("Data", [])
            for r in records:
                code = r.get("Code", r.get("code", "")).strip()
                if code and len(code) <= 6:
                    tickers.append(f"{code}.JK")
    except Exception:
        pass

    if not tickers:
        tickers = list(ALL_IDX_STATIC)

    # Deduplicate preserve order
    seen: Set[str] = set()
    unique = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    if max_count is not None and max_count > 0:
        return unique[:max_count]
    return unique


def get_universe(all_idx: bool = True, limit: Optional[int] = None) -> list[str]:
    """
    Returns the list of tickers to scan.
    - all_idx=True: try full IDX list (~800+ stocks), fallback to static
    - all_idx=False: use static IDX_STOCKS (LQ45 core) list only
    """
    if all_idx:
        return fetch_all_idx_tickers(max_count=limit)
    from config import IDX_STOCKS
    return list(IDX_STOCKS)
