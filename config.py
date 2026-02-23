"""
Configuration for Indonesia Stock Trading AI
"""
from datetime import datetime, timedelta

# LQ45 major stocks - liquid Indonesia stocks (Yahoo Finance uses .JK suffix)
IDX_STOCKS = [
    "BBCA.JK",  # Bank Central Asia
    "BBRI.JK",  # Bank Rakyat Indonesia
    "BMRI.JK",  # Bank Mandiri
    "ASII.JK",  # Astra International
    "UNTR.JK",  # United Tractors
    "GGRM.JK",  # Gudang Garam
    "INDF.JK",  # Indofood Sukses Makmur
    "SMGR.JK",  # Semen Indonesia
    "TLKM.JK",  # Telekomunikasi Indonesia
    "BREN.JK",  # Barito Renewables Energy
    "ADRO.JK",  # Adaro Energy
    "BBNI.JK",  # Bank BNI
    "TOWR.JK",  # Tower Bersama
    "PTBA.JK",  # Bukit Asam
    "ANTM.JK",  # Aneka Tambang
    "GOTO.JK",  # GoTo Gojek Tokopedia
    "ICBP.JK",  # Indofood CBP
    "CPIN.JK",  # Charoen Pokphand Indonesia
    "EMTK.JK",  # Elang Mahkota Teknologi
    "BUKA.JK",  # Bukalapak
    "ITMG.JK",  # Indo Tambangraya Megah
    "AMMN.JK",  # Amman Mineral
    "BNGA.JK",  # Bank CIMB Niaga
    "PGAS.JK",  # Perusahaan Gas Negara
    "MDKA.JK",  # Merdeka Copper Gold
    "TPIA.JK",  # Chandra Asri Petrochemical
]

# Indicator parameters
STOCH_RSI_PERIOD = 14
STOCH_RSI_K = 3
STOCH_RSI_D = 3

# Smart Money lookback period (days)
SMI_PERIOD = 14

# Rebound detection thresholds
OVERSOLD_THRESHOLD = 20  # Stoch RSI %K below = oversold (potential buy)
OVERBOUGHT_THRESHOLD = 80  # Stoch RSI %K above = overbought (potential sell)
SMI_BULLISH_THRESHOLD = 0  # SMI above = smart money accumulating

# Data range for analysis
DEFAULT_LOOKBACK_DAYS = 90


def get_data_period():
    """Get start date for data fetch"""
    return datetime.now() - timedelta(days=DEFAULT_LOOKBACK_DAYS)
