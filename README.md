# Indonesia Stock Trading AI

Real-time Indonesia stock tracking with buy/sell timing and rebound screening based on **Stochastic RSI** and **Smart Money accumulation**.

## Features

- **Buy/Sell Timing** – Best time to buy or sell based on Stochastic RSI and Smart Money Index (SMI)
- **Rebound Screening** – Stocks with rebound potential (oversold + smart money accumulation)
- **LQ45 Focus** – Tracks major Indonesia stocks (e.g. BBCA, BBRI, ASII) with `.JK` tickers
- **Indicators**
  - **Stochastic RSI** – Oversold (<20) = potential buy, Overbought (>80) = potential sell
  - **Smart Money Index** – Institutional accumulation (positive) vs distribution (negative)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Streamlit (recommended):**
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open http://localhost:8501 in your browser.

**CLI:**
```bash
python main.py
```

### Output

1. **BUY/SELL signals** – Actions with confidence and reasons
2. **Rebound candidates** – Stocks likely to rebound based on oversold levels and smart money flow
3. **Tips** – Short explanation of the indicators

## Configuration

Edit `config.py` to:

- Add/remove stocks in `IDX_STOCKS`
- Change thresholds: `OVERSOLD_THRESHOLD`, `OVERBOUGHT_THRESHOLD`, `SMI_BULLISH_THRESHOLD`
- Adjust indicator periods: `STOCH_RSI_PERIOD`, `SMI_PERIOD`

## Data Provider

This app now reads market data from **Yahoo Finance** (default), **Twelve Data**, **GOAPI**, or **Alpha Vantage**.

Set environment variables before running:

- `DATA_PROVIDER` = `yfinance`, `twelvedata`, `goapi`, or `alphavantage`
- `TWELVEDATA_API_KEY` (required for Twelve Data)
- `GOAPI_API_KEY` (required for GOAPI)
- `ALPHAVANTAGE_API_KEY` (required for Alpha Vantage)
- `GOAPI_BASE_URL` (default `https://api.goapi.id/v1`)
- `GOAPI_OHLCV_ENDPOINT` (default `/stock/idx/{symbol}/historical`)
- `GOAPI_QUOTE_ENDPOINT` (default `/stock/idx/prices`)
- `.JK` tickers are automatically converted to `:IDX` for Twelve Data

You can also use a local `.env` file (ignored by git). Example:
```bash
DATA_PROVIDER=twelvedata
TWELVEDATA_API_KEY=YOUR_KEY
```

GOAPI example:
```bash
DATA_PROVIDER=goapi
GOAPI_API_KEY=YOUR_KEY
```

Alpha Vantage example:
```bash
DATA_PROVIDER=alphavantage
ALPHAVANTAGE_API_KEY=YOUR_KEY
```

Enhanced yfinance example (default):
```bash
DATA_PROVIDER=yfinance
YF_MAX_RETRIES=3
YF_RETRY_BACKOFF_SEC=1.0
```

Optional tuning to reduce rate-limit errors:
- `DATA_CACHE_TTL_SEC` (default `30`) – reuse cached OHLCV for a short time
- `TWELVEDATA_MAX_RETRIES` (default `2`) – retry on 429
- `TWELVEDATA_RETRY_BACKOFF_SEC` (default `2.0`) – backoff between retries
- `GOAPI_SLEEP_BETWEEN_CALLS_SEC` (default `0.2`) – pause between GOAPI calls
- `ALPHAVANTAGE_MAX_RETRIES` (default `2`) – retry on errors
- `ALPHAVANTAGE_RETRY_BACKOFF_SEC` (default `2.0`) – backoff between retries
- `YF_MAX_RETRIES` (default `3`) – retry on Yahoo Finance errors
- `YF_RETRY_BACKOFF_SEC` (default `1.0`) – backoff between retries

## Deploy on Render (Streamlit)

1. Create a Web Service and connect your repo
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`

## Data Source

Uses **Twelve Data** by default. Note: Twelve Data lists IDX coverage as end-of-day, not real-time. For real-time IDX data, use a provider like GOAPI and set `DATA_PROVIDER=goapi`.

## Disclaimer

This is for educational/research use. Past performance does not guarantee future results. Do your own research before trading.
