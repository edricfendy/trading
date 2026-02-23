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

Run the analyzer:

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

## Data Source

Uses **yfinance** (Yahoo Finance) with `.JK` suffix for Indonesia stocks. Data is delayed (≈15 min). For real-time trading, consider a paid IDX API (e.g. RapidAPI Indonesia Stock Exchange).

## Disclaimer

This is for educational/research use. Past performance does not guarantee future results. Do your own research before trading.
