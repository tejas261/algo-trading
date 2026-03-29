# AlgoTrader User Guide

## Setup

```bash
cd "algo trading"
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 1. Get Data

### Crypto (Binance — no API key needed)

```bash
# BTC 4h candles, last 2 years
python scripts/fetch_btc_data.py --start 2024-03-29 --end 2026-03-29

# To change coin or interval, edit scripts/fetch_btc_data.py:
#   SYMBOL = "BTCUSDT"   (or "ETHUSDT", "SOLUSDT", etc.)
#   INTERVAL = "4h"      (or "1h", "1d")
#   OUTPUT_PATH = ... / "data" / "btc_4h.csv"
```

### Indian Equity (Yahoo Finance — no API key needed)

```bash
pip install yfinance

# NIFTY 50 — 1h candles, ~2 years (max for 1h)
python scripts/fetch_india_data.py --symbol NIFTY --interval 1h

# SENSEX — daily candles, 5 years
python scripts/fetch_india_data.py --symbol SENSEX --interval 1d --days 1825

# Individual stock
python scripts/fetch_india_data.py --symbol RELIANCE.NS --interval 1h

# Bank NIFTY
python scripts/fetch_india_data.py --symbol BANKNIFTY --interval 1h
```

**yfinance interval limits:**

| Interval | Max history |
|----------|------------|
| 1m | 7 days |
| 5m, 15m | 60 days |
| 1h | 730 days (~2 years) |
| 1d | unlimited |

Your CSV must have columns: `timestamp, open, high, low, close, volume`.

---

## 2. Configure

### Config structure

```
config/
  base_config.yaml             # shared settings for CRYPTO (edit this first)
  india_base_config.yaml       # shared settings for INDIAN EQUITY
  crypto_config.yaml           # strategy: trend_breakout_v2 → inherits base_config.yaml
  crypto_donchian_config.yaml  # strategy: donchian_trend_55  → inherits base_config.yaml
  india_nifty_config.yaml      # strategy: trend_breakout_v2  → inherits india_base_config.yaml
```

Strategy configs inherit from their base config. To change symbol, timeframe, capital, fees, or risk — edit **only the base config**. Strategy configs only hold indicator and exit parameters.

### To change symbol, timeframe, or capital

Edit `config/base_config.yaml` (crypto) or `config/india_base_config.yaml` (equity):

```yaml
common:
  symbol: "BTCUSDT"         # change to "ETHUSDT", "NIFTY", etc.
  timeframe: "4h"           # change to "1h", "1d", etc.
  initial_capital: 10000.0  # starting equity
  currency: "USDT"          # or "INR"
  market_type: "crypto"     # crypto | equity
  instrument_type: "perpetual"  # spot | perpetual
  leverage: 2.0
```

### To change fees, risk, or slippage

Also in the base config:

```yaml
risk:
  risk_per_trade_pct: 0.75
  max_trades_per_day: 2
  consecutive_loss_stop: 3
  cooldown_candles_after_stopout: 3
  atr_volatility_filter:
    max_atr_ratio: 1.5

paper:
  cost_model:
    taker_fee_bps: 1.0      # 1 for limit orders, 5 for market orders
    base_slippage_bps: 3.0
    tax_bps: 0.0             # set 10.0 for Indian equity (STT/CTT)
```

### To change strategy parameters

Edit the strategy config file:

```yaml
# config/crypto_config.yaml
strategy:
  indicators:
    adx_threshold: 22       # higher = fewer but stronger signals
    channel_period: 20       # breakout lookback (bars)
  exit:
    initial_stop_atr_multiple: 2.5   # wider = fewer stop-outs
    trailing_stop_atr_multiple: 4.0  # wider = lets winners run
```

---

## 3. Pick a Strategy

| Strategy | Config File | Entry | Exit |
|----------|------------|-------|------|
| **trend_breakout_v2** | `crypto_config.yaml` / `india_nifty_config.yaml` | EMA alignment + ADX + Donchian breakout | EMA crossover or ATR trailing stop |
| **donchian_trend_55** | `crypto_donchian_config.yaml` | 55-bar Donchian channel breakout | Opposite channel breakout or ATR trailing stop |

---

## 4. Backtest

All settings (symbol, timeframe, capital) come from the config. Just point at data.

### Crypto

```bash
# trend_breakout_v2
python main.py --config config/crypto_config.yaml backtest --data data/btc_4h.csv

# donchian_trend_55
python main.py --config config/crypto_donchian_config.yaml backtest --data data/btc_4h.csv

# With date filter and custom output directory
python main.py --config config/crypto_config.yaml backtest \
  --data data/btc_4h.csv --start 2025-01-01 --end 2025-12-31 \
  --output-dir output/btc_2025
```

### Indian Equity

```bash
python main.py --config config/india_nifty_config.yaml backtest --data data/nifty_1h.csv
```

### CLI overrides (take precedence over config)

```
--symbol ETHUSDT            # override symbol
--timeframe 1h              # override timeframe
--initial-capital 50000     # override capital
--start 2025-01-01          # filter start date
--end 2025-12-31            # filter end date
--output-dir output/my_run  # where reports go
--verbose                   # debug-level logging
```

---

## 5. Paper Trading

Simulated live execution — runs the LangGraph agent loop over historical data, processing one candle at a time.

### Crypto

```bash
python main.py --config config/crypto_config.yaml paper --data data/btc_4h.csv
```

### Indian Equity

```bash
python main.py --config config/india_nifty_config.yaml paper --data data/nifty_1h.csv
```

---

## 6. Live Trading

Real orders through a broker adapter. All trades require human approval.

```bash
# Crypto via Delta Exchange
python main.py --config config/crypto_config.yaml live --broker delta

# Indian equity via Groww
python main.py --config config/india_nifty_config.yaml live --broker groww
```

You will see a confirmation prompt before any orders are placed. Broker API credentials must be set in the config (or environment variables).

---

## 7. Agent Mode

Full agentic LangGraph workflow — combines data fetch, strategy, risk, execution, and monitoring in a single orchestrated loop.

```bash
# Agent-driven backtest
python main.py --config config/crypto_config.yaml agent --mode backtest --data data/btc_4h.csv

# Agent-driven paper trading
python main.py --config config/crypto_config.yaml agent --mode paper --data data/btc_4h.csv

# Agent-driven live trading (requires human approval)
python main.py --config config/crypto_config.yaml agent --mode live
```

---

## 8. Read the Output

Reports are saved to the `--output-dir` folder (default: `output/`):

| File | Contents |
|------|----------|
| `summary.json` | All performance metrics |
| `trade_log.csv` | Every trade with PnL, exit reason, funding paid, slippage |
| `equity_chart.png` | Equity curve plot |
| `drawdown_chart.png` | Drawdown over time |
| `equity_curve.csv` | Raw equity data points |

---

## 9. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Quick Reference

```
algo trading/
  main.py                          # CLI entry point
  config/
    base_config.yaml               # shared settings — crypto
    india_base_config.yaml         # shared settings — Indian equity
    crypto_config.yaml             # trend_breakout_v2 (crypto)
    crypto_donchian_config.yaml    # donchian_trend_55 (crypto)
    india_nifty_config.yaml        # trend_breakout_v2 (NIFTY)
  data/
    btc_4h.csv                     # 2yr of 4h BTC candles
    nifty_1h.csv                   # 2yr of 1h NIFTY candles
  scripts/
    fetch_btc_data.py              # crypto data fetcher (Binance)
    fetch_india_data.py            # Indian market data fetcher (yfinance)
  output/                          # backtest reports land here
  src/
    strategy/                      # signals, indicators, strategy logic
    engine/                        # backtester, costs, risk, portfolio, metrics
    adapters/                      # data + broker adapters
    agents/                        # LangGraph workflow
    models/                        # Pydantic data models
```
