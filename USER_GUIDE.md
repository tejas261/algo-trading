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

Fetch BTC/USDT candles from Binance (no API key needed):

```bash
# Edit scripts/fetch_btc_data.py to set INTERVAL and OUTPUT_PATH, then:
python scripts/fetch_btc_data.py --start 2024-01-01 --end 2026-03-28
```

Current defaults: `INTERVAL = "4h"`, output = `data/btc_4h.csv`.

Your CSV must have columns: `timestamp, open, high, low, close, volume`.

---

## 2. Configure

There are two config files:

| File | What it controls |
|------|-----------------|
| `config/base_config.yaml` | **Everything shared** — symbol, timeframe, capital, leverage, fees, risk, instrument spec |
| `config/crypto_config.yaml` or `config/crypto_donchian_config.yaml` | **Strategy-specific** — indicator params, entry/exit rules |

### To change symbol, timeframe, or capital

Edit **only** `config/base_config.yaml`:

```yaml
common:
  symbol: "BTCUSDT"         # change to "ETHUSDT", etc.
  timeframe: "4h"           # change to "1h", "1d", etc.
  initial_capital: 10000.0  # starting equity
  currency: "USDT"
  market_type: "crypto"     # crypto | equity
  instrument_type: "perpetual"  # spot | perpetual
  leverage: 2.0
```

These values automatically propagate to all other config sections. No need to change anything else.

### To change fees, risk, or slippage

Also in `config/base_config.yaml`:

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
```

### To change strategy parameters

Edit the strategy config file:

```yaml
# config/crypto_config.yaml (trend_breakout_v2)
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
| **trend_breakout_v2** | `config/crypto_config.yaml` | EMA alignment + ADX + Donchian breakout | EMA crossover or ATR trailing stop |
| **donchian_trend_55** | `config/crypto_donchian_config.yaml` | 55-bar Donchian channel breakout | Opposite channel breakout or ATR trailing stop |

---

## 4. Run a Backtest

```bash
# trend_breakout_v2 (symbol, timeframe, capital all come from base_config.yaml)
python main.py --config config/crypto_config.yaml backtest --data data/btc_4h.csv

# donchian_trend_55
python main.py --config config/crypto_donchian_config.yaml backtest --data data/btc_4h.csv
```

### Optional CLI overrides (take precedence over config)

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

## 5. Read the Output

Reports are saved to the `--output-dir` folder (default: `output/`):

| File | Contents |
|------|----------|
| `summary.json` | All performance metrics |
| `trade_log.csv` | Every trade with PnL, exit reason, funding paid, slippage |
| `equity_chart.png` | Equity curve plot |
| `drawdown_chart.png` | Drawdown over time |
| `equity_curve.csv` | Raw equity data points |

---

## 6. Other Modes

### Paper Trading (simulated live)

```bash
python main.py --config config/crypto_config.yaml paper --data data/btc_4h.csv
```

### Live Trading (real orders, requires broker adapter)

```bash
python main.py --config config/crypto_config.yaml live --broker delta
```

Requires confirmation prompt. All trades need human approval.

### Agent Mode (LangGraph orchestration)

```bash
python main.py --config config/crypto_config.yaml agent --mode backtest --data data/btc_4h.csv
```

---

## 7. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Quick Reference

```
algo trading/
  main.py                          # CLI entry point
  config/
    base_config.yaml               # shared settings (edit this first)
    crypto_config.yaml             # trend_breakout_v2 strategy
    crypto_donchian_config.yaml    # donchian_trend_55 strategy
  data/
    btc_4h.csv                     # 2yr of 4h BTC candles
    btc_1h.csv                     # 3mo of 1h BTC candles
  scripts/
    fetch_btc_data.py              # data fetcher (Binance)
  output/                          # backtest reports land here
  src/
    strategy/                      # signals, indicators, strategy logic
    engine/                        # backtester, costs, risk, portfolio, metrics
    adapters/                      # data + broker adapters
    agents/                        # LangGraph workflow
    models/                        # Pydantic data models
```
