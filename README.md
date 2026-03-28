# Algo Trading Framework

A deterministic, config-driven algorithmic trading framework with LLM-orchestrated workflow management via LangGraph.

---

## Architecture Overview

The framework is built around a strict separation between **deterministic strategy logic** and **agentic orchestration**:

- **Strategy is deterministic.** All trade decisions -- direction, entry/exit prices, position sizes, stop losses, and targets -- are computed by pure, side-effect-free functions using technical indicators and config-driven rules. Given the same data and config, the strategy always produces the same output. No randomness, no LLM influence.

- **Orchestration is agentic.** An LLM (via LangGraph) orchestrates the *workflow*: loading data, running the strategy, monitoring positions, requesting human approvals, handling exceptions, reconciling state, and generating reports. The LLM **never** decides trade direction, entry/exit prices, or position sizes.

### Layered Architecture

```
Models -> Strategy -> Engine -> Adapters -> Agents
```

| Layer       | Responsibility                                                  |
|-------------|------------------------------------------------------------------|
| **Models**  | Pydantic data classes (candles, orders, fills, positions, trades) |
| **Strategy**| Indicator computation and signal generation (pure functions)     |
| **Engine**  | Backtester, risk engine, portfolio tracker, cost model, metrics  |
| **Adapters**| Data ingestion (CSV, live feeds) and order execution (paper, broker APIs) |
| **Agents**  | LangGraph nodes that wire the pipeline together with human-in-the-loop gates |

### Data Flow

```
                  +-------------------+
                  |   Market Data     |
                  |   (CSV / Live)    |
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  |   Data Adapter    |
                  | (csv_data_adapter)|
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  |    Strategy       |
                  | (indicators +     |
                  |  signal gen)      |
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  |   Risk Engine     |
                  | (pre-trade checks,|
                  |  position sizing) |
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  |  Human Approval   |
                  | (interrupt in     |
                  |  live mode)       |
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  | Execution Adapter |
                  | (paper / broker)  |
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  |   Portfolio &     |
                  |   Reconciliation  |
                  +--------+----------+
                           |
                           v
                  +-------------------+
                  |   Reporting &     |
                  |   Audit Log       |
                  +-------------------+
```

---

## Project Structure

```
algo-trading/
|-- config/
|   |-- app_config.yaml          # Mode, data source, execution adapter, agent settings
|   |-- strategy_config.yaml     # Indicator periods, entry/exit rules, targets
|   |-- risk_config.yaml         # Position sizing, limits, circuit breakers
|   |-- broker_config.yaml       # Broker API keys, cost models per venue
|
|-- src/
|   |-- models/
|   |   |-- market_data.py       # OHLCVBar dataclass
|   |   |-- order.py             # Order, OrderSide, OrderType, OrderStatus
|   |   |-- fill.py              # Fill (execution record)
|   |   |-- position.py          # Position tracking
|   |   |-- trade.py             # Trade lifecycle (open -> partial exits -> close)
|   |   |-- trade_intent.py      # TradeIntent and PartialTarget
|   |   |-- portfolio.py         # Portfolio snapshot model
|   |   |-- results.py           # BacktestResults and PerformanceMetrics
|   |
|   |-- strategy/
|   |   |-- indicators.py        # EMA, SMA, ATR, ADX, channel extremes
|   |   |-- signals.py           # Signal generation (row-level and vectorized)
|   |   |-- strategy.py          # TrendBreakoutStrategy wrapper
|   |
|   |-- engine/
|   |   |-- backtester.py        # Candle-by-candle backtest engine
|   |   |-- risk.py              # RiskEngine with config-driven pre-trade checks
|   |   |-- portfolio.py         # PortfolioTracker (equity, cash, positions)
|   |   |-- costs.py             # CostModel (fees, slippage, taxes)
|   |   |-- execution_simulator.py  # Intra-candle exit simulation (stops, targets, trailing)
|   |   |-- metrics.py           # Performance metrics (Sharpe, Sortino, drawdown, etc.)
|   |   |-- reporting.py         # Report generation
|   |   |-- reconciler.py        # State reconciliation
|   |
|   |-- adapters/
|   |   |-- data/
|   |   |   |-- base_data_adapter.py   # Abstract data adapter interface
|   |   |   |-- csv_data_adapter.py    # CSV file ingestion
|   |   |-- execution/
|   |       |-- base_execution_adapter.py  # Abstract execution adapter interface
|   |       |-- paper_execution_adapter.py # Paper trading simulator
|   |
|   |-- agents/
|   |   |-- state.py             # TradingState TypedDict (shared graph state)
|   |   |-- schemas.py           # Pydantic schemas for agent I/O
|   |   |-- tools.py             # LangChain tool definitions
|   |   |-- nodes/
|   |       |-- load_market_data.py       # Fetch/load OHLCV data
|   |       |-- run_strategy.py           # Execute strategy pipeline
|   |       |-- run_risk_checks.py        # Evaluate risk rules
|   |       |-- build_trade_intent.py     # Construct TradeIntent from signal
|   |       |-- request_human_approval.py # Human-in-the-loop gate
|   |       |-- execute_trade.py          # Submit orders via adapter
|   |       |-- monitor_positions.py      # Track open positions
|   |       |-- reconcile_state.py        # Sync internal vs venue state
|   |       |-- generate_report.py        # Build performance reports
|   |       |-- emergency_stop.py         # Kill switch / circuit breaker
|   |
|   |-- services/
|   |   |-- notifier.py          # Notification dispatch
|   |   |-- storage.py           # Persistence layer
|   |   |-- audit_logger.py      # Structured audit logging
|   |
|   |-- utils/
|       |-- logger.py            # Logging configuration
|       |-- time_utils.py        # Timestamp helpers
|       |-- validation.py        # Input validation utilities
|
|-- tests/
|   |-- __init__.py
|
|-- requirements.txt
|-- delta_strategy_backtest.py   # Standalone legacy backtest script
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd algo-trading

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

For agent mode (LangGraph orchestration with LLM):

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Usage

### Backtest

Run a backtest on historical data:

```bash
python main.py backtest --data data/sample.csv --symbol NIFTY
```

### Paper Trading

Run in paper-trading mode with simulated execution:

```bash
python main.py paper --data data/sample.csv --symbol NIFTY
```

### Live Trading

Connect to a broker for live execution:

```bash
python main.py live --symbol NIFTY --broker paper
```

### Agent Mode

Run the full LangGraph-orchestrated workflow:

```bash
python main.py agent --mode backtest --data data/sample.csv
```

In agent mode, the LLM orchestrates the pipeline (data loading, strategy execution, risk checks, approvals, reporting) but all trade decisions remain deterministic.

---

## Strategy: 1h Trend Breakout

A trend-following breakout strategy on 1-hour candles.

### Entry Rules

**Long entry** -- all conditions must be true on candle close:

1. Close is above the 200-period EMA (bullish regime)
2. 20-period EMA is above the 50-period EMA (trend alignment)
3. ADX(14) is above 20 (trending market, not ranging)
4. Close breaks above the previous bar's 20-period highest high (breakout)
5. Volume exceeds the 20-period volume SMA (participation confirmation)

**Short entry** -- mirror conditions:

1. Close is below the 200-period EMA (bearish regime)
2. 20-period EMA is below the 50-period EMA
3. ADX(14) is above 20
4. Close breaks below the previous bar's 20-period lowest low
5. Volume exceeds the 20-period volume SMA

Signals are confirmed on candle close. Entry is filled at the next candle's open.

### Exit Rules

**Partial exits** (scaled out in stages):

| Tranche | Size | Target            |
|---------|------|-------------------|
| 1st     | 30%  | 1.0R (1x risk)    |
| 2nd     | 30%  | 2.0R (2x risk)    |

**Trailing stop** on the remaining 40%:

- Trails at 1.5x ATR from the highest high (longs) or lowest low (shorts)

**Initial stop loss:**

- Placed at 1.5x ATR from entry price (configurable via `stop_atr_multiplier`)

### Risk Management Rules

- **Risk per trade:** 1% of equity (fixed-fractional sizing)
- **Leverage:** Up to 2x
- **Max trades per day:** 2
- **Consecutive loss stop:** Trading pauses after 2 consecutive losses
- **Cooldown after stop-out:** 3 candles before re-entry in the same direction
- **ATR volatility filter:** Rejects entries when current ATR exceeds 1.8x its 50-bar rolling mean
- **Daily drawdown limit:** 3% (live mode)
- **Max exposure:** 100% of equity
- **Kill switch:** Manual emergency halt
- **Circuit breaker:** Triggers after 3 consecutive order rejections or slippage exceeding 50 bps over the last 5 trades

---

## Adding a Broker Adapter

To connect a new broker or exchange:

### Step 1: Create the adapter class

Create a new file in `src/adapters/execution/`, e.g. `zerodha_execution_adapter.py`:

```python
from src.adapters.execution.base_execution_adapter import BaseExecutionAdapter
from src.models.order import Order, OrderSide
from src.models.fill import Fill


class ZerodhaExecutionAdapter(BaseExecutionAdapter):

    def __init__(self, config: dict):
        self.api_key = config["api_key"]
        self.api_secret = config["api_secret"]
        # ...

    def connect(self) -> None:
        # Establish API session
        ...

    def disconnect(self) -> None:
        # Close session
        ...

    def health_check(self) -> bool:
        # Ping the API
        ...

    def get_balance(self) -> float:
        # Fetch account balance
        ...

    def get_positions(self) -> list:
        # Fetch open positions
        ...

    def get_open_orders(self) -> list:
        # Fetch resting orders
        ...

    def place_market_order(self, symbol, side, quantity) -> Order:
        ...

    def place_limit_order(self, symbol, side, quantity, price) -> Order:
        ...

    def place_stop_order(self, symbol, side, quantity, stop_price) -> Order:
        ...

    def cancel_order(self, order_id) -> bool:
        ...

    def cancel_all_orders(self, symbol=None) -> bool:
        ...

    def close_position(self, symbol) -> list[Fill]:
        ...

    def close_all_positions(self) -> list[Fill]:
        ...

    def fetch_fills(self, order_id) -> list[Fill]:
        ...

    def reconcile(self) -> dict:
        ...
```

### Step 2: Add broker config

Add a section to `config/broker_config.yaml`:

```yaml
zerodha:
  api_key: ""
  api_secret: ""
  base_url: "https://api.kite.trade"
  cost_model:
    maker_fee_bps: 0.0
    taker_fee_bps: 3.0
    slippage_bps: 2.0
    tax_bps: 10.0
    funding_bps_per_8h: 0.0
```

### Step 3: Register in the adapter factory

Add the new adapter to the factory or registry so it can be selected via `app_config.yaml`:

```yaml
execution:
  adapter: "zerodha"
```

### Step 4: Test

Write integration tests that verify `connect`, `place_market_order`, `cancel_order`, and `reconcile` work against the broker's sandbox/testnet.

---

## Human Approval Flow

In **live mode**, the framework pauses before executing any trade and waits for human approval.

### How it works

1. The `request_human_approval` node detects that `mode == "live"`.
2. It calls LangGraph's `interrupt()` function, which **suspends the graph execution** and surfaces the trade details to the operator.
3. The operator sees the proposed trade: symbol, side, entry price, stop loss, position size, targets, ATR, and total risk.
4. The operator responds with one of:
   - **Approve** (`{"approved": true}`) -- execution proceeds.
   - **Reject** (`{"approved": false}`) -- the trade is skipped.
   - A modified intent (future extension).
5. The graph resumes from the interrupt point with the operator's decision.

### Behavior by mode

| Mode      | Approval behavior   |
|-----------|---------------------|
| Backtest  | Auto-approved       |
| Paper     | Auto-approved       |
| Live      | Requires human approval via `interrupt()` |

If `langgraph` is not installed or the interrupt fails, live-mode trades are **rejected by default** for safety.

---

## Configuration

### app_config.yaml

Controls the overall application behavior:

- `mode`: `backtest`, `paper`, or `live`
- `data.source`: Data adapter to use (`csv`)
- `data.default_path`: Path to the data file
- `execution.adapter`: Which execution adapter (`paper`, `delta`, `groww`)
- `execution.dry_run`: If true, orders are logged but not sent
- `agent.llm_model`: LLM model for agent orchestration (e.g. `gpt-4o-mini`)
- `agent.human_approval_required_live`: Whether live trades require human approval

### strategy_config.yaml

Defines the strategy's indicator periods and trade rules:

- `indicators`: EMA periods (20/50/200), ADX period (14), breakout lookback (20), ATR period (14), volume SMA period (20)
- `entry`: Confirm on close, enter on next open, one position at a time
- `exit.partial_exits`: List of `{pct, target_r}` tranches
- `exit.trailing`: ATR multiplier for the trailing stop on remaining position
- `exit.stop_atr_multiplier`: Initial stop-loss distance in ATR multiples

### risk_config.yaml

Risk management parameters:

- `risk_per_trade_pct`: Equity percentage risked per trade (1.0%)
- `leverage`: Maximum leverage (2.0x)
- `max_trades_per_day`: Daily trade cap (2)
- `consecutive_loss_stop`: Halt after N consecutive losses (2)
- `cooldown_candles_after_stopout`: Bars to wait after stop-out (3)
- `atr_volatility_filter`: Reject trades when ATR spikes above threshold
- `live_controls`: Daily drawdown limit, exposure cap, kill switch, circuit breaker

### broker_config.yaml

Per-broker configuration:

- `initial_capital`, `currency` (paper mode)
- `api_key`, `api_secret`, `base_url` (live brokers)
- `cost_model`: Maker/taker fees (bps), slippage (bps), tax (bps), funding rate

---

## Testing

Run the test suite:

```bash
pytest tests/
```

### What is tested

- Indicator calculations (EMA, SMA, ATR, ADX, channel extremes)
- Signal generation (long, short, no-signal conditions)
- Risk engine checks (cooldown, daily limits, consecutive losses, volatility filter, drawdown, exposure, kill switch)
- Position sizing (fixed-fractional risk model)
- Cost model (fee and slippage application)
- Portfolio tracking (opens, partial exits, closes, equity curve)
- Backtester end-to-end (signal-to-trade lifecycle)
- Execution simulator (stop loss, partial targets, trailing stop)

---

## Limitations and Warnings

**Candle-based execution limitations.** The backtester operates on completed candles. It does not model intra-candle price movement ordering -- when a candle's high and low both cross a stop and a target, the simulator must make assumptions about which was hit first. This can introduce subtle biases.

**Backtest does not equal live performance.** Historical results reflect idealized conditions. Differences in fill quality, timing, order book depth, and market microstructure mean that live performance will diverge from backtest results.

**Overfitting risks.** Optimizing strategy parameters on historical data risks curve-fitting. Out-of-sample validation and walk-forward analysis are strongly recommended before deploying any configuration.

**Slippage and liquidity not fully modeled.** The cost model applies a fixed slippage in basis points. It does not account for order book depth, market impact of large orders, or time-varying liquidity conditions.

**No intra-bar timing.** Signals are evaluated on candle close and entries are filled at the next candle's open. The framework does not support tick-level or sub-candle execution logic.

**Single-asset focus.** The current implementation tracks one position at a time per the strategy config. Multi-asset portfolio management is not yet supported.

---

**This software is NOT financial advice. It is an educational and research tool. Trading involves substantial risk of loss. Past performance is not indicative of future results. Use at your own risk.**

---

## License

MIT
