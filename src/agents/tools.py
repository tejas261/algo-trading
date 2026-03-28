"""LangChain tools that wrap deterministic engine and adapter functions.

Each tool is a thin wrapper returning a string summary suitable for LLM
consumption.  Shared runtime objects (adapters, engines, portfolio tracker)
are stored in the module-level ``_tool_state`` dict and must be initialised
via :func:`init_tool_state` before any tool is invoked.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared mutable state -- populated by init_tool_state()
# ---------------------------------------------------------------------------

_tool_state: dict[str, Any] = {}


def init_tool_state(
    data_dir: str | Path = "data",
    initial_capital: float = 100_000.0,
    risk_config: dict[str, Any] | None = None,
    strategy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Initialise the shared state used by all tools.

    This must be called once at application startup (or at the beginning of a
    backtest run) before any tool is invoked.

    Parameters
    ----------
    data_dir:
        Root directory for CSV data files.
    initial_capital:
        Starting cash balance for the paper execution adapter.
    risk_config:
        Configuration dict forwarded to :class:`RiskEngine`.
    strategy_config:
        Configuration dict forwarded to :class:`TrendBreakoutStrategy`.

    Returns
    -------
    dict
        Reference to the ``_tool_state`` dict (mainly useful for tests).
    """
    from src.adapters.data.csv_data_adapter import CsvDataAdapter
    from src.adapters.execution.paper_execution_adapter import PaperExecutionAdapter
    from src.engine.portfolio import PortfolioTracker
    from src.engine.reporting import ReportGenerator
    from src.engine.risk import RiskEngine
    from src.strategy.strategy import TrendBreakoutStrategy

    data_adapter = CsvDataAdapter(data_dir=data_dir)
    data_adapter.connect()

    exec_adapter = PaperExecutionAdapter(initial_balance=initial_capital)
    exec_adapter.connect()

    _tool_state.update({
        "data_adapter": data_adapter,
        "exec_adapter": exec_adapter,
        "risk_engine": RiskEngine(config=risk_config),
        "strategy": TrendBreakoutStrategy(config=strategy_config),
        "portfolio": PortfolioTracker(initial_capital=initial_capital),
        "report_generator": ReportGenerator(),
        "market_df": None,          # populated by fetch_market_data
        "strategy_df": None,        # populated by run_strategy
        "trades": [],               # accumulated trade records
        "initial_capital": initial_capital,
    })

    logger.info(
        "Tool state initialised (data_dir=%s, capital=%.2f)",
        data_dir,
        initial_capital,
    )
    return _tool_state


def _require_state(key: str) -> Any:
    """Retrieve a value from ``_tool_state`` or raise with a clear message."""
    if key not in _tool_state:
        raise RuntimeError(
            f"Tool state key '{key}' is not initialised. "
            f"Call init_tool_state() before using any tools."
        )
    return _tool_state[key]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def fetch_market_data(
    symbol: str,
    timeframe: str,
    source: str = "csv",
    path: str = "",
) -> str:
    """Load OHLCV market data and return summary statistics.

    Currently supports CSV files only.  The loaded DataFrame is cached in
    tool state so that subsequent tools (e.g. ``run_strategy``) can use it.

    Args:
        symbol: Instrument identifier (e.g. "BTC/USDT").
        timeframe: Candle interval (e.g. "1h", "15m").
        source: Data source type. Only "csv" is supported today.
        path: Optional explicit file path.  If empty, the adapter resolves
              the file from its data directory using the default naming
              convention ``{symbol}_{timeframe}.csv``.

    Returns:
        A human-readable summary of the loaded data.
    """
    if source != "csv":
        return f"Error: unsupported data source '{source}'. Only 'csv' is supported."

    try:
        adapter = _require_state("data_adapter")

        # If an explicit path was provided, register it in the adapter's file map
        if path:
            adapter._file_map[(symbol, timeframe)] = path

        df = adapter.fetch_ohlcv(symbol=symbol, timeframe=timeframe)
        _tool_state["market_df"] = df

        rows = len(df)
        start = str(df["timestamp"].iloc[0])
        end = str(df["timestamp"].iloc[-1])
        close_last = float(df["close"].iloc[-1])
        close_mean = float(df["close"].mean())
        vol_mean = float(df["volume"].mean())

        return (
            f"Loaded {rows} bars for {symbol} ({timeframe}) from {start} to {end}.\n"
            f"Last close: {close_last:.4f} | Mean close: {close_mean:.4f} | "
            f"Mean volume: {vol_mean:.2f}"
        )
    except Exception as exc:
        logger.exception("fetch_market_data failed")
        return f"Error fetching market data: {exc}"


@tool
def run_strategy(symbol: str) -> str:
    """Run TrendBreakoutStrategy on the currently loaded market data.

    The strategy computes indicators and generates vectorized signals.
    The enriched DataFrame is cached in tool state.

    Args:
        symbol: Instrument identifier (used for labelling only -- the
                strategy runs on whatever data was last loaded).

    Returns:
        A summary of the signal on the most recent bar plus key indicators.
    """
    try:
        strategy = _require_state("strategy")
        df = _require_state("market_df")
        if df is None:
            return "Error: no market data loaded. Call fetch_market_data first."

        result_df = strategy.run(df)
        _tool_state["strategy_df"] = result_df

        last = result_df.iloc[-1]
        signal_val = last.get("signal", "NO_SIGNAL")
        # Handle enum values
        signal_str = signal_val.value if hasattr(signal_val, "value") else str(signal_val)

        # Gather key indicator values from the last row
        indicator_keys = [
            "ema_fast", "ema_mid", "ema_slow", "adx", "atr",
            "upper_channel", "lower_channel", "volume_sma",
        ]
        indicators = {}
        for k in indicator_keys:
            if k in last.index:
                val = last[k]
                if val is not None and str(val) != "nan":
                    indicators[k] = round(float(val), 6)

        indicators_str = json.dumps(indicators, indent=2)
        return (
            f"Strategy result for {symbol} (latest bar):\n"
            f"  Signal: {signal_str}\n"
            f"  Close: {last['close']:.4f}\n"
            f"  Indicators:\n{indicators_str}"
        )
    except Exception as exc:
        logger.exception("run_strategy failed")
        return f"Error running strategy: {exc}"


@tool
def check_risk(
    equity: float,
    entry_price: float,
    stop_price: float,
    side: str,
) -> str:
    """Run the risk engine's pre-trade checks and return the decision.

    Args:
        equity: Current account equity.
        entry_price: Expected entry price.
        stop_price: Stop-loss price.
        side: Trade direction -- "LONG" or "SHORT".

    Returns:
        A summary of the risk decision including approval, reasons, and
        approved position size.
    """
    try:
        from src.models.order import OrderSide

        risk_engine = _require_state("risk_engine")

        direction = OrderSide.LONG if side.upper() == "LONG" else OrderSide.SHORT

        context: dict[str, Any] = {
            "equity": equity,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "direction": direction,
        }

        decision = risk_engine.run_all_checks(context)

        return (
            f"Risk decision:\n"
            f"  Approved: {decision.approved}\n"
            f"  Position size: {decision.position_size:.4f}\n"
            f"  Reasons: {'; '.join(decision.reasons) if decision.reasons else 'All checks passed'}"
        )
    except Exception as exc:
        logger.exception("check_risk failed")
        return f"Error in risk check: {exc}"


@tool
def get_portfolio_status() -> str:
    """Return the current portfolio state including equity, cash, and positions.

    Returns:
        A human-readable summary of the portfolio.
    """
    try:
        portfolio = _require_state("portfolio")
        exec_adapter = _require_state("exec_adapter")

        equity = portfolio.get_equity()
        cash = portfolio.cash
        initial = portfolio.initial_capital
        positions = exec_adapter.get_positions()
        open_orders = exec_adapter.get_open_orders()

        pos_summaries = []
        for p in positions:
            qty = getattr(p, "quantity", getattr(p, "current_quantity", 0))
            if qty > 0:
                side = getattr(p, "side", "UNKNOWN")
                side_str = side.value if hasattr(side, "value") else str(side)
                avg_entry = getattr(p, "average_entry_price", getattr(p, "entry_price", 0))
                pos_summaries.append(
                    f"    {p.symbol}: {side_str} qty={qty:.4f} entry={avg_entry:.4f}"
                )

        pnl = equity - initial
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0.0

        lines = [
            f"Portfolio status:",
            f"  Equity: {equity:.2f}",
            f"  Cash: {cash:.2f}",
            f"  Initial capital: {initial:.2f}",
            f"  P&L: {pnl:+.2f} ({pnl_pct:+.2f}%)",
            f"  Open positions: {len(pos_summaries)}",
        ]
        if pos_summaries:
            lines.append("  Positions:")
            lines.extend(pos_summaries)
        lines.append(f"  Open orders: {len(open_orders)}")

        return "\n".join(lines)
    except Exception as exc:
        logger.exception("get_portfolio_status failed")
        return f"Error getting portfolio status: {exc}"


@tool
def place_trade(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    stop_loss: float,
) -> str:
    """Place a trade via the execution adapter (paper mode by default).

    Submits a market order for the entry and a stop order for the stop-loss.

    Args:
        symbol: Instrument identifier.
        side: "LONG" or "SHORT".
        quantity: Number of units to trade.
        price: Reference entry price (market order fills at current price).
        stop_loss: Stop-loss price.

    Returns:
        A summary of the placed orders and fill information.
    """
    try:
        from src.models.order import OrderSide

        exec_adapter = _require_state("exec_adapter")

        entry_side = OrderSide.LONG if side.upper() == "LONG" else OrderSide.SHORT
        stop_side = OrderSide.SHORT if entry_side == OrderSide.LONG else OrderSide.LONG

        # Update market price so the market order can fill immediately
        exec_adapter.update_market_price(symbol, price)

        # Place entry (market order)
        entry_order = exec_adapter.place_market_order(
            symbol=symbol,
            side=entry_side,
            quantity=quantity,
        )

        # Place stop-loss order
        stop_order = exec_adapter.place_stop_order(
            symbol=symbol,
            side=stop_side,
            quantity=quantity,
            stop_price=stop_loss,
        )

        # Collect fill info
        fills = exec_adapter.fetch_fills(entry_order.order_id)
        fill_price = fills[0].price if fills else None
        fill_qty = fills[0].quantity if fills else None

        return (
            f"Trade placed for {symbol}:\n"
            f"  Entry order: {entry_order.order_id} ({entry_order.status.value})\n"
            f"  Fill price: {fill_price:.4f if fill_price else 'pending'}\n"
            f"  Fill quantity: {fill_qty:.4f if fill_qty else 'pending'}\n"
            f"  Stop-loss order: {stop_order.order_id} at {stop_loss:.4f}\n"
            f"  Side: {side.upper()}"
        )
    except Exception as exc:
        logger.exception("place_trade failed")
        return f"Error placing trade: {exc}"


@tool
def get_open_positions() -> str:
    """Return all currently open positions.

    Returns:
        A summary of each open position including symbol, side, quantity,
        entry price, and unrealised P&L.
    """
    try:
        exec_adapter = _require_state("exec_adapter")
        positions = exec_adapter.get_positions()

        open_pos = []
        for p in positions:
            qty = getattr(p, "quantity", getattr(p, "current_quantity", 0))
            if qty > 0:
                open_pos.append(p)

        if not open_pos:
            return "No open positions."

        lines = [f"Open positions ({len(open_pos)}):"]
        for p in open_pos:
            side = getattr(p, "side", "UNKNOWN")
            side_str = side.value if hasattr(side, "value") else str(side)
            avg_entry = getattr(p, "average_entry_price", getattr(p, "entry_price", 0))
            unrealized = getattr(p, "unrealized_pnl", 0.0)
            lines.append(
                f"  {p.symbol}: {side_str} qty={qty:.4f} "
                f"entry={avg_entry:.4f} unrealized_pnl={unrealized:+.4f}"
            )
        return "\n".join(lines)
    except Exception as exc:
        logger.exception("get_open_positions failed")
        return f"Error getting open positions: {exc}"


@tool
def cancel_all_orders_tool(symbol: str) -> str:
    """Cancel all open orders for a given symbol.

    Args:
        symbol: Instrument identifier whose orders should be cancelled.

    Returns:
        Confirmation message.
    """
    try:
        exec_adapter = _require_state("exec_adapter")

        orders_before = exec_adapter.get_open_orders()
        matching = [o for o in orders_before if o.symbol == symbol]

        exec_adapter.cancel_all_orders(symbol=symbol)

        return (
            f"Cancelled {len(matching)} open order(s) for {symbol}."
        )
    except Exception as exc:
        logger.exception("cancel_all_orders_tool failed")
        return f"Error cancelling orders: {exc}"


@tool
def emergency_stop_tool() -> str:
    """Emergency stop: flatten all positions and cancel all open orders.

    This is the nuclear option -- use only when immediate risk reduction
    is required.

    Returns:
        Summary of actions taken.
    """
    try:
        exec_adapter = _require_state("exec_adapter")
        risk_engine = _require_state("risk_engine")

        # Activate kill switch to block new trades
        risk_engine.set_kill_switch(active=True)

        # Cancel all resting orders
        exec_adapter.cancel_all_orders()

        # Close every open position
        close_fills = exec_adapter.close_all_positions()

        positions_after = exec_adapter.get_positions()
        still_open = sum(
            1 for p in positions_after
            if getattr(p, "quantity", 0) > 0
        )

        return (
            f"EMERGENCY STOP executed:\n"
            f"  Kill switch: ACTIVATED\n"
            f"  All orders cancelled\n"
            f"  Positions closed: {len(close_fills)} fill(s)\n"
            f"  Remaining open positions: {still_open}\n"
            f"  WARNING: Kill switch is active -- no new trades will be accepted "
            f"until it is manually reset."
        )
    except Exception as exc:
        logger.exception("emergency_stop_tool failed")
        return f"Error during emergency stop: {exc}"


@tool
def generate_report_tool() -> str:
    """Generate an end-of-day summary report.

    Gathers portfolio metrics, trade counts, and P&L information into
    a structured report string.

    Returns:
        A multi-line report summary.
    """
    try:
        portfolio = _require_state("portfolio")
        exec_adapter = _require_state("exec_adapter")
        initial_capital = _require_state("initial_capital")

        equity = portfolio.get_equity()
        cash = portfolio.cash
        positions = exec_adapter.get_positions()
        open_orders = exec_adapter.get_open_orders()

        open_count = sum(
            1 for p in positions
            if getattr(p, "quantity", getattr(p, "current_quantity", 0)) > 0
        )

        # Reconcile adapter state
        recon = exec_adapter.reconcile()
        realized_pnl = recon.get("realized_pnl", 0.0)

        # Compute drawdown from equity curve if available
        equity_curve = portfolio.get_equity_curve()
        drawdown_pct = 0.0
        if equity_curve:
            peak = max(eq for _, eq in equity_curve)
            if peak > 0:
                drawdown_pct = ((peak - equity) / peak) * 100.0

        pnl_total = equity - initial_capital
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        lines = [
            "=" * 50,
            f"  END-OF-DAY REPORT  --  {now_str}",
            "=" * 50,
            f"  Equity:            {equity:>12.2f}",
            f"  Cash:              {cash:>12.2f}",
            f"  Initial capital:   {initial_capital:>12.2f}",
            f"  Total P&L:         {pnl_total:>+12.2f}",
            f"  Realized P&L:      {realized_pnl:>+12.2f}",
            f"  Drawdown:          {drawdown_pct:>11.2f}%",
            f"  Open positions:    {open_count:>12d}",
            f"  Open orders:       {len(open_orders):>12d}",
            "=" * 50,
        ]

        return "\n".join(lines)
    except Exception as exc:
        logger.exception("generate_report_tool failed")
        return f"Error generating report: {exc}"
