"""LangGraph state definition for the trading workflow."""

from __future__ import annotations

from typing import TypedDict


class TradingState(TypedDict):
    """Shared state that flows through every node in the trading graph.

    Fields
    ------
    mode:
        Execution mode -- ``"backtest"``, ``"paper"``, or ``"live"``.
    symbol:
        Instrument identifier (e.g. ``"BTC/USDT"``).
    timeframe:
        Candle interval string (e.g. ``"1h"``, ``"15m"``).
    current_timestamp:
        ISO-8601 timestamp of the bar currently being processed.
    market_data:
        Serialized OHLCV data (dict form of a DataFrame slice).
    latest_snapshot:
        Current bar plus computed indicator values.
    indicators:
        Full indicator output for the loaded dataset.
    signal:
        Trade signal produced by the strategy node --
        ``"LONG"``, ``"SHORT"``, or ``"NO_SIGNAL"``.
    trade_intent:
        Structured intent describing the proposed trade
        (symbol, side, quantity, entry, stop, etc.).
    risk_decision:
        Output of the risk engine (approved, reasons, position_size).
    approval_status:
        Human-in-the-loop gate --
        ``"pending"``, ``"approved"``, ``"rejected"``, or ``"modified"``.
    execution_result:
        Fill details returned by the execution adapter.
    open_positions:
        List of serialized open positions.
    open_orders:
        List of serialized open (resting) orders.
    portfolio_state:
        Snapshot of portfolio equity, cash, exposure, etc.
    anomaly_flags:
        Accumulator for anomaly or risk-alert strings.
    audit_events:
        Chronological log of structured audit records.
    report_payload:
        End-of-day (or end-of-backtest) report data.
    error:
        If set, contains the most recent error message.
    step:
        Name of the graph node currently executing.
    should_continue:
        Controls the main loop -- ``False`` halts the graph.
    """

    mode: str
    symbol: str
    timeframe: str
    current_timestamp: str | None
    market_data: dict | None
    latest_snapshot: dict | None
    indicators: dict | None
    signal: str | None
    trade_intent: dict | None
    risk_decision: dict | None
    approval_status: str | None
    execution_result: dict | None
    open_positions: list[dict]
    open_orders: list[dict]
    portfolio_state: dict | None
    anomaly_flags: list[str]
    audit_events: list[dict]
    report_payload: dict | None
    error: str | None
    step: str
    should_continue: bool
