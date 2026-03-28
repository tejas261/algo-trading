"""LangGraph workflow orchestration for the algo-trading framework.

Defines the trading state graph, conditional routing functions, and
convenience helpers for running trading cycles in different modes.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import StateGraph, START, END
from langgraph.graph.graph import CompiledGraph
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import TradingState
from src.agents.nodes import (
    load_market_data_node,
    run_strategy_node,
    run_risk_checks_node,
    build_trade_intent_node,
    request_human_approval_node,
    execute_trade_node,
    monitor_positions_node,
    reconcile_state_node,
    generate_report_node,
    emergency_stop_node,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Critical anomaly keywords that trigger the emergency-stop path
# ---------------------------------------------------------------------------
_CRITICAL_ANOMALY_PREFIXES = frozenset({
    "stale_market_data",
    "executed_trade_not_in_positions",
    "zero_quantity_position",
    "invalid_entry_price",
    "missing_side_on_position",
})


def _is_critical_anomaly(flag: str) -> bool:
    """Return True if *flag* starts with a known critical prefix."""
    return any(flag.startswith(prefix) for prefix in _CRITICAL_ANOMALY_PREFIXES)


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------


def route_after_risk(state: TradingState) -> str:
    """Decide whether to proceed with a trade or skip to monitoring.

    Routes to ``monitor_positions`` when:
      - The signal is ``NO_SIGNAL``
      - Risk checks did not approve the trade (``should_continue`` is False)

    Otherwise routes to ``build_trade_intent``.
    """
    signal = state.get("signal")
    should_continue = state.get("should_continue", False)

    if signal == "NO_SIGNAL" or signal is None:
        logger.info("route_after_risk -> monitor_positions (no signal)")
        return "monitor_positions"

    # Also check the enum form if the strategy node returned an enum value
    if hasattr(signal, "value") and signal.value == "NO_SIGNAL":
        logger.info("route_after_risk -> monitor_positions (no signal enum)")
        return "monitor_positions"

    if not should_continue:
        logger.info("route_after_risk -> monitor_positions (risk not approved)")
        return "monitor_positions"

    logger.info("route_after_risk -> build_trade_intent")
    return "build_trade_intent"


def route_approval(state: TradingState) -> str:
    """Decide whether to gate on human approval.

    In ``live`` mode the graph pauses at the human-approval node.
    In all other modes (``paper``, ``backtest``) the graph skips straight
    to execution, relying on auto-approval within the approval node or
    bypassing it entirely.
    """
    mode = state.get("mode", "backtest")

    if mode == "live":
        logger.info("route_approval -> request_human_approval (live mode)")
        return "request_human_approval"

    logger.info("route_approval -> execute_trade (%s mode)", mode)
    return "execute_trade"


def route_after_approval(state: TradingState) -> str:
    """Route based on the human approval decision.

    ``approved``  -> ``execute_trade``
    anything else -> ``monitor_positions``
    """
    approval = state.get("approval_status", "rejected")

    if approval == "approved":
        logger.info("route_after_approval -> execute_trade (approved)")
        return "execute_trade"

    logger.info("route_after_approval -> monitor_positions (status=%s)", approval)
    return "monitor_positions"


def route_after_monitor(state: TradingState) -> str:
    """Route based on anomaly flags detected during position monitoring.

    If any anomaly flag is classified as *critical*, the graph is routed
    to the ``emergency_stop`` node.  Otherwise it proceeds normally to
    ``reconcile_state``.
    """
    anomaly_flags: list[str] = state.get("anomaly_flags", [])

    critical = [f for f in anomaly_flags if _is_critical_anomaly(f)]

    if critical:
        logger.warning(
            "route_after_monitor -> emergency_stop (%d critical flag(s))",
            len(critical),
        )
        return "emergency_stop"

    logger.info("route_after_monitor -> reconcile_state")
    return "reconcile_state"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_trading_graph() -> CompiledGraph:
    """Build and compile the trading state graph with a memory checkpointer.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to be invoked.
    """
    graph = StateGraph(TradingState)

    # -- Register nodes --------------------------------------------------------
    graph.add_node("load_market_data", load_market_data_node)
    graph.add_node("run_strategy", run_strategy_node)
    graph.add_node("run_risk_checks", run_risk_checks_node)
    graph.add_node("build_trade_intent", build_trade_intent_node)
    graph.add_node("request_human_approval", request_human_approval_node)
    graph.add_node("execute_trade", execute_trade_node)
    graph.add_node("monitor_positions", monitor_positions_node)
    graph.add_node("reconcile_state", reconcile_state_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("emergency_stop", emergency_stop_node)

    # -- Linear edges ----------------------------------------------------------
    graph.add_edge(START, "load_market_data")
    graph.add_edge("load_market_data", "run_strategy")
    graph.add_edge("run_strategy", "run_risk_checks")
    graph.add_edge("execute_trade", "monitor_positions")
    graph.add_edge("reconcile_state", "generate_report")
    graph.add_edge("generate_report", END)
    graph.add_edge("emergency_stop", "generate_report")

    # -- Conditional edges -----------------------------------------------------
    graph.add_conditional_edges(
        "run_risk_checks",
        route_after_risk,
        {
            "monitor_positions": "monitor_positions",
            "build_trade_intent": "build_trade_intent",
        },
    )

    graph.add_conditional_edges(
        "build_trade_intent",
        route_approval,
        {
            "request_human_approval": "request_human_approval",
            "execute_trade": "execute_trade",
        },
    )

    graph.add_conditional_edges(
        "request_human_approval",
        route_after_approval,
        {
            "execute_trade": "execute_trade",
            "monitor_positions": "monitor_positions",
        },
    )

    graph.add_conditional_edges(
        "monitor_positions",
        route_after_monitor,
        {
            "emergency_stop": "emergency_stop",
            "reconcile_state": "reconcile_state",
        },
    )

    # -- Compile with checkpointer ---------------------------------------------
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


# ---------------------------------------------------------------------------
# State initialisation helper
# ---------------------------------------------------------------------------


def create_initial_state(
    mode: str,
    symbol: str,
    timeframe: str,
    **kwargs: Any,
) -> TradingState:
    """Create a properly initialised ``TradingState`` dict.

    All fields are set to safe defaults and can be overridden via
    *kwargs*.

    Parameters
    ----------
    mode:
        ``"backtest"``, ``"paper"``, or ``"live"``.
    symbol:
        Instrument identifier (e.g. ``"BTC/USDT"``).
    timeframe:
        Candle interval string (e.g. ``"1h"``).
    **kwargs:
        Arbitrary overrides merged into the returned state.

    Returns
    -------
    TradingState
        A fully populated state dictionary.
    """
    state: TradingState = {
        "mode": mode,
        "symbol": symbol,
        "timeframe": timeframe,
        "current_timestamp": None,
        "market_data": None,
        "latest_snapshot": None,
        "indicators": None,
        "signal": None,
        "trade_intent": None,
        "risk_decision": None,
        "approval_status": None,
        "execution_result": None,
        "open_positions": [],
        "open_orders": [],
        "portfolio_state": None,
        "anomaly_flags": [],
        "audit_events": [],
        "report_payload": None,
        "error": None,
        "step": "init",
        "should_continue": True,
    }
    state.update(kwargs)  # type: ignore[typeddict-item]
    return state


# ---------------------------------------------------------------------------
# High-level invocation helpers
# ---------------------------------------------------------------------------


def run_trading_cycle(
    mode: str,
    symbol: str,
    timeframe: str,
    *,
    config_path: str | None = None,
    data_path: str | None = None,
) -> dict:
    """Run a single trading cycle through the full graph.

    For **live** mode the graph may pause at the human-approval node.
    When that happens this function catches the ``GraphInterrupt``,
    logs the pending approval payload, and returns the intermediate
    state so the caller can resume after obtaining human input.

    Parameters
    ----------
    mode:
        ``"backtest"``, ``"paper"``, or ``"live"``.
    symbol:
        Instrument identifier.
    timeframe:
        Candle interval string.
    config_path:
        Optional path to a YAML/JSON config file injected into state.
    data_path:
        Optional path to historical data (CSV) used in backtest mode.

    Returns
    -------
    dict
        The final (or interrupted) graph state.
    """
    graph = build_trading_graph()

    extra_kwargs: dict[str, Any] = {}
    if config_path is not None:
        extra_kwargs["config_path"] = config_path
    if data_path is not None:
        extra_kwargs["data_path"] = data_path

    initial_state = create_initial_state(mode, symbol, timeframe, **extra_kwargs)

    thread_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}

    if mode == "live":
        return _run_with_interrupt_handling(graph, initial_state, run_config)

    # Paper / backtest -- straight invocation
    result = graph.invoke(initial_state, config=run_config)
    logger.info("Trading cycle complete for %s (%s mode)", symbol, mode)
    return result


def _run_with_interrupt_handling(
    graph: CompiledGraph,
    initial_state: TradingState,
    run_config: dict,
) -> dict:
    """Invoke the graph, handling the interrupt/resume pattern for live mode.

    If the graph pauses at the human-approval interrupt, this function
    returns a dict containing the intermediate ``state`` plus metadata
    (``thread_id``, ``interrupted``, ``approval_payload``) so the
    caller can inspect the trade proposal, collect a human decision,
    and later call ``resume_after_approval`` to continue the graph.
    """
    try:
        result = graph.invoke(initial_state, config=run_config)
        logger.info("Live trading cycle completed without interrupt")
        return result
    except Exception as exc:
        # LangGraph raises GraphInterrupt when interrupt() is called
        exc_type_name = type(exc).__name__
        if exc_type_name == "GraphInterrupt":
            logger.info("Graph interrupted for human approval")
            # Retrieve current state from the checkpointer
            state_snapshot = graph.get_state(run_config)
            return {
                "interrupted": True,
                "thread_id": run_config["configurable"]["thread_id"],
                "state": state_snapshot.values if state_snapshot else {},
                "approval_payload": getattr(exc, "value", None),
                "run_config": run_config,
            }
        raise


def resume_after_approval(
    graph: CompiledGraph,
    run_config: dict,
    approved: bool,
) -> dict:
    """Resume a previously interrupted graph after human approval.

    Parameters
    ----------
    graph:
        The same compiled graph instance (or a freshly built one sharing
        the same checkpointer).
    run_config:
        The run config dict returned in the interrupted result
        (must contain the original ``thread_id``).
    approved:
        Human decision -- ``True`` to approve, ``False`` to reject.

    Returns
    -------
    dict
        The final graph state after completion.
    """
    from langgraph.types import Command

    result = graph.invoke(
        Command(resume={"approved": approved}),
        config=run_config,
    )
    logger.info(
        "Resumed graph after approval (approved=%s); cycle complete",
        approved,
    )
    return result


# ---------------------------------------------------------------------------
# Backtest-specific entry point
# ---------------------------------------------------------------------------


def run_backtest_graph(
    symbol: str,
    timeframe: str,
    data_path: str,
    *,
    config_path: str | None = None,
) -> dict:
    """Run the trading graph in backtest mode.

    This is a thin wrapper around :func:`run_trading_cycle` that locks
    the mode to ``"backtest"`` and requires a ``data_path``.

    Parameters
    ----------
    symbol:
        Instrument identifier.
    timeframe:
        Candle interval string (e.g. ``"1h"``).
    data_path:
        Path to the historical OHLCV CSV file.
    config_path:
        Optional path to a config file.

    Returns
    -------
    dict
        The final graph state including the ``report_payload``.
    """
    if not data_path:
        raise ValueError("data_path is required for backtest mode")

    result = run_trading_cycle(
        mode="backtest",
        symbol=symbol,
        timeframe=timeframe,
        config_path=config_path,
        data_path=data_path,
    )
    logger.info(
        "Backtest complete for %s (%s). Report: %s",
        symbol,
        timeframe,
        "generated" if result.get("report_payload") else "not generated",
    )
    return result
