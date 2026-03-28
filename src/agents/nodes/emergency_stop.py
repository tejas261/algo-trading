"""Node: Emergency stop -- flatten all positions and cancel all orders."""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.state import TradingState
from src.adapters.execution.paper_execution_adapter import PaperExecutionAdapter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def emergency_stop_node(state: TradingState) -> dict:
    """Flatten all positions, cancel all orders, and halt the workflow.

    This is the panic button.  It closes every open position at market,
    cancels all resting orders, and sets should_continue=False to
    terminate the graph.

    Returns a partial state update with:
        should_continue, audit_events, open_positions, anomaly_flags
    """
    mode = state.get("mode", "backtest")
    config = state.get("config", {})
    audit_events: list[dict] = list(state.get("audit_events", []))
    anomaly_flags: list[str] = list(state.get("anomaly_flags", []))

    emergency_ts = datetime.now(tz=timezone.utc)
    logger.critical("EMERGENCY STOP triggered at %s", emergency_ts.isoformat())

    close_results: list[dict] = []
    cancel_success = False

    try:
        if mode in ("paper", "backtest"):
            execution_config = config.get("execution", {})
            adapter = PaperExecutionAdapter(
                initial_balance=float(execution_config.get("initial_balance", 100_000.0)),
                slippage_pct=float(execution_config.get("slippage_pct", 0.001)),
                commission_pct=float(execution_config.get("commission_pct", 0.001)),
            )
            adapter.connect()

            # Feed latest price so stop/close orders can fill
            latest_snapshot = state.get("latest_snapshot", {})
            current_price = latest_snapshot.get("close", 0.0)
            symbol = state.get("symbol", "")
            if current_price > 0 and symbol:
                adapter.update_market_price(
                    symbol=symbol,
                    price=current_price,
                    high=latest_snapshot.get("high", current_price),
                    low=latest_snapshot.get("low", current_price),
                )

            # Cancel all open orders
            cancel_success = adapter.cancel_all_orders()
            logger.info("Cancelled all orders: success=%s", cancel_success)

            # Close all positions
            fills = adapter.close_all_positions()
            for fill in fills:
                close_results.append({
                    "fill_id": str(fill.fill_id),
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "quantity": fill.quantity,
                    "price": fill.price,
                })

            adapter.disconnect()

        elif mode == "live":
            logger.error(
                "Live emergency stop requires a live adapter -- not yet implemented. "
                "Manual intervention required."
            )
            anomaly_flags.append("emergency_stop_live_not_implemented")

    except Exception as exc:
        logger.critical("Emergency stop execution failed: %s", exc, exc_info=True)
        anomaly_flags.append(f"emergency_stop_error: {exc}")

    # Log the emergency event
    audit_events.append({
        "timestamp": emergency_ts.isoformat(),
        "event_type": "emergency_stop",
        "data": {
            "mode": mode,
            "orders_cancelled": cancel_success,
            "positions_closed": len(close_results),
            "close_fills": close_results,
            "anomaly_flags_at_trigger": list(anomaly_flags),
        },
    })

    logger.critical(
        "Emergency stop complete: cancelled_orders=%s, closed_positions=%d",
        cancel_success, len(close_results),
    )

    return {
        "should_continue": False,
        "open_positions": [],
        "audit_events": audit_events,
        "anomaly_flags": anomaly_flags,
    }
