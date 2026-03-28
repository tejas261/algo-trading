"""Node: Execute a trade via the appropriate execution adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.state import TradingState
from src.adapters.execution.paper_execution_adapter import PaperExecutionAdapter
from src.models.order import OrderSide
from src.utils.logger import get_logger

logger = get_logger(__name__)


def execute_trade_node(state: TradingState) -> dict:
    """Execute the trade described by the current TradeIntent.

    Uses PaperExecutionAdapter for paper/backtest modes.  In live mode,
    the adapter would be swapped for a live broker adapter (not yet
    implemented).

    Returns a partial state update with:
        execution_result, audit_events
    """
    trade_intent = state.get("trade_intent")
    approval_status = state.get("approval_status", "rejected")
    mode = state.get("mode", "backtest")
    config = state.get("config", {})
    audit_events: list[dict] = list(state.get("audit_events", []))

    if trade_intent is None:
        logger.info("No trade intent to execute")
        return {"execution_result": {"status": "skipped", "reason": "no_trade_intent"}}

    if approval_status != "approved":
        logger.info("Trade not approved (status=%s); skipping execution", approval_status)
        audit_events.append({
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event_type": "trade_skipped",
            "reason": f"approval_status={approval_status}",
            "symbol": trade_intent.symbol,
        })
        return {
            "execution_result": {"status": "skipped", "reason": f"not_approved: {approval_status}"},
            "audit_events": audit_events,
        }

    # Select execution adapter based on mode
    try:
        if mode in ("paper", "backtest"):
            execution_config = config.get("execution", {})
            adapter = PaperExecutionAdapter(
                initial_balance=float(execution_config.get("initial_balance", 100_000.0)),
                slippage_pct=float(execution_config.get("slippage_pct", 0.001)),
                commission_pct=float(execution_config.get("commission_pct", 0.001)),
            )
            adapter.connect()

            # Feed current market price so the adapter can fill orders
            latest_snapshot = state.get("latest_snapshot", {})
            current_price = latest_snapshot.get("close", trade_intent.entry_price)
            adapter.update_market_price(
                symbol=trade_intent.symbol,
                price=current_price,
                high=latest_snapshot.get("high", current_price),
                low=latest_snapshot.get("low", current_price),
            )

            # Place the entry order as a market order
            order = adapter.place_market_order(
                symbol=trade_intent.symbol,
                side=trade_intent.side,
                quantity=trade_intent.position_size,
            )

            # Place stop-loss order
            stop_side = OrderSide.SHORT if trade_intent.side == OrderSide.LONG else OrderSide.LONG
            stop_order = adapter.place_stop_order(
                symbol=trade_intent.symbol,
                side=stop_side,
                quantity=trade_intent.position_size,
                stop_price=trade_intent.stop_loss,
            )

            # Retrieve fills for the entry order
            fills = adapter.fetch_fills(str(order.order_id))

            execution_result = {
                "status": "executed",
                "mode": mode,
                "order_id": str(order.order_id),
                "stop_order_id": str(stop_order.order_id),
                "symbol": trade_intent.symbol,
                "side": trade_intent.side.value,
                "quantity": trade_intent.position_size,
                "entry_price": trade_intent.entry_price,
                "stop_loss": trade_intent.stop_loss,
                "fills_count": len(fills),
                "order_status": order.status.value,
                "executed_at": datetime.now(tz=timezone.utc).isoformat(),
            }

            adapter.disconnect()

        elif mode == "live":
            # Live execution adapter would be injected via config
            logger.warning("Live execution adapter not yet implemented")
            execution_result = {
                "status": "error",
                "reason": "live_adapter_not_implemented",
            }
        else:
            logger.error("Unknown execution mode: %s", mode)
            execution_result = {
                "status": "error",
                "reason": f"unknown_mode: {mode}",
            }

    except Exception as exc:
        logger.error("Trade execution failed: %s", exc, exc_info=True)
        execution_result = {
            "status": "error",
            "reason": str(exc),
        }

    # Audit log
    audit_events.append({
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "event_type": "trade_executed" if execution_result.get("status") == "executed" else "trade_execution_failed",
        "data": execution_result,
    })

    logger.info("Execution result: %s", execution_result.get("status"))

    return {
        "execution_result": execution_result,
        "audit_events": audit_events,
    }
