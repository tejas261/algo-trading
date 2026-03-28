"""Node: Request human approval before trade execution."""

from __future__ import annotations

from src.agents.state import TradingState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def request_human_approval_node(state: TradingState) -> dict:
    """Pause for human approval in live mode; auto-approve otherwise.

    In live mode, uses LangGraph's ``interrupt()`` to pause the graph
    and wait for a human to approve or reject the trade.  In paper and
    backtest modes, the trade is auto-approved to allow unattended runs.

    Returns a partial state update with:
        approval_status ("approved" | "rejected" | "pending")
    """
    mode = state.get("mode", "backtest")
    trade_intent = state.get("trade_intent")

    if trade_intent is None:
        logger.info("No trade intent; nothing to approve")
        return {"approval_status": "rejected"}

    # Paper and backtest modes auto-approve
    if mode in ("paper", "backtest"):
        logger.info(
            "Auto-approving trade in %s mode: %s %s @ %.4f",
            mode,
            trade_intent.side.value,
            trade_intent.symbol,
            trade_intent.entry_price,
        )
        return {"approval_status": "approved"}

    # Live mode: use LangGraph interrupt for human-in-the-loop
    if mode == "live":
        try:
            from langgraph.types import interrupt

            approval_request = {
                "type": "trade_approval",
                "symbol": trade_intent.symbol,
                "side": trade_intent.side.value,
                "entry_price": trade_intent.entry_price,
                "stop_loss": trade_intent.stop_loss,
                "position_size": trade_intent.position_size,
                "targets": [
                    {"price": t.price, "pct": t.pct}
                    for t in trade_intent.targets
                ],
                "atr": trade_intent.atr,
                "total_risk": trade_intent.total_risk,
            }

            logger.info("Requesting human approval for trade: %s", approval_request)

            # interrupt() pauses the graph and returns the human's response
            # when the graph is resumed
            response = interrupt(approval_request)

            if isinstance(response, dict):
                approved = response.get("approved", False)
            elif isinstance(response, bool):
                approved = response
            elif isinstance(response, str):
                approved = response.lower() in ("yes", "true", "approved", "approve")
            else:
                logger.warning("Unexpected approval response type: %s", type(response))
                approved = False

            status = "approved" if approved else "rejected"
            logger.info("Human approval decision: %s", status)
            return {"approval_status": status}

        except ImportError:
            logger.warning(
                "langgraph not available; falling back to auto-reject in live mode"
            )
            return {"approval_status": "rejected"}
        except Exception as exc:
            logger.error("Approval request failed: %s", exc, exc_info=True)
            return {"approval_status": "rejected"}

    # Unknown mode -- reject for safety
    logger.warning("Unknown mode '%s'; rejecting trade for safety", mode)
    return {"approval_status": "rejected"}
