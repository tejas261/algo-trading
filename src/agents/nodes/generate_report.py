"""Node: Generate a daily report payload."""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.state import TradingState
from src.strategy.signals import SignalType
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_report_node(state: TradingState) -> dict:
    """Generate a daily report payload summarising the current trading session.

    Aggregates data from market state, signals, execution results,
    positions, anomalies, and portfolio metrics into a single report dict.

    Returns a partial state update with:
        report_payload
    """
    symbol = state.get("symbol", "")
    timeframe = state.get("timeframe", "")
    mode = state.get("mode", "backtest")
    signal = state.get("signal", SignalType.NO_SIGNAL)
    trade_intent = state.get("trade_intent")
    risk_decision = state.get("risk_decision")
    execution_result = state.get("execution_result", {})
    approval_status = state.get("approval_status", "")
    open_positions = state.get("open_positions", [])
    anomaly_flags = state.get("anomaly_flags", [])
    portfolio_state = state.get("portfolio_state", {})
    latest_snapshot = state.get("latest_snapshot", {})
    audit_events = state.get("audit_events", [])

    report_ts = datetime.now(tz=timezone.utc)

    # Build the report
    report_payload: dict = {
        "generated_at": report_ts.isoformat(),
        "symbol": symbol,
        "timeframe": timeframe,
        "mode": mode,
        "market_summary": {
            "latest_close": latest_snapshot.get("close"),
            "latest_high": latest_snapshot.get("high"),
            "latest_low": latest_snapshot.get("low"),
            "latest_volume": latest_snapshot.get("volume"),
            "bar_count": latest_snapshot.get("bar_count", 0),
        },
        "signal": {
            "type": signal.value if isinstance(signal, SignalType) else str(signal),
        },
        "risk": {},
        "trade": {},
        "positions": {
            "open_count": len(open_positions),
            "positions": open_positions,
        },
        "portfolio": {
            "equity": portfolio_state.get("equity"),
            "cash": portfolio_state.get("cash"),
            "total_exposure": portfolio_state.get("total_exposure"),
            "daily_pnl": portfolio_state.get("daily_pnl"),
        },
        "anomalies": {
            "count": len(anomaly_flags),
            "flags": anomaly_flags,
        },
        "audit_event_count": len(audit_events),
    }

    # Risk decision
    if risk_decision is not None:
        report_payload["risk"] = {
            "approved": risk_decision.approved,
            "reasons": risk_decision.reasons,
            "position_size": risk_decision.position_size,
        }

    # Trade intent and execution
    if trade_intent is not None:
        report_payload["trade"] = {
            "symbol": trade_intent.symbol,
            "side": trade_intent.side.value,
            "entry_price": trade_intent.entry_price,
            "stop_loss": trade_intent.stop_loss,
            "position_size": trade_intent.position_size,
            "targets": [
                {"price": t.price, "pct": t.pct}
                for t in trade_intent.targets
            ],
            "total_risk": trade_intent.total_risk,
            "approval_status": approval_status,
            "execution_status": execution_result.get("status"),
        }

    logger.info(
        "Report generated for %s/%s: signal=%s, anomalies=%d, positions=%d",
        symbol, timeframe,
        signal.value if isinstance(signal, SignalType) else signal,
        len(anomaly_flags),
        len(open_positions),
    )

    return {"report_payload": report_payload}
