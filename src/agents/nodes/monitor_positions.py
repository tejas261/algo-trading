"""Node: Monitor open positions for anomalies."""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.state import TradingState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def monitor_positions_node(state: TradingState) -> dict:
    """Check open positions for anomalies, stale data, and state mismatches.

    Inspects the current open positions and latest market snapshot to
    detect issues like stale price data, missing positions, or unexpected
    state.

    Returns a partial state update with:
        anomaly_flags, open_positions
    """
    open_positions: list[dict] = list(state.get("open_positions", []))
    latest_snapshot = state.get("latest_snapshot", {})
    config = state.get("config", {})
    anomaly_flags: list[str] = list(state.get("anomaly_flags", []))
    execution_result = state.get("execution_result", {})

    monitoring_config = config.get("monitoring", {})
    stale_data_threshold_seconds = int(monitoring_config.get("stale_data_threshold_seconds", 3600))

    # Check for stale market data
    last_ts = latest_snapshot.get("timestamp")
    if last_ts is not None:
        try:
            if hasattr(last_ts, "timestamp"):
                # pandas Timestamp or datetime
                last_epoch = last_ts.timestamp()
            else:
                last_epoch = datetime.fromisoformat(str(last_ts)).timestamp()

            now_epoch = datetime.now(tz=timezone.utc).timestamp()
            staleness = now_epoch - last_epoch

            if staleness > stale_data_threshold_seconds:
                flag = f"stale_market_data: {staleness:.0f}s since last update"
                anomaly_flags.append(flag)
                logger.warning(flag)
        except Exception as exc:
            logger.warning("Could not parse timestamp for staleness check: %s", exc)

    # Check position consistency
    for pos in open_positions:
        symbol = pos.get("symbol", "unknown")
        quantity = pos.get("quantity", 0.0)
        side = pos.get("side", "")

        # Zero-quantity open position is an anomaly
        if quantity <= 0:
            flag = f"zero_quantity_position: {symbol}"
            anomaly_flags.append(flag)
            logger.warning(flag)

        # Missing side
        if not side:
            flag = f"missing_side_on_position: {symbol}"
            anomaly_flags.append(flag)
            logger.warning(flag)

        # Check if entry price is sensible
        entry_price = pos.get("entry_price", 0.0)
        if entry_price <= 0:
            flag = f"invalid_entry_price: {symbol} entry={entry_price}"
            anomaly_flags.append(flag)
            logger.warning(flag)

    # If we just executed a trade, verify it shows up in positions
    if execution_result.get("status") == "executed":
        executed_symbol = execution_result.get("symbol", "")
        if executed_symbol:
            position_symbols = {p.get("symbol") for p in open_positions}
            if executed_symbol not in position_symbols:
                flag = f"executed_trade_not_in_positions: {executed_symbol}"
                anomaly_flags.append(flag)
                logger.warning(flag)

    if not anomaly_flags:
        logger.info("Position monitoring complete: no anomalies detected")
    else:
        logger.warning(
            "Position monitoring found %d anomaly flag(s)",
            len(anomaly_flags),
        )

    return {
        "anomaly_flags": anomaly_flags,
        "open_positions": open_positions,
    }
