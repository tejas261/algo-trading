"""Node: Reconcile internal state with execution adapter state."""

from __future__ import annotations

from src.agents.state import TradingState
from src.engine.reconciler import Reconciler, Severity
from src.utils.logger import get_logger

logger = get_logger(__name__)


def reconcile_state_node(state: TradingState) -> dict:
    """Run the reconciler between internal state and adapter-reported state.

    Compares positions, orders, and balances to detect discrepancies.
    Critical discrepancies are added to anomaly_flags.

    Returns a partial state update with:
        portfolio_state, anomaly_flags
    """
    portfolio_state: dict = dict(state.get("portfolio_state", {}))
    anomaly_flags: list[str] = list(state.get("anomaly_flags", []))
    config = state.get("config", {})
    execution_result = state.get("execution_result", {})

    reconciler = Reconciler()

    # Build internal positions dict from open_positions
    internal_positions = {}
    open_positions = state.get("open_positions", [])
    for pos in open_positions:
        symbol = pos.get("symbol")
        if symbol:
            internal_positions[symbol] = pos

    # Get broker-reported state from execution result or portfolio
    broker_positions = portfolio_state.get("broker_positions", {})
    broker_orders = portfolio_state.get("broker_orders", [])
    broker_balance = float(portfolio_state.get("broker_balance", 0.0))
    internal_balance = float(portfolio_state.get("internal_balance", 0.0))

    # Reconcile balance
    tolerance_pct = float(config.get("reconciliation", {}).get("balance_tolerance_pct", 0.1))
    balance_ok = True
    if internal_balance > 0 and broker_balance > 0:
        balance_ok = reconciler.reconcile_balance(
            internal_balance=internal_balance,
            broker_balance=broker_balance,
            tolerance_pct=tolerance_pct,
        )
        if not balance_ok:
            flag = (
                f"balance_mismatch: internal={internal_balance:.2f} "
                f"broker={broker_balance:.2f}"
            )
            anomaly_flags.append(flag)
            logger.warning(flag)

    # Position discrepancies -- only run if we have broker data to compare
    position_discrepancies: list[dict] = []
    if broker_positions:
        # Reconciler expects Position objects for internal; we pass dicts
        # so we do a simplified comparison here
        all_symbols = set(internal_positions.keys()) | set(broker_positions.keys())
        for symbol in sorted(all_symbols):
            internal = internal_positions.get(symbol)
            broker = broker_positions.get(symbol)

            if internal is not None and broker is None:
                disc = {
                    "field": f"{symbol}.exists",
                    "internal_value": True,
                    "broker_value": False,
                    "severity": Severity.CRITICAL.value,
                }
                position_discrepancies.append(disc)
            elif internal is None and broker is not None:
                disc = {
                    "field": f"{symbol}.exists",
                    "internal_value": False,
                    "broker_value": True,
                    "severity": Severity.CRITICAL.value,
                }
                position_discrepancies.append(disc)
            elif internal is not None and broker is not None:
                # Compare quantities
                int_qty = float(internal.get("quantity", 0.0))
                brk_qty = float(broker.get("quantity", 0.0))
                if abs(int_qty - brk_qty) > 1e-6:
                    disc = {
                        "field": f"{symbol}.quantity",
                        "internal_value": int_qty,
                        "broker_value": brk_qty,
                        "severity": Severity.CRITICAL.value if abs(int_qty - brk_qty) / max(int_qty, 1e-9) > 0.01 else Severity.WARNING.value,
                    }
                    position_discrepancies.append(disc)

    # Add critical discrepancies to anomaly flags
    for disc in position_discrepancies:
        if disc.get("severity") == Severity.CRITICAL.value:
            flag = f"reconciliation_critical: {disc['field']} internal={disc['internal_value']} broker={disc['broker_value']}"
            anomaly_flags.append(flag)
            logger.warning(flag)

    # Update portfolio state with reconciliation results
    portfolio_state["last_reconciliation"] = {
        "balance_ok": balance_ok,
        "position_discrepancies": position_discrepancies,
        "discrepancy_count": len(position_discrepancies),
    }

    if not position_discrepancies and balance_ok:
        logger.info("Reconciliation complete: no discrepancies found")
    else:
        logger.warning(
            "Reconciliation found %d position discrepancies, balance_ok=%s",
            len(position_discrepancies), balance_ok,
        )

    return {
        "portfolio_state": portfolio_state,
        "anomaly_flags": anomaly_flags,
    }
