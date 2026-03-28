"""Reconciliation between internal state and broker-reported state.

Detects discrepancies in positions, orders, and balances to ensure
the internal book stays in sync with the exchange.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Sequence

from src.models.order import Order, OrderStatus
from src.models.position import Position
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """Severity level for a reconciliation discrepancy."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


def _discrepancy(
    field: str,
    internal_value: Any,
    broker_value: Any,
    severity: str | Severity = Severity.WARNING,
) -> dict[str, Any]:
    """Build a standardised discrepancy record."""
    return {
        "field": field,
        "internal_value": internal_value,
        "broker_value": broker_value,
        "severity": severity.value if isinstance(severity, Severity) else severity,
    }


class Reconciler:
    """Compares internal trading state against broker-reported state.

    All ``reconcile_*`` methods return lists of discrepancy dicts so the
    caller can decide how to handle them (log, alert, auto-correct, etc.).
    """

    def reconcile_positions(
        self,
        internal_positions: dict[str, Position],
        broker_positions: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compare internal position book against broker-reported positions.

        ``broker_positions`` is a dict keyed by symbol, where each value
        contains at least ``quantity`` (float) and ``side`` (str).
        Optionally ``entry_price`` (float).

        Args:
            internal_positions: Our internal position state.
            broker_positions: Broker-reported positions.

        Returns:
            List of discrepancy dicts.
        """
        discrepancies: list[dict[str, Any]] = []

        all_symbols = set(internal_positions.keys()) | set(broker_positions.keys())

        for symbol in sorted(all_symbols):
            internal = internal_positions.get(symbol)
            broker = broker_positions.get(symbol)

            # Position exists internally but not at broker
            if internal is not None and broker is None:
                discrepancies.append(_discrepancy(
                    field=f"{symbol}.exists",
                    internal_value=True,
                    broker_value=False,
                    severity=Severity.CRITICAL,
                ))
                continue

            # Position exists at broker but not internally
            if internal is None and broker is not None:
                discrepancies.append(_discrepancy(
                    field=f"{symbol}.exists",
                    internal_value=False,
                    broker_value=True,
                    severity=Severity.CRITICAL,
                ))
                continue

            # Both exist -- compare fields
            assert internal is not None and broker is not None

            # Side
            broker_side = broker.get("side", "")
            if internal.side.value != broker_side:
                discrepancies.append(_discrepancy(
                    field=f"{symbol}.side",
                    internal_value=internal.side.value,
                    broker_value=broker_side,
                    severity=Severity.CRITICAL,
                ))

            # Quantity
            broker_qty = float(broker.get("quantity", 0.0))
            if abs(internal.current_quantity - broker_qty) > 1e-6:
                severity = Severity.CRITICAL if abs(internal.current_quantity - broker_qty) / max(internal.current_quantity, 1e-9) > 0.01 else Severity.WARNING
                discrepancies.append(_discrepancy(
                    field=f"{symbol}.quantity",
                    internal_value=internal.current_quantity,
                    broker_value=broker_qty,
                    severity=severity,
                ))

            # Entry price (if provided by broker)
            broker_entry = broker.get("entry_price")
            if broker_entry is not None:
                broker_entry = float(broker_entry)
                if abs(internal.entry_price - broker_entry) > 1e-6:
                    discrepancies.append(_discrepancy(
                        field=f"{symbol}.entry_price",
                        internal_value=internal.entry_price,
                        broker_value=broker_entry,
                        severity=Severity.WARNING,
                    ))

        if discrepancies:
            logger.warning(
                "Position reconciliation found %d discrepancies",
                len(discrepancies),
            )
        else:
            logger.info("Position reconciliation: all positions match")

        return discrepancies

    def reconcile_orders(
        self,
        internal_orders: Sequence[Order],
        broker_orders: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compare internal orders against broker-reported orders.

        ``broker_orders`` is a list of dicts, each containing at least
        ``order_id`` (str), ``status`` (str), ``filled_quantity`` (float).

        Args:
            internal_orders: Our internal order records.
            broker_orders: Broker-reported order records.

        Returns:
            List of discrepancy dicts.
        """
        discrepancies: list[dict[str, Any]] = []

        # Index broker orders by order_id
        broker_by_id: dict[str, dict[str, Any]] = {}
        for bo in broker_orders:
            oid = str(bo.get("order_id", ""))
            if oid:
                broker_by_id[oid] = bo

        internal_by_id: dict[str, Order] = {str(o.order_id): o for o in internal_orders}

        all_ids = set(internal_by_id.keys()) | set(broker_by_id.keys())

        for oid in sorted(all_ids):
            internal = internal_by_id.get(oid)
            broker = broker_by_id.get(oid)

            if internal is not None and broker is None:
                discrepancies.append(_discrepancy(
                    field=f"order.{oid}.exists",
                    internal_value=True,
                    broker_value=False,
                    severity=Severity.WARNING,
                ))
                continue

            if internal is None and broker is not None:
                discrepancies.append(_discrepancy(
                    field=f"order.{oid}.exists",
                    internal_value=False,
                    broker_value=True,
                    severity=Severity.WARNING,
                ))
                continue

            assert internal is not None and broker is not None

            # Status
            broker_status = broker.get("status", "")
            if internal.status.value != broker_status:
                discrepancies.append(_discrepancy(
                    field=f"order.{oid}.status",
                    internal_value=internal.status.value,
                    broker_value=broker_status,
                    severity=Severity.WARNING,
                ))

            # Filled quantity
            broker_filled = float(broker.get("filled_quantity", 0.0))
            if abs(internal.filled_quantity - broker_filled) > 1e-6:
                discrepancies.append(_discrepancy(
                    field=f"order.{oid}.filled_quantity",
                    internal_value=internal.filled_quantity,
                    broker_value=broker_filled,
                    severity=Severity.CRITICAL,
                ))

        if discrepancies:
            logger.warning(
                "Order reconciliation found %d discrepancies",
                len(discrepancies),
            )
        else:
            logger.info("Order reconciliation: all orders match")

        return discrepancies

    def reconcile_balance(
        self,
        internal_balance: float,
        broker_balance: float,
        tolerance_pct: float = 0.1,
    ) -> bool:
        """Check whether internal and broker balances agree within tolerance.

        Args:
            internal_balance: Our tracked cash/equity balance.
            broker_balance: Broker-reported balance.
            tolerance_pct: Acceptable difference as a percentage of the
                broker balance (e.g. 0.1 for 0.1%).

        Returns:
            ``True`` if balances match within tolerance.
        """
        if broker_balance == 0 and internal_balance == 0:
            return True

        reference = max(abs(broker_balance), abs(internal_balance), 1e-9)
        diff_pct = abs(internal_balance - broker_balance) / reference * 100.0

        if diff_pct <= tolerance_pct:
            logger.info(
                "Balance reconciliation OK: internal=%.4f broker=%.4f diff=%.4f%%",
                internal_balance, broker_balance, diff_pct,
            )
            return True

        logger.warning(
            "Balance mismatch: internal=%.4f broker=%.4f diff=%.4f%% (tolerance=%.4f%%)",
            internal_balance, broker_balance, diff_pct, tolerance_pct,
        )
        return False
