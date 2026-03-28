"""Tests for src.engine.reconciler."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.engine.reconciler import Reconciler
from src.models.fill import Fill
from src.models.order import OrderSide
from src.models.position import Position


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_position(
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.LONG,
    entry_price: float = 100.0,
    quantity: float = 10.0,
) -> Position:
    """Create a Position for reconciliation tests."""
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    fill = Fill(
        order_id=uuid4(),
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=entry_price,
        timestamp=ts,
    )
    return Position(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        entry_quantity=quantity,
        entry_fill=fill,
        entry_timestamp=ts,
        current_quantity=quantity,
    )


# ---------------------------------------------------------------------------
# Position reconciliation
# ---------------------------------------------------------------------------

class TestReconcilePositions:
    def test_matching_positions(self):
        """No discrepancies when internal and broker positions match."""
        reconciler = Reconciler()
        internal = {"BTC/USDT": _make_position("BTC/USDT", quantity=10.0)}
        broker = {
            "BTC/USDT": {
                "quantity": 10.0,
                "side": "LONG",
                "entry_price": 100.0,
            }
        }
        discrepancies = reconciler.reconcile_positions(internal, broker)
        assert len(discrepancies) == 0

    def test_quantity_mismatch(self):
        """Should detect when quantities differ between internal and broker."""
        reconciler = Reconciler()
        internal = {"BTC/USDT": _make_position("BTC/USDT", quantity=10.0)}
        broker = {
            "BTC/USDT": {
                "quantity": 8.0,  # mismatch
                "side": "LONG",
            }
        }
        discrepancies = reconciler.reconcile_positions(internal, broker)
        assert len(discrepancies) >= 1
        qty_disc = [d for d in discrepancies if "quantity" in d["field"]]
        assert len(qty_disc) == 1
        assert qty_disc[0]["internal_value"] == 10.0
        assert qty_disc[0]["broker_value"] == 8.0

    def test_missing_position_at_broker(self):
        """Should detect position in internal but not at broker."""
        reconciler = Reconciler()
        internal = {"BTC/USDT": _make_position("BTC/USDT")}
        broker = {}  # broker has no position
        discrepancies = reconciler.reconcile_positions(internal, broker)
        assert len(discrepancies) >= 1
        assert discrepancies[0]["internal_value"] is True
        assert discrepancies[0]["broker_value"] is False

    def test_missing_position_internally(self):
        """Should detect position at broker but not internally."""
        reconciler = Reconciler()
        internal = {}
        broker = {
            "ETH/USDT": {
                "quantity": 5.0,
                "side": "LONG",
            }
        }
        discrepancies = reconciler.reconcile_positions(internal, broker)
        assert len(discrepancies) >= 1
        assert discrepancies[0]["internal_value"] is False
        assert discrepancies[0]["broker_value"] is True

    def test_side_mismatch(self):
        """Should detect side mismatch."""
        reconciler = Reconciler()
        internal = {"BTC/USDT": _make_position("BTC/USDT", side=OrderSide.LONG)}
        broker = {
            "BTC/USDT": {
                "quantity": 10.0,
                "side": "SHORT",  # mismatch
            }
        }
        discrepancies = reconciler.reconcile_positions(internal, broker)
        side_disc = [d for d in discrepancies if "side" in d["field"]]
        assert len(side_disc) >= 1


# ---------------------------------------------------------------------------
# Balance reconciliation
# ---------------------------------------------------------------------------

class TestReconcileBalance:
    def test_balance_within_tolerance(self):
        """Should pass when balances are within tolerance."""
        reconciler = Reconciler()
        # 0.1% tolerance; 100_000 vs 100_050 = 0.05% diff
        result = reconciler.reconcile_balance(
            internal_balance=100_000.0,
            broker_balance=100_050.0,
            tolerance_pct=0.1,
        )
        assert result is True

    def test_balance_outside_tolerance(self):
        """Should fail when balances differ beyond tolerance."""
        reconciler = Reconciler()
        # 0.1% tolerance; 100_000 vs 101_000 = 1.0% diff
        result = reconciler.reconcile_balance(
            internal_balance=100_000.0,
            broker_balance=101_000.0,
            tolerance_pct=0.1,
        )
        assert result is False

    def test_balance_both_zero(self):
        """Both zero should be considered matching."""
        reconciler = Reconciler()
        result = reconciler.reconcile_balance(0.0, 0.0)
        assert result is True

    def test_balance_exact_match(self):
        """Exact match should pass."""
        reconciler = Reconciler()
        result = reconciler.reconcile_balance(50_000.0, 50_000.0)
        assert result is True
