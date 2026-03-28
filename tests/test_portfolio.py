"""Tests for src.engine.portfolio.PortfolioTracker."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.engine.portfolio import PortfolioTracker
from src.models.order import OrderSide
from src.models.position import PositionStatus


class TestOpenPosition:
    def test_open_position(self, make_trade_intent):
        """Opening a position should deduct notional + cost from cash."""
        tracker = PortfolioTracker(initial_capital=100_000.0)
        intent = make_trade_intent(
            entry_price=100.0,
            stop_loss=95.0,
            position_size=10.0,
        )

        fill_price = 100.0
        fill_qty = 10.0
        cost = 5.0

        position = tracker.open_position(
            trade_intent=intent,
            fill_price=fill_price,
            fill_qty=fill_qty,
            cost=cost,
        )

        assert position.symbol == "BTC/USDT"
        assert position.entry_price == 100.0
        assert position.current_quantity == 10.0
        # Cash should be: 100_000 - (100 * 10) - 5 = 98_995
        assert tracker.cash == pytest.approx(98_995.0)

    def test_open_position_equity(self, make_trade_intent):
        """Equity should remain approximately equal after opening (before price move)."""
        tracker = PortfolioTracker(initial_capital=100_000.0)
        intent = make_trade_intent(
            entry_price=100.0,
            stop_loss=95.0,
            position_size=10.0,
        )

        tracker.open_position(
            trade_intent=intent,
            fill_price=100.0,
            fill_qty=10.0,
            cost=0.0,  # zero cost to isolate equity check
        )
        tracker.update_mark_prices({"BTC/USDT": 100.0})
        equity = tracker.get_equity()
        assert equity == pytest.approx(100_000.0, abs=1.0)


class TestClosePosition:
    def test_close_position(self, make_trade_intent):
        """Closing a position should calculate PnL correctly."""
        tracker = PortfolioTracker(initial_capital=100_000.0)
        intent = make_trade_intent(
            entry_price=100.0,
            stop_loss=95.0,
            position_size=10.0,
        )

        tracker.open_position(
            trade_intent=intent,
            fill_price=100.0,
            fill_qty=10.0,
            cost=0.0,
        )

        # Close at 110 -> profit of (110 - 100) * 10 = 100
        pnl = tracker.close_position(
            symbol="BTC/USDT",
            exit_price=110.0,
            cost=0.0,
        )
        assert pnl == pytest.approx(100.0)

        # After closing, no open positions
        assert len(tracker.positions) == 0

    def test_close_position_with_loss(self, make_trade_intent):
        """Closing at a loss should return negative PnL."""
        tracker = PortfolioTracker(initial_capital=100_000.0)
        intent = make_trade_intent(
            entry_price=100.0,
            stop_loss=95.0,
            position_size=10.0,
        )
        tracker.open_position(
            trade_intent=intent,
            fill_price=100.0,
            fill_qty=10.0,
            cost=0.0,
        )
        pnl = tracker.close_position("BTC/USDT", exit_price=90.0, cost=0.0)
        assert pnl == pytest.approx(-100.0)


class TestPartialExit:
    def test_partial_exit(self, make_trade_intent):
        """Partial close should update quantity and track PnL."""
        tracker = PortfolioTracker(initial_capital=100_000.0)
        intent = make_trade_intent(
            entry_price=100.0,
            stop_loss=95.0,
            position_size=10.0,
        )
        tracker.open_position(
            trade_intent=intent,
            fill_price=100.0,
            fill_qty=10.0,
            cost=0.0,
        )

        # Exit 4 units at 105
        pnl = tracker.apply_partial_exit(
            symbol="BTC/USDT",
            quantity=4.0,
            exit_price=105.0,
            cost=0.0,
        )
        assert pnl == pytest.approx(20.0)  # (105-100)*4 = 20

        # Should still have 6 units
        pos = tracker.positions["BTC/USDT"]
        assert pos.current_quantity == pytest.approx(6.0)
        assert pos.status == PositionStatus.PARTIALLY_CLOSED


class TestDailySnapshot:
    def test_daily_snapshot(self, make_trade_intent):
        """Snapshot should record current equity."""
        tracker = PortfolioTracker(initial_capital=100_000.0)
        tracker.update_mark_prices({"BTC/USDT": 100.0})

        snapshot = tracker.record_daily_snapshot(
            snapshot_date=date(2025, 1, 1),
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert snapshot.equity == pytest.approx(100_000.0)
        assert snapshot.date == date(2025, 1, 1)

    def test_daily_returns_list(self, make_trade_intent):
        """Multiple snapshots should accumulate in daily returns list."""
        tracker = PortfolioTracker(initial_capital=100_000.0)

        for i in range(3):
            tracker.record_daily_snapshot(
                snapshot_date=date(2025, 1, 1 + i),
                timestamp=datetime(2025, 1, 1 + i, tzinfo=timezone.utc),
            )
        returns = tracker.get_daily_returns()
        assert len(returns) == 3


class TestEquityCurve:
    def test_equity_curve(self, make_trade_intent):
        """Equity curve should track observations over time."""
        tracker = PortfolioTracker(initial_capital=100_000.0)

        ts1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        ts3 = datetime(2025, 1, 3, tzinfo=timezone.utc)

        tracker.record_equity_point(ts1)
        tracker.record_equity_point(ts2)
        tracker.record_equity_point(ts3)

        curve = tracker.get_equity_curve()
        assert len(curve) == 3
        # All points should be at initial capital (no trades)
        for ts, eq in curve:
            assert eq == pytest.approx(100_000.0)

    def test_equity_curve_reflects_pnl(self, make_trade_intent):
        """Equity curve should reflect PnL from trades."""
        tracker = PortfolioTracker(initial_capital=100_000.0)
        intent = make_trade_intent(
            entry_price=100.0,
            stop_loss=95.0,
            position_size=10.0,
        )
        tracker.open_position(
            trade_intent=intent,
            fill_price=100.0,
            fill_qty=10.0,
            cost=0.0,
        )

        # Record before price move
        tracker.update_mark_prices({"BTC/USDT": 100.0})
        ts1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
        tracker.record_equity_point(ts1)

        # Price goes up
        tracker.update_mark_prices({"BTC/USDT": 110.0})
        ts2 = datetime(2025, 1, 2, tzinfo=timezone.utc)
        tracker.record_equity_point(ts2)

        curve = tracker.get_equity_curve()
        assert len(curve) == 2
        # Second point should reflect the unrealized gain
        assert curve[1][1] > curve[0][1]
