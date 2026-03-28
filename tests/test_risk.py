"""Tests for src.engine.risk."""

from __future__ import annotations

import pytest

from src.engine.risk import RiskDecision, RiskEngine
from src.models.order import OrderSide
from src.models.trade import TradeStatus


class TestPositionSizing:
    def test_position_sizing(self, risk_config):
        """Correct position size for given equity, risk, and stop distance."""
        engine = RiskEngine(risk_config)
        # equity=100_000, risk=1%, entry=100, stop=95 -> risk_per_unit=5
        # risk_amount = 100_000 * 0.01 = 1_000
        # raw_size = 1_000 / 5 = 200
        # leveraged_max = (100_000 * 1.0) / 100 = 1_000
        # size = min(200, 1_000) = 200
        size = engine.calculate_position_size(
            equity=100_000.0,
            entry_price=100.0,
            stop_price=95.0,
            risk_pct=1.0,
            leverage=1.0,
        )
        assert size == pytest.approx(200.0)

    def test_position_sizing_with_leverage(self, risk_config):
        """Leverage cap should constrain position size."""
        engine = RiskEngine(risk_config)
        # risk_pct=5%, entry=100, stop=99 -> risk_per_unit=1
        # risk_amount = 10_000 * 0.05 = 500
        # raw_size = 500 / 1 = 500
        # leveraged_max = (10_000 * 2.0) / 100 = 200
        # size = min(500, 200) = 200
        size = engine.calculate_position_size(
            equity=10_000.0,
            entry_price=100.0,
            stop_price=99.0,
            risk_pct=5.0,
            leverage=2.0,
        )
        assert size == pytest.approx(200.0)

    def test_position_sizing_zero_equity(self, risk_config):
        """Zero equity should return zero size."""
        engine = RiskEngine(risk_config)
        size = engine.calculate_position_size(0.0, 100.0, 95.0)
        assert size == 0.0

    def test_position_sizing_same_entry_stop(self, risk_config):
        """Entry == stop should return zero size (zero risk per unit)."""
        engine = RiskEngine(risk_config)
        size = engine.calculate_position_size(100_000.0, 100.0, 100.0)
        assert size == 0.0


class TestCooldown:
    def test_cooldown_active(self, risk_config):
        """Cooldown should block entry within the cooldown window."""
        engine = RiskEngine(risk_config)  # cooldown_bars=5
        result = engine.check_cooldown(
            last_stopout_bar_idx=10,
            current_bar_idx=13,
            direction=OrderSide.LONG,
        )
        assert result is False, "Should be blocked: only 3 bars since stop-out, need 5"

    def test_cooldown_expired(self, risk_config):
        """Cooldown should allow entry after enough bars have passed."""
        engine = RiskEngine(risk_config)  # cooldown_bars=5
        result = engine.check_cooldown(
            last_stopout_bar_idx=10,
            current_bar_idx=15,
            direction=OrderSide.LONG,
        )
        assert result is True, "Should be allowed: 5 bars elapsed since stop-out"

    def test_cooldown_no_prior_stopout(self, risk_config):
        """No prior stop-out should always allow entry."""
        engine = RiskEngine(risk_config)
        result = engine.check_cooldown(
            last_stopout_bar_idx=None,
            current_bar_idx=0,
            direction=OrderSide.LONG,
        )
        assert result is True


class TestDailyTradeLimit:
    def test_daily_trade_limit(self, risk_config):
        """Should block after max trades per day."""
        engine = RiskEngine(risk_config)  # max_trades_per_day=3
        assert engine.check_daily_trade_limit(trades_today=2) is True
        assert engine.check_daily_trade_limit(trades_today=3) is False
        assert engine.check_daily_trade_limit(trades_today=4) is False


class TestConsecutiveLossStop:
    def test_consecutive_loss_stop(self, risk_config, make_trade):
        """Should block after N consecutive losses."""
        engine = RiskEngine(risk_config)  # max_consecutive_losses=3
        # Create 3 consecutive losing trades
        losing_trades = [
            make_trade(
                status=TradeStatus.CLOSED,
                entry_price=100.0,
                exit_price=95.0,  # loss
            )
            for _ in range(3)
        ]
        result = engine.check_consecutive_losses(losing_trades)
        assert result is False, "Should block after 3 consecutive losses"

    def test_consecutive_loss_reset_by_win(self, risk_config, make_trade):
        """A winning trade should reset the loss streak."""
        engine = RiskEngine(risk_config)
        trades = [
            make_trade(status=TradeStatus.CLOSED, entry_price=100.0, exit_price=95.0),
            make_trade(status=TradeStatus.CLOSED, entry_price=100.0, exit_price=95.0),
            make_trade(status=TradeStatus.CLOSED, entry_price=100.0, exit_price=110.0),  # win
            make_trade(status=TradeStatus.CLOSED, entry_price=100.0, exit_price=95.0),
        ]
        result = engine.check_consecutive_losses(trades)
        assert result is True, "Streak is only 1 after the win"


class TestVolatilityFilter:
    def test_volatility_filter(self, risk_config):
        """Should block when ATR is too high relative to its mean."""
        engine = RiskEngine(risk_config)  # volatility_max_ratio=2.0
        # current_atr = 10, rolling_mean = 4 -> ratio = 2.5 > 2.0
        result = engine.check_volatility_filter(
            current_atr=10.0,
            rolling_atr_mean=4.0,
        )
        assert result is False, "ATR ratio 2.5 exceeds max ratio 2.0"

    def test_volatility_filter_passes(self, risk_config):
        """Should pass when ATR is within acceptable range."""
        engine = RiskEngine(risk_config)
        result = engine.check_volatility_filter(
            current_atr=5.0,
            rolling_atr_mean=4.0,
        )
        assert result is True, "ATR ratio 1.25 is within max ratio 2.0"


class TestRunAllChecks:
    def test_all_checks_pass(self, risk_config, make_trade):
        """All checks passing should return an approved decision with position size."""
        engine = RiskEngine(risk_config)
        context = {
            "equity": 100_000.0,
            "entry_price": 100.0,
            "stop_price": 95.0,
            "leverage": 1.0,
            "direction": OrderSide.LONG,
            "current_bar_idx": 100,
            "last_stopout_bar_idx": None,
            "trades_today": 0,
            "recent_trades": [],
            "current_atr": 3.0,
            "rolling_atr_mean": 2.5,
            "day_start_equity": 100_000.0,
            "total_exposure": 0.0,
        }
        decision = engine.run_all_checks(context)
        assert decision.approved is True
        assert decision.position_size > 0
        assert len(decision.reasons) == 0

    def test_all_checks_fail(self, risk_config, make_trade):
        """Multiple failing checks should report all reasons."""
        config = {
            **risk_config,
            "kill_switch": True,             # fail: kill switch
            "max_trades_per_day": 1,         # will fail
            "max_consecutive_losses": 1,     # will fail
        }
        engine = RiskEngine(config)

        losing_trades = [
            make_trade(status=TradeStatus.CLOSED, entry_price=100.0, exit_price=95.0),
        ]

        context = {
            "equity": 100_000.0,
            "entry_price": 100.0,
            "stop_price": 95.0,
            "leverage": 1.0,
            "direction": OrderSide.LONG,
            "current_bar_idx": 2,
            "last_stopout_bar_idx": 1,       # fail: cooldown
            "trades_today": 5,               # fail: daily limit
            "recent_trades": losing_trades,  # fail: consecutive losses
            "current_atr": 20.0,
            "rolling_atr_mean": 5.0,         # fail: volatility ratio = 4
            "day_start_equity": 100_000.0,
            "total_exposure": 0.0,
        }
        decision = engine.run_all_checks(context)
        assert decision.approved is False
        assert decision.position_size == 0.0
        assert len(decision.reasons) >= 3, (
            f"Expected multiple failure reasons, got {len(decision.reasons)}: "
            f"{decision.reasons}"
        )

    def test_kill_switch_blocks(self):
        """Kill switch should block all trades."""
        engine = RiskEngine({"kill_switch": True})
        decision = engine.run_all_checks({"equity": 100_000.0})
        assert decision.approved is False
        assert any("kill switch" in r.lower() for r in decision.reasons)
