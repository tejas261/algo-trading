"""Tests for the LangGraph agent nodes.

These tests exercise the node functions in isolation by passing in
TradingState dicts and verifying the returned partial-state updates.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.state import TradingState
from src.engine.risk import RiskDecision
from src.models.order import OrderSide
from src.strategy.signals import SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_initial_state(**overrides) -> dict:
    """Build a TradingState dict with sensible defaults."""
    state: dict = {
        "mode": "paper",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "current_timestamp": None,
        "market_data": None,
        "latest_snapshot": None,
        "indicators": None,
        "signal": None,
        "trade_intent": None,
        "risk_decision": None,
        "approval_status": None,
        "execution_result": None,
        "open_positions": [],
        "open_orders": [],
        "portfolio_state": None,
        "anomaly_flags": [],
        "audit_events": [],
        "report_payload": None,
        "error": None,
        "step": "init",
        "should_continue": True,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# State creation
# ---------------------------------------------------------------------------

class TestInitialStateCreation:
    def test_initial_state_creation(self):
        """State dict should have all required TradingState fields."""
        state = _make_initial_state()
        required_fields = [
            "mode", "symbol", "timeframe", "current_timestamp",
            "market_data", "latest_snapshot", "indicators", "signal",
            "trade_intent", "risk_decision", "approval_status",
            "execution_result", "open_positions", "open_orders",
            "portfolio_state", "anomaly_flags", "audit_events",
            "report_payload", "error", "step", "should_continue",
        ]
        for field in required_fields:
            assert field in state, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# Risk checks node routing
# ---------------------------------------------------------------------------

class TestRouteAfterRisk:
    def test_route_after_risk_no_signal(self):
        """When signal is NO_SIGNAL, risk node should set should_continue=False."""
        from src.agents.nodes.run_risk_checks import run_risk_checks_node

        state = _make_initial_state(signal=SignalType.NO_SIGNAL)
        result = run_risk_checks_node(state)

        assert result["should_continue"] is False
        decision = result["risk_decision"]
        assert decision.approved is False

    def test_route_after_risk_approved(self):
        """When signal is valid and risk passes, should_continue should be True."""
        from src.agents.nodes.run_risk_checks import run_risk_checks_node

        # Build minimal indicators DataFrame
        indicators = pd.DataFrame({
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
            "atr_14": [2.0],
        })

        state = _make_initial_state(
            signal=SignalType.LONG,
            indicators=indicators,
            config={
                "risk": {
                    "risk_per_trade_pct": 1.0,
                    "max_leverage": 1.0,
                    "cooldown_bars": 0,
                    "max_trades_per_day": 999,
                    "max_consecutive_losses": 999,
                    "atr_stop_multiplier": 1.5,
                    "initial_equity": 100_000.0,
                },
            },
            portfolio_state={
                "equity": 100_000.0,
                "day_start_equity": 100_000.0,
                "total_exposure": 0.0,
                "trades_today": 0,
            },
        )
        result = run_risk_checks_node(state)
        assert result["should_continue"] is True
        assert result["risk_decision"].approved is True
        assert result["risk_decision"].position_size > 0


# ---------------------------------------------------------------------------
# Approval node routing
# ---------------------------------------------------------------------------

class TestRouteApproval:
    def test_route_approval_live(self):
        """In live mode without langgraph, approval should fall back to rejected."""
        from src.agents.nodes.request_human_approval import request_human_approval_node
        from src.models.trade_intent import PartialTarget, TradeIntent
        from datetime import datetime, timezone

        intent = TradeIntent(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            targets=[PartialTarget(price=110.0, pct=100.0)],
            position_size=10.0,
            atr=2.0,
            signal_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        state = _make_initial_state(mode="live", trade_intent=intent)
        result = request_human_approval_node(state)
        # Without langgraph installed, live mode should fall back to rejected
        assert result["approval_status"] == "rejected"

    def test_route_approval_paper(self):
        """In paper mode, approval should be auto-approved."""
        from src.agents.nodes.request_human_approval import request_human_approval_node
        from src.models.trade_intent import PartialTarget, TradeIntent
        from datetime import datetime, timezone

        intent = TradeIntent(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            targets=[PartialTarget(price=110.0, pct=100.0)],
            position_size=10.0,
            atr=2.0,
            signal_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        state = _make_initial_state(mode="paper", trade_intent=intent)
        result = request_human_approval_node(state)
        assert result["approval_status"] == "approved"

    def test_route_approval_backtest(self):
        """In backtest mode, approval should be auto-approved."""
        from src.agents.nodes.request_human_approval import request_human_approval_node
        from src.models.trade_intent import PartialTarget, TradeIntent
        from datetime import datetime, timezone

        intent = TradeIntent(
            symbol="BTC/USDT",
            side=OrderSide.LONG,
            entry_price=100.0,
            stop_loss=95.0,
            targets=[PartialTarget(price=110.0, pct=100.0)],
            position_size=10.0,
            atr=2.0,
            signal_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        state = _make_initial_state(mode="backtest", trade_intent=intent)
        result = request_human_approval_node(state)
        assert result["approval_status"] == "approved"

    def test_route_approval_no_intent(self):
        """No trade intent should result in rejection."""
        from src.agents.nodes.request_human_approval import request_human_approval_node

        state = _make_initial_state(trade_intent=None)
        result = request_human_approval_node(state)
        assert result["approval_status"] == "rejected"


# ---------------------------------------------------------------------------
# Build trade intent node
# ---------------------------------------------------------------------------

class TestBuildTradeIntentNode:
    def test_builds_intent_on_valid_signal(self):
        """Should build a TradeIntent when signal and risk are both valid."""
        from src.agents.nodes.build_trade_intent import build_trade_intent_node
        from datetime import datetime, timezone

        indicators = pd.DataFrame({
            "close": [100.0],
            "high": [101.0],
            "low": [99.0],
            "atr_14": [2.0],
            "timestamp": [datetime(2025, 1, 1, tzinfo=timezone.utc)],
        })

        risk_decision = RiskDecision(
            approved=True,
            reasons=[],
            position_size=50.0,
        )

        state = _make_initial_state(
            signal=SignalType.LONG,
            risk_decision=risk_decision,
            indicators=indicators,
            symbol="BTC/USDT",
            config={"strategy": {"atr_stop_multiplier": 1.5}},
        )

        result = build_trade_intent_node(state)
        intent = result["trade_intent"]
        assert intent is not None
        assert intent.symbol == "BTC/USDT"
        assert intent.side == OrderSide.LONG
        assert intent.position_size == 50.0

    def test_no_intent_on_no_signal(self):
        """Should return None trade_intent when signal is NO_SIGNAL."""
        from src.agents.nodes.build_trade_intent import build_trade_intent_node

        state = _make_initial_state(signal=SignalType.NO_SIGNAL)
        result = build_trade_intent_node(state)
        assert result["trade_intent"] is None


# ---------------------------------------------------------------------------
# Monitor positions node
# ---------------------------------------------------------------------------

class TestMonitorPositions:
    def test_no_anomalies_on_clean_state(self):
        """Clean state should produce no anomaly flags."""
        from src.agents.nodes.monitor_positions import monitor_positions_node

        state = _make_initial_state(
            open_positions=[],
            latest_snapshot={},
            execution_result={},  # must be dict, not None
        )
        result = monitor_positions_node(state)
        assert len(result["anomaly_flags"]) == 0

    def test_detects_zero_quantity(self):
        """Should flag a position with zero quantity."""
        from src.agents.nodes.monitor_positions import monitor_positions_node

        state = _make_initial_state(
            open_positions=[{
                "symbol": "BTC/USDT",
                "quantity": 0.0,
                "side": "LONG",
                "entry_price": 100.0,
            }],
            latest_snapshot={},   # must be dict, not None
            execution_result={},  # must be dict, not None
        )
        result = monitor_positions_node(state)
        flags = result["anomaly_flags"]
        assert any("zero_quantity" in f for f in flags)
