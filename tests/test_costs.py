"""Tests for src.engine.costs."""

from __future__ import annotations

import pytest

from src.engine.costs import CostModel
from src.models.order import OrderSide


@pytest.fixture
def cost_model(cost_config) -> CostModel:
    """CostModel built from the standard cost config fixture."""
    return CostModel.from_config(cost_config)


class TestEntryCost:
    def test_entry_cost_taker(self, cost_model):
        """Taker fee should be applied for non-maker orders."""
        # notional = 100 * 10 = 1000
        # taker_fee = 1000 * (5 / 10_000) = 0.50
        # tax = 1000 * (1 / 10_000) = 0.10
        # total = 0.60
        cost = cost_model.calculate_entry_cost(price=100.0, quantity=10.0, is_maker=False)
        assert cost == pytest.approx(0.60)

    def test_entry_cost_maker(self, cost_model):
        """Maker fee should be applied for maker (limit) orders."""
        # notional = 100 * 10 = 1000
        # maker_fee = 1000 * (2 / 10_000) = 0.20
        # tax = 1000 * (1 / 10_000) = 0.10
        # total = 0.30
        cost = cost_model.calculate_entry_cost(price=100.0, quantity=10.0, is_maker=True)
        assert cost == pytest.approx(0.30)

    def test_entry_cost_zero_fees(self):
        """Zero-fee model should return zero cost."""
        model = CostModel()
        cost = model.calculate_entry_cost(price=100.0, quantity=10.0)
        assert cost == 0.0


class TestExitCost:
    def test_exit_cost_taker(self, cost_model):
        """Exit cost should match entry cost calculation for same parameters."""
        entry_cost = cost_model.calculate_entry_cost(100.0, 10.0, is_maker=False)
        exit_cost = cost_model.calculate_exit_cost(100.0, 10.0, is_maker=False)
        assert exit_cost == pytest.approx(entry_cost)


class TestSlippage:
    def test_slippage_long(self, cost_model):
        """Long entry slippage should increase the fill price."""
        adjusted = cost_model.apply_slippage(100.0, OrderSide.LONG)
        # slippage_bps = 3 -> rate = 3/10000 = 0.0003
        # adjusted = 100 * (1 + 0.0003) = 100.03
        assert adjusted == pytest.approx(100.03)
        assert adjusted > 100.0, "Long slippage should worsen (increase) price"

    def test_slippage_short(self, cost_model):
        """Short entry slippage should decrease the fill price."""
        adjusted = cost_model.apply_slippage(100.0, OrderSide.SHORT)
        # adjusted = 100 * (1 - 0.0003) = 99.97
        assert adjusted == pytest.approx(99.97)
        assert adjusted < 100.0, "Short slippage should worsen (decrease) price"

    def test_slippage_zero(self):
        """Zero slippage should return the original price."""
        model = CostModel(slippage_bps=0.0, base_slippage_bps=0.0, min_slippage_bps=0.0)
        assert model.apply_slippage(100.0, OrderSide.LONG) == 100.0
        assert model.apply_slippage(100.0, OrderSide.SHORT) == 100.0


class TestFundingCost:
    def test_funding_cost(self, cost_model):
        """Funding cost should be pro-rated by holding duration."""
        # funding_bps_per_8h = 1.0, notional = 10_000, held = 24 hours
        # periods = 24 / 8 = 3
        # cost = 10_000 * (1 / 10_000) * 3 = 3.0
        cost = cost_model.calculate_funding(notional=10_000.0, hours_held=24.0)
        assert cost == pytest.approx(3.0)

    def test_funding_cost_partial_period(self, cost_model):
        """Funding should be pro-rated for partial 8-hour periods."""
        # 4 hours = 0.5 periods
        # cost = 10_000 * (1 / 10_000) * 0.5 = 0.5
        cost = cost_model.calculate_funding(notional=10_000.0, hours_held=4.0)
        assert cost == pytest.approx(0.5)

    def test_funding_cost_zero_hours(self, cost_model):
        """Zero holding time should return zero funding cost."""
        cost = cost_model.calculate_funding(notional=10_000.0, hours_held=0.0)
        assert cost == 0.0

    def test_funding_cost_zero_rate(self):
        """Zero funding rate should return zero cost."""
        model = CostModel(funding_bps_per_8h=0.0)
        cost = model.calculate_funding(notional=10_000.0, hours_held=24.0)
        assert cost == 0.0


class TestFromConfig:
    def test_from_config(self, cost_config):
        """from_config should populate all fields correctly."""
        model = CostModel.from_config(cost_config)
        assert model.maker_fee_bps == 2.0
        assert model.taker_fee_bps == 5.0
        assert model.slippage_bps == 3.0
        assert model.tax_bps == 1.0
        assert model.funding_bps_per_8h == 1.0

    def test_from_config_defaults(self):
        """Missing config keys should default to zero."""
        model = CostModel.from_config({})
        assert model.maker_fee_bps == 0.0
        assert model.taker_fee_bps == 0.0
