"""Shared pytest fixtures for the algo-trading test suite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.models.fill import Fill
from src.models.order import OrderSide
from src.models.trade import Trade, TradeStatus
from src.models.trade_intent import PartialTarget, TradeIntent
from src.strategy.signals import SignalType


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_config() -> dict:
    """Standard risk configuration for tests."""
    return {
        "risk_per_trade_pct": 1.0,
        "max_leverage": 3.0,
        "cooldown_bars": 5,
        "max_trades_per_day": 3,
        "max_consecutive_losses": 3,
        "volatility_max_ratio": 2.0,
        "max_daily_drawdown_pct": 5.0,
        "max_exposure_pct": 50.0,
        "kill_switch": False,
    }


@pytest.fixture
def cost_config() -> dict:
    """Standard cost model configuration for tests."""
    return {
        "maker_fee_bps": 2.0,
        "taker_fee_bps": 5.0,
        "slippage_bps": 3.0,
        "tax_bps": 1.0,
        "funding_bps_per_8h": 1.0,
    }


@pytest.fixture
def strategy_config() -> dict:
    """Standard strategy configuration for tests."""
    return {
        "strategy_name": "test_strategy",
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "stop_loss_atr_mult": 2.0,
        "target_atr_mult": 3.0,
        "leverage": 1.0,
        "signal_column": "signal",
        "atr_column": "atr_14",
    }


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a 100-bar OHLCV DataFrame with realistic price action."""
    np.random.seed(42)
    n = 100
    base_price = 100.0
    timestamps = [
        datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        for i in range(n)
    ]

    closes = [base_price]
    for _ in range(n - 1):
        change = np.random.normal(0, 0.5)
        closes.append(closes[-1] + change)
    closes = np.array(closes)

    highs = closes + np.abs(np.random.normal(0.5, 0.3, n))
    lows = closes - np.abs(np.random.normal(0.5, 0.3, n))
    opens = lows + (highs - lows) * np.random.uniform(0.2, 0.8, n)
    volume = np.random.uniform(1000, 5000, n)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


@pytest.fixture
def constant_series() -> pd.Series:
    """Series of 50 constant values at 100.0."""
    return pd.Series([100.0] * 50)


@pytest.fixture
def trending_up_series() -> pd.Series:
    """Monotonically increasing series from 100 to 149."""
    return pd.Series(range(100, 150), dtype=float)


@pytest.fixture
def trending_down_series() -> pd.Series:
    """Monotonically decreasing series from 150 to 101."""
    return pd.Series(range(150, 100, -1), dtype=float)


@pytest.fixture
def sideways_ohlcv_df() -> pd.DataFrame:
    """50-bar DataFrame with price oscillating in a tight range (sideways)."""
    np.random.seed(99)
    n = 100
    base = 100.0
    timestamps = [
        datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        for i in range(n)
    ]
    # Small oscillations around base
    closes = base + np.sin(np.linspace(0, 20 * np.pi, n)) * 0.5
    highs = closes + 0.3
    lows = closes - 0.3
    opens = closes + np.random.uniform(-0.1, 0.1, n)
    # Clamp opens within high/low
    opens = np.clip(opens, lows, highs)
    volume = np.full(n, 2000.0)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


@pytest.fixture
def trending_ohlcv_df() -> pd.DataFrame:
    """100-bar DataFrame with a strong uptrend."""
    n = 300
    timestamps = [
        datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        for i in range(n)
    ]
    closes = np.linspace(100, 200, n)
    highs = closes + 1.5
    lows = closes - 1.5
    opens = closes - 0.5
    volume = np.full(n, 3000.0)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_fill():
    """Factory for creating Fill objects."""
    def _make(
        symbol: str = "BTC/USDT",
        side: OrderSide = OrderSide.LONG,
        quantity: float = 1.0,
        price: float = 100.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        timestamp: datetime | None = None,
    ) -> Fill:
        return Fill(
            order_id=uuid4(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp or datetime(2025, 1, 1, tzinfo=timezone.utc),
            commission=commission,
            slippage=slippage,
        )
    return _make


@pytest.fixture
def make_trade(make_fill):
    """Factory for creating Trade objects."""
    def _make(
        symbol: str = "BTC/USDT",
        side: OrderSide = OrderSide.LONG,
        entry_price: float = 100.0,
        entry_quantity: float = 1.0,
        status: TradeStatus = TradeStatus.OPEN,
        realized_pnl: float = 0.0,
        exit_price: float | None = None,
    ) -> Trade:
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        entry_fill = make_fill(
            symbol=symbol,
            side=side,
            quantity=entry_quantity,
            price=entry_price,
            timestamp=ts,
        )
        trade = Trade(
            symbol=symbol,
            side=side,
            entry_fill=entry_fill,
            entry_price=entry_price,
            entry_quantity=entry_quantity,
            entry_timestamp=ts,
            initial_stop_loss=entry_price * 0.98 if side == OrderSide.LONG else entry_price * 1.02,
            initial_risk_per_share=entry_price * 0.02,
        )

        if status == TradeStatus.CLOSED and exit_price is not None:
            exit_fill = make_fill(
                symbol=symbol,
                side=side,
                quantity=entry_quantity,
                price=exit_price,
                timestamp=ts + timedelta(hours=1),
            )
            trade.close(exit_fill)

        return trade
    return _make


@pytest.fixture
def make_trade_intent():
    """Factory for creating TradeIntent objects."""
    def _make(
        symbol: str = "BTC/USDT",
        side: OrderSide = OrderSide.LONG,
        entry_price: float = 100.0,
        stop_loss: float = 95.0,
        position_size: float = 10.0,
        atr: float = 2.5,
    ) -> TradeIntent:
        if side == OrderSide.LONG:
            target_price = entry_price + atr * 3
        else:
            target_price = entry_price - atr * 3

        return TradeIntent(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets=[PartialTarget(price=target_price, pct=100.0)],
            position_size=position_size,
            atr=atr,
            signal_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
    return _make
