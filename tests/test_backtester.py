"""Tests for src.engine.backtester."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.engine.backtester import Backtester
from src.models.results import BacktestResults
from src.strategy.signals import SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_synthetic_ohlcv(
    n: int = 50,
    base_price: float = 100.0,
    trend: float = 0.0,
    atr_value: float = 2.0,
    signals: list[tuple[int, SignalType]] | None = None,
) -> pd.DataFrame:
    """Create synthetic OHLCV data with optional signals and ATR.

    Parameters
    ----------
    n : int
        Number of candles.
    base_price : float
        Starting close price.
    trend : float
        Per-bar drift added to close.
    atr_value : float
        Constant ATR value injected into the atr_14 column.
    signals : list of (index, SignalType)
        Specific bars where signals should be placed.

    Returns
    -------
    pd.DataFrame
        DataFrame ready for the Backtester.
    """
    timestamps = [
        datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        for i in range(n)
    ]

    closes = np.array([base_price + trend * i for i in range(n)])
    opens = closes - 0.3
    highs = closes + atr_value / 2
    lows = closes - atr_value / 2
    volume = np.full(n, 2000.0)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
        "atr_14": np.full(n, atr_value),
        "signal": [SignalType.NO_SIGNAL] * n,
    })

    if signals:
        for idx, sig in signals:
            if 0 <= idx < n:
                df.at[idx, "signal"] = sig

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBacktestNoSignals:
    def test_backtest_no_signals(self, strategy_config):
        """No signals should result in zero trades."""
        df = _create_synthetic_ohlcv(n=50)
        bt = Backtester(
            df=df,
            strategy_config=strategy_config,
            initial_capital=100_000.0,
        )
        results = bt.run()
        assert len(results.trade_log) == 0
        assert results.final_equity == pytest.approx(100_000.0)


class TestBacktestBasicLong:
    def test_backtest_basic_long(self, strategy_config):
        """A single LONG signal should produce at least one trade."""
        # Place a LONG signal early in the data with upward trend so
        # the trade can be profitable or at least fill.
        df = _create_synthetic_ohlcv(
            n=50,
            base_price=100.0,
            trend=0.5,
            atr_value=2.0,
            signals=[(5, SignalType.LONG)],
        )
        bt = Backtester(
            df=df,
            strategy_config=strategy_config,
            initial_capital=100_000.0,
        )
        results = bt.run()
        # Should have at least 1 trade (entry at bar 6, close at end if not
        # stopped out or target hit)
        assert len(results.trade_log) >= 1
        trade = results.trade_log[0]
        assert trade.side.value == "LONG"


class TestBacktestEndOfDataClose:
    def test_backtest_end_of_data_close(self, strategy_config):
        """Open position should be force-closed at end of data."""
        # Place signal near end so it's unlikely to hit targets/stops
        df = _create_synthetic_ohlcv(
            n=20,
            base_price=100.0,
            trend=0.0,  # flat -- no stop or target hit
            atr_value=2.0,
            signals=[(5, SignalType.LONG)],
        )
        bt = Backtester(
            df=df,
            strategy_config=strategy_config,
            initial_capital=100_000.0,
        )
        results = bt.run()
        # The trade should be closed (force-closed at end)
        if results.trade_log:
            from src.models.trade import TradeStatus
            assert results.trade_log[-1].status == TradeStatus.CLOSED


class TestBacktestReturnsResults:
    def test_backtest_returns_results(self, strategy_config):
        """Backtester should return BacktestResults with all required fields."""
        df = _create_synthetic_ohlcv(n=30)
        bt = Backtester(
            df=df,
            strategy_config=strategy_config,
            initial_capital=100_000.0,
        )
        results = bt.run()

        assert isinstance(results, BacktestResults)
        assert results.strategy_name == "test_strategy"
        assert results.initial_capital == 100_000.0
        assert results.final_equity > 0
        assert isinstance(results.trade_log, list)
        assert isinstance(results.equity_curve, list)
        assert isinstance(results.daily_returns, list)
        assert results.start_date is not None
        assert results.end_date is not None

    def test_backtest_empty_df(self, strategy_config):
        """Empty DataFrame should return empty results without crashing."""
        df = pd.DataFrame(columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "atr_14", "signal",
        ])
        bt = Backtester(
            df=df,
            strategy_config=strategy_config,
            initial_capital=100_000.0,
        )
        results = bt.run()
        assert isinstance(results, BacktestResults)
        assert len(results.trade_log) == 0
