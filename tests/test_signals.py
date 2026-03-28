"""Tests for src.strategy.signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.signals import SignalType, generate_signal, generate_signals_vectorized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_indicator_row(
    close: float = 110.0,
    volume: float = 3000.0,
    ema_20: float = 108.0,
    ema_50: float = 105.0,
    ema_200: float = 100.0,
    adx_14: float = 30.0,
    highest_high_20: float = 109.0,
    lowest_low_20: float = 91.0,
    volume_sma_20: float = 2000.0,
) -> pd.Series:
    """Build a single row with all indicator columns needed for signal logic."""
    return pd.Series({
        "close": close,
        "volume": volume,
        "ema_20": ema_20,
        "ema_50": ema_50,
        "ema_200": ema_200,
        "adx_14": adx_14,
        "highest_high_20": highest_high_20,
        "lowest_low_20": lowest_low_20,
        "volume_sma_20": volume_sma_20,
    })


def _build_indicator_df(
    n: int = 5,
    **overrides,
) -> pd.DataFrame:
    """Build a DataFrame of identical rows with indicator columns."""
    rows = [_build_indicator_row(**overrides) for _ in range(n)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Row-level signal
# ---------------------------------------------------------------------------

class TestGenerateSignal:
    def test_long_signal_conditions(self):
        """All long conditions met should produce LONG signal."""
        # LONG conditions:
        # close > ema200, ema20 > ema50, adx > 20, close > prev_hh, vol > vol_sma
        row = _build_indicator_row(
            close=115.0,       # > ema200 (100)
            ema_20=112.0,      # > ema50 (105)
            ema_50=105.0,
            ema_200=100.0,
            adx_14=30.0,       # > 20
            volume=3000.0,     # > vol_sma (2000)
            volume_sma_20=2000.0,
            highest_high_20=114.0,  # prev bar's HH
        )
        prev_rows = _build_indicator_df(
            n=1,
            highest_high_20=114.0,  # close (115) > prev_hh (114)
            lowest_low_20=90.0,
        )
        result = generate_signal(row, prev_rows)
        assert result == SignalType.LONG

    def test_short_signal_conditions(self):
        """All short conditions met should produce SHORT signal."""
        # SHORT conditions:
        # close < ema200, ema20 < ema50, adx > 20, close < prev_ll, vol > vol_sma
        row = _build_indicator_row(
            close=85.0,        # < ema200 (100)
            ema_20=88.0,       # < ema50 (95)
            ema_50=95.0,
            ema_200=100.0,
            adx_14=30.0,       # > 20
            volume=3000.0,     # > vol_sma (2000)
            volume_sma_20=2000.0,
            lowest_low_20=90.0,
        )
        prev_rows = _build_indicator_df(
            n=1,
            highest_high_20=110.0,
            lowest_low_20=86.0,  # close (85) < prev_ll (86)
        )
        result = generate_signal(row, prev_rows)
        assert result == SignalType.SHORT

    def test_no_signal_missing_condition(self):
        """Missing one condition (weak ADX) should produce NO_SIGNAL."""
        row = _build_indicator_row(
            close=115.0,
            ema_20=112.0,
            ema_50=105.0,
            ema_200=100.0,
            adx_14=15.0,       # < 20 -- fails ADX filter
            volume=3000.0,
            volume_sma_20=2000.0,
        )
        prev_rows = _build_indicator_df(
            n=1,
            highest_high_20=114.0,
            lowest_low_20=90.0,
        )
        result = generate_signal(row, prev_rows)
        assert result == SignalType.NO_SIGNAL

    def test_no_signal_empty_prev_rows(self):
        """Empty previous rows should produce NO_SIGNAL."""
        row = _build_indicator_row()
        prev_rows = pd.DataFrame()
        result = generate_signal(row, prev_rows)
        assert result == SignalType.NO_SIGNAL

    def test_no_signal_nan_values(self):
        """NaN in any required field should produce NO_SIGNAL."""
        row = _build_indicator_row(close=float("nan"))
        prev_rows = _build_indicator_df(n=1)
        result = generate_signal(row, prev_rows)
        assert result == SignalType.NO_SIGNAL

    def test_no_signal_low_adx(self):
        """ADX below threshold should block signal even if other conditions met."""
        row = _build_indicator_row(
            close=115.0,
            ema_20=112.0,
            ema_50=105.0,
            ema_200=100.0,
            adx_14=15.0,  # below default threshold of 18
            volume=5000.0,
            volume_sma_20=2000.0,
        )
        prev_rows = _build_indicator_df(
            n=1,
            highest_high_20=114.0,
            lowest_low_20=90.0,
        )
        result = generate_signal(row, prev_rows)
        assert result == SignalType.NO_SIGNAL


# ---------------------------------------------------------------------------
# Vectorized signal generation
# ---------------------------------------------------------------------------

class TestVectorizedSignals:
    def test_vectorized_signals(self):
        """Vectorized results should match row-by-row results."""
        n = 30
        np.random.seed(42)

        closes = np.linspace(95, 115, n)
        df = pd.DataFrame({
            "close": closes,
            "volume": np.random.uniform(1500, 3500, n),
            "ema_20": closes - 2,
            "ema_50": closes - 5,
            "ema_200": np.full(n, 100.0),
            "adx_14": np.random.uniform(15, 35, n),
            "highest_high_20": closes - 1,  # shifted below close for some breakouts
            "lowest_low_20": closes + 1,    # shifted above close for some breakdowns
            "volume_sma_20": np.full(n, 2000.0),
        })

        vectorized = generate_signals_vectorized(df)

        # Check row-by-row for rows where prev_rows is available
        for i in range(1, n):
            row = df.iloc[i]
            prev_rows = df.iloc[:i]
            row_signal = generate_signal(row, prev_rows)
            assert vectorized.iloc[i] == row_signal, (
                f"Mismatch at index {i}: vectorized={vectorized.iloc[i]}, "
                f"row_by_row={row_signal}"
            )

    def test_vectorized_returns_series(self):
        """Vectorized function should return a pandas Series."""
        df = _build_indicator_df(n=10)
        result = generate_signals_vectorized(df)
        assert isinstance(result, pd.Series)
        assert len(result) == 10

    def test_vectorized_all_no_signal(self):
        """When no conditions are met, all signals should be NO_SIGNAL."""
        df = _build_indicator_df(
            n=10,
            adx_14=10.0,  # weak ADX blocks everything
        )
        result = generate_signals_vectorized(df)
        assert (result == SignalType.NO_SIGNAL).all()
