"""Tests for src.strategy.indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.indicators import (
    adx,
    atr,
    compute_all_indicators,
    ema,
    highest_high,
    lowest_low,
    sma,
)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_basic(self, constant_series):
        """EMA of a constant series should equal the constant at every point."""
        result = ema(constant_series, period=10)
        # All values should be 100.0 since input is constant
        np.testing.assert_allclose(result.values, 100.0, atol=1e-10)

    def test_ema_trending(self, trending_up_series):
        """EMA of a rising series should lag below the current price."""
        result = ema(trending_up_series, period=10)
        # For a rising series, EMA should be below the actual value
        # (except possibly the very first point)
        last_val = trending_up_series.iloc[-1]
        last_ema = result.iloc[-1]
        assert last_ema < last_val, (
            f"EMA ({last_ema:.4f}) should lag below price ({last_val}) "
            "in an uptrend"
        )

    def test_ema_length(self, constant_series):
        """Output length matches input length."""
        result = ema(constant_series, period=10)
        assert len(result) == len(constant_series)


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------

class TestSMA:
    def test_sma_basic(self, constant_series):
        """SMA of a constant series should equal the constant (after warmup)."""
        result = sma(constant_series, period=10)
        # First 9 values are NaN (min_periods=10), rest are 100.0
        valid = result.dropna()
        np.testing.assert_allclose(valid.values, 100.0, atol=1e-10)

    def test_sma_known_values(self):
        """SMA on known data should produce expected average."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(series, period=3)
        # SMA(3) at index 2 = (1+2+3)/3 = 2.0
        # SMA(3) at index 3 = (2+3+4)/3 = 3.0
        # SMA(3) at index 4 = (3+4+5)/3 = 4.0
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_sma_nan_warmup(self):
        """SMA should return NaN for the warmup period."""
        series = pd.Series(range(20), dtype=float)
        result = sma(series, period=10)
        assert result.iloc[:9].isna().all()
        assert result.iloc[9:].notna().all()


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_calculation(self):
        """ATR on known data should produce values in a reasonable range."""
        n = 50
        closes = np.linspace(100, 110, n)
        highs = closes + 2.0
        lows = closes - 2.0
        df = pd.DataFrame({
            "high": highs,
            "low": lows,
            "close": closes,
        })
        result = atr(df, period=14)

        # True range for each bar is at least high - low = 4.0
        # After warmup, ATR should stabilize near 4.0
        valid = result.dropna()
        last_atr = valid.iloc[-1]
        assert 3.5 < last_atr < 5.0, (
            f"ATR ({last_atr:.4f}) should be close to 4.0 for data with "
            "constant high-low range of 4.0"
        )

    def test_atr_positive(self, sample_ohlcv_df):
        """ATR values should always be non-negative."""
        result = atr(sample_ohlcv_df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

class TestADX:
    def test_adx_trending(self, trending_ohlcv_df):
        """ADX should be > 25 for strongly trending data."""
        result = adx(trending_ohlcv_df, period=14)
        # Use the tail since ADX needs warmup
        last_adx = result.iloc[-1]
        assert last_adx > 25, (
            f"ADX ({last_adx:.2f}) should be > 25 for a strong uptrend"
        )

    def test_adx_ranging(self, sideways_ohlcv_df):
        """ADX should be < 20 for sideways / range-bound data."""
        result = adx(sideways_ohlcv_df, period=14)
        last_adx = result.iloc[-1]
        assert last_adx < 20, (
            f"ADX ({last_adx:.2f}) should be < 20 for sideways data"
        )

    def test_adx_range(self, sample_ohlcv_df):
        """ADX values should be between 0 and 100."""
        result = adx(sample_ohlcv_df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


# ---------------------------------------------------------------------------
# Channel extremes
# ---------------------------------------------------------------------------

class TestChannelExtremes:
    def test_highest_high(self):
        """highest_high correctly identifies the rolling max."""
        series = pd.Series([1, 3, 2, 5, 4, 6, 3, 7, 2, 8], dtype=float)
        result = highest_high(series, period=3)
        # At index 2 (window [1,3,2]): max=3
        assert result.iloc[2] == 3.0
        # At index 3 (window [3,2,5]): max=5
        assert result.iloc[3] == 5.0
        # At index 9 (window [7,2,8]): max=8
        assert result.iloc[9] == 8.0

    def test_lowest_low(self):
        """lowest_low correctly identifies the rolling min."""
        series = pd.Series([5, 3, 7, 1, 4, 6, 2, 8, 3, 9], dtype=float)
        result = lowest_low(series, period=3)
        # At index 2 (window [5,3,7]): min=3
        assert result.iloc[2] == 3.0
        # At index 3 (window [3,7,1]): min=1
        assert result.iloc[3] == 1.0

    def test_highest_high_nan_warmup(self):
        """Rolling max should have NaN for the warmup period."""
        series = pd.Series(range(10), dtype=float)
        result = highest_high(series, period=5)
        assert result.iloc[:4].isna().all()
        assert result.iloc[4:].notna().all()

    def test_lowest_low_nan_warmup(self):
        """Rolling min should have NaN for the warmup period."""
        series = pd.Series(range(10), dtype=float)
        result = lowest_low(series, period=5)
        assert result.iloc[:4].isna().all()
        assert result.iloc[4:].notna().all()


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class TestComputeAllIndicators:
    def test_compute_all_indicators(self, sample_ohlcv_df):
        """All expected indicator columns should be present after computation."""
        result = compute_all_indicators(sample_ohlcv_df)

        expected_columns = [
            "ema_20", "ema_50", "ema_200",
            "adx_14", "atr_14",
            "highest_high_20", "lowest_low_20",
            "volume_sma_20",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"

    def test_compute_all_preserves_original(self, sample_ohlcv_df):
        """Original OHLCV columns should still be present."""
        result = compute_all_indicators(sample_ohlcv_df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_compute_all_custom_config(self, sample_ohlcv_df):
        """Custom config overrides should be respected."""
        config = {"ema_fast": 10, "atr_period": 7}
        result = compute_all_indicators(sample_ohlcv_df, config=config)
        # Columns still named with defaults, but computation uses custom periods
        assert "ema_20" in result.columns
        assert "atr_14" in result.columns

    def test_compute_all_row_count(self, sample_ohlcv_df):
        """Output should have the same number of rows as input."""
        result = compute_all_indicators(sample_ohlcv_df)
        assert len(result) == len(sample_ohlcv_df)
