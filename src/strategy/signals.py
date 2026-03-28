"""
Signal generation for trend-following strategies.

Supports two strategy variants:
  - trend_breakout_v2: EMA regime + ADX + breakout (no volume filter)
  - donchian_trend_55: EMA(200) regime + 55-bar Donchian breakout

Provides both row-by-row and fully vectorized implementations.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal enum
# ---------------------------------------------------------------------------

class SignalType(Enum):
    """Discrete trade-signal types."""

    LONG = "LONG"
    SHORT = "SHORT"
    NO_SIGNAL = "NO_SIGNAL"


# ---------------------------------------------------------------------------
# trend_breakout_v2: row-level signal
# ---------------------------------------------------------------------------

def generate_signal(
    row: pd.Series,
    prev_rows: pd.DataFrame,
    adx_threshold: float = 18.0,
) -> SignalType:
    """Evaluate trend_breakout_v2 rules for a single bar.

    Rules (no volume filter):
      Long:  close > EMA(200), EMA(20) > EMA(50), ADX > threshold,
             close > prev highest_high_20
      Short: close < EMA(200), EMA(20) < EMA(50), ADX > threshold,
             close < prev lowest_low_20
    """
    if prev_rows.empty:
        return SignalType.NO_SIGNAL

    prev = prev_rows.iloc[-1]

    close = row["close"]
    ema20 = row["ema_20"]
    ema50 = row["ema_50"]
    ema200 = row["ema_200"]
    adx_val = row["adx_14"]

    prev_hh = prev["highest_high_20"]
    prev_ll = prev["lowest_low_20"]

    required = [close, ema20, ema50, ema200, adx_val, prev_hh, prev_ll]
    if any(pd.isna(v) for v in required):
        return SignalType.NO_SIGNAL

    adx_filter = adx_val > adx_threshold

    # Long entry
    if (
        close > ema200
        and ema20 > ema50
        and adx_filter
        and close > prev_hh
    ):
        return SignalType.LONG

    # Short entry
    if (
        close < ema200
        and ema20 < ema50
        and adx_filter
        and close < prev_ll
    ):
        return SignalType.SHORT

    return SignalType.NO_SIGNAL


# ---------------------------------------------------------------------------
# trend_breakout_v2: vectorized
# ---------------------------------------------------------------------------

def generate_signals_vectorized(
    df: pd.DataFrame,
    adx_threshold: float = 18.0,
) -> pd.Series:
    """Vectorized trend_breakout_v2 signal generation (no volume filter)."""
    close = df["close"]
    ema20 = df["ema_20"]
    ema50 = df["ema_50"]
    ema200 = df["ema_200"]
    adx_val = df["adx_14"]

    prev_hh = df["highest_high_20"].shift(1)
    prev_ll = df["lowest_low_20"].shift(1)

    adx_filter = adx_val > adx_threshold

    long_cond = (
        (close > ema200)
        & (ema20 > ema50)
        & adx_filter
        & (close > prev_hh)
    )

    short_cond = (
        (close < ema200)
        & (ema20 < ema50)
        & adx_filter
        & (close < prev_ll)
    )

    signals = pd.Series(SignalType.NO_SIGNAL, index=df.index)
    signals[long_cond] = SignalType.LONG
    signals[short_cond] = SignalType.SHORT

    return signals


# ---------------------------------------------------------------------------
# donchian_trend_55: vectorized
# ---------------------------------------------------------------------------

def generate_signals_donchian(df: pd.DataFrame) -> pd.Series:
    """Vectorized Donchian Trend 55 signal generation.

    Long:  close > EMA(200) AND close > highest_high_55 (prev bar)
    Short: close < EMA(200) AND close < lowest_low_55 (prev bar)
    """
    close = df["close"]
    ema200 = df["ema_200"]

    prev_hh55 = df["highest_high_55"].shift(1)
    prev_ll55 = df["lowest_low_55"].shift(1)

    long_cond = (close > ema200) & (close > prev_hh55)
    short_cond = (close < ema200) & (close < prev_ll55)

    signals = pd.Series(SignalType.NO_SIGNAL, index=df.index)
    signals[long_cond] = SignalType.LONG
    signals[short_cond] = SignalType.SHORT

    return signals
