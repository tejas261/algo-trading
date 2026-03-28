"""
Technical indicator calculations for the algo-trading framework.

All functions operate on pandas DataFrames/Series and return Series.
Indicator formulas follow standard TA conventions, using Wilder's
smoothing where appropriate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average using the standard span-based decay factor.

    alpha = 2 / (period + 1)
    """
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average (rolling mean)."""
    return series.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range using Wilder's smoothing (alpha = 1/period).

    Requires columns: high, low, close.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing is an EMA with alpha = 1 / period
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Directional movement / ADX
# ---------------------------------------------------------------------------

def adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Average Directional Index (Wilder).

    Steps:
      1. Compute +DM / -DM per bar.
      2. Smooth +DM, -DM, and TR with Wilder's smoothing (alpha=1/period).
      3. +DI = 100 * smoothed(+DM) / smoothed(TR)
         -DI = 100 * smoothed(-DM) / smoothed(TR)
      4. DX  = 100 * |+DI - -DI| / (+DI + -DI)
      5. ADX = Wilder-smoothed DX (alpha=1/period)

    Requires columns: high, low, close.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True range components
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Wilder's smoothing
    alpha = 1.0 / period
    smoothed_tr = tr.ewm(alpha=alpha, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # Directional indicators
    plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
    minus_di = 100.0 * smoothed_minus_dm / smoothed_tr

    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()

    # Avoid division by zero
    dx = pd.Series(
        np.where(di_sum != 0, 100.0 * di_diff / di_sum, 0.0),
        index=df.index,
    )

    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx_series


# ---------------------------------------------------------------------------
# Channel extremes
# ---------------------------------------------------------------------------

def highest_high(series: pd.Series, period: int) -> pd.Series:
    """Rolling maximum over *period* bars."""
    return series.rolling(window=period, min_periods=period).max()


def lowest_low(series: pd.Series, period: int) -> pd.Series:
    """Rolling minimum over *period* bars."""
    return series.rolling(window=period, min_periods=period).min()


# ---------------------------------------------------------------------------
# Composite helper
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict = {
    "ema_fast": 20,
    "ema_mid": 50,
    "ema_slow": 200,
    "adx_period": 14,
    "atr_period": 14,
    "channel_period": 20,
    "volume_sma_period": 20,
    "atr_filter_lookback": 50,
}


def compute_all_indicators(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Add all strategy-required indicator columns to *df* and return it.

    Added columns:
        ema_20, ema_50, ema_200, adx_14, atr_14,
        highest_high_20, lowest_low_20, volume_sma_20,
        atr_14_sma_50,
        highest_high_55, lowest_low_55, lowest_low_20_close, highest_high_20_close

    The caller may override periods via *config*; keys that are absent
    fall back to sensible defaults.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}

    out = df.copy()

    out["ema_20"] = ema(out["close"], cfg["ema_fast"])
    out["ema_50"] = ema(out["close"], cfg["ema_mid"])
    out["ema_200"] = ema(out["close"], cfg["ema_slow"])

    out["adx_14"] = adx(out, cfg["adx_period"])
    out["atr_14"] = atr(out, cfg["atr_period"])

    out["highest_high_20"] = highest_high(out["high"], cfg["channel_period"])
    out["lowest_low_20"] = lowest_low(out["low"], cfg["channel_period"])

    out["volume_sma_20"] = sma(out["volume"], cfg["volume_sma_period"])

    # Rolling ATR mean for volatility filter
    atr_filter_lookback = cfg.get("atr_filter_lookback", 50)
    out["atr_14_sma_50"] = sma(out["atr_14"], atr_filter_lookback)

    # Donchian 55-bar channels (for donchian_trend_55 strategy)
    out["highest_high_55"] = highest_high(out["high"], 55)
    out["lowest_low_55"] = lowest_low(out["low"], 55)

    # Opposite 20-bar breakout on close (for donchian exit)
    out["lowest_low_20_close"] = lowest_low(out["close"], 20)
    out["highest_high_20_close"] = highest_high(out["close"], 20)

    return out
