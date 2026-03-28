"""
High-level strategy wrapper.

Encapsulates indicator computation and signal generation into a single
deterministic, side-effect-free pipeline.

Supports two strategy variants via the ``strategy_name`` config key:
  - ``trend_breakout_v2`` (default)
  - ``donchian_trend_55``
"""

from __future__ import annotations

import pandas as pd

from src.strategy.indicators import compute_all_indicators
from src.strategy.signals import (
    SignalType,
    generate_signals_donchian,
    generate_signals_vectorized,
)


class TrendBreakoutStrategy:
    """Trend-following strategy with configurable signal generation.

    Parameters
    ----------
    config : dict | None
        Indicator period overrides and strategy selection.

        * ``strategy_name``: ``"trend_breakout_v2"`` (default) or ``"donchian_trend_55"``
        * ``ema_fast``  (default 20)
        * ``ema_mid``   (default 50)
        * ``ema_slow``  (default 200)
        * ``adx_period`` (default 14)
        * ``adx_threshold`` (default 18)
        * ``atr_period`` (default 14)
        * ``channel_period`` (default 20)
        * ``atr_filter_lookback`` (default 50)
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config: dict = config or {}
        self._strategy_name = self.config.get("strategy_name", "trend_breakout_v2")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full strategy pipeline.

        1. Compute all technical indicators.
        2. Generate entry signals based on selected strategy variant.

        Returns a copy of *df* enriched with indicator and ``signal`` columns.
        """
        self._validate_input(df)

        result = compute_all_indicators(df, self.config)

        if self._strategy_name == "donchian_trend_55":
            result["signal"] = generate_signals_donchian(result)
        else:
            adx_threshold = float(self.config.get("adx_threshold", 18.0))
            result["signal"] = generate_signals_vectorized(result, adx_threshold=adx_threshold)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        """Raise if required OHLCV columns are missing."""
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing required columns: {sorted(missing)}"
            )

    def __repr__(self) -> str:
        return f"TrendBreakoutStrategy(strategy={self._strategy_name!r}, config={self.config!r})"
