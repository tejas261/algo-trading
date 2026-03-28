"""Pydantic models for market data: OHLCV bars and snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class OHLCVBar(BaseModel):
    """A single OHLCV candlestick bar."""

    model_config = {"frozen": True}

    symbol: str
    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0, default=0.0)

    @model_validator(mode="after")
    def _check_ohlc_consistency(self) -> "OHLCVBar":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high < self.open or self.high < self.close:
            raise ValueError(f"high ({self.high}) must be >= open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError(f"low ({self.low}) must be <= open and close")
        return self

    @property
    def mid(self) -> float:
        """Midpoint of high and low."""
        return (self.high + self.low) / 2.0

    @property
    def range(self) -> float:
        """High minus low."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Absolute body size (|close - open|)."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def typical_price(self) -> float:
        """(High + Low + Close) / 3."""
        return (self.high + self.low + self.close) / 3.0


class MarketSnapshot(BaseModel):
    """A single OHLCV bar enriched with pre-computed indicator values.

    Use ``indicators`` to attach any signal or feature values that the
    strategy needs for decision-making.
    """

    bar: OHLCVBar
    indicators: dict[str, Any] = Field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return self.bar.symbol

    @property
    def timestamp(self) -> datetime:
        return self.bar.timestamp

    @property
    def close(self) -> float:
        return self.bar.close

    def get_indicator(self, name: str, default: Any = None) -> Any:
        """Retrieve an indicator value by name."""
        return self.indicators.get(name, default)

    def require_indicator(self, name: str) -> Any:
        """Retrieve an indicator value, raising ``KeyError`` if absent."""
        if name not in self.indicators:
            raise KeyError(f"Indicator '{name}' not found in snapshot for {self.symbol}")
        return self.indicators[name]
