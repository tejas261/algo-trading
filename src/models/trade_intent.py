"""TradeIntent model -- deterministic, serialisable output of a strategy signal."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator

from src.models.order import OrderSide


class PartialTarget(BaseModel):
    """A single profit-taking target: exit *pct* % of the position at *price*."""

    model_config = {"frozen": True}

    price: float = Field(gt=0)
    pct: float = Field(gt=0, le=100, description="Percentage of total position to exit at this target")


class TradeIntent(BaseModel):
    """Deterministic, auditable output of a strategy's signal logic.

    This is the *intent* -- it describes what the strategy wants to do.
    The execution engine is responsible for converting this into one or more
    :class:`Order` objects.
    """

    intent_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    targets: list[PartialTarget] = Field(min_length=1)
    position_size: float = Field(gt=0, description="Number of shares / lots")
    atr: Optional[float] = Field(default=None, gt=0, description="ATR at signal time for sizing / trailing")
    signal_timestamp: datetime
    metadata: dict[str, object] = Field(default_factory=dict, description="Extra info for logging / debugging")

    @model_validator(mode="after")
    def _validate_targets(self) -> "TradeIntent":
        total_pct = sum(t.pct for t in self.targets)
        if abs(total_pct - 100.0) > 1e-6:
            raise ValueError(f"Target percentages must sum to 100, got {total_pct:.4f}")
        return self

    @model_validator(mode="after")
    def _validate_stop_direction(self) -> "TradeIntent":
        if self.side == OrderSide.LONG and self.stop_loss >= self.entry_price:
            raise ValueError(
                f"LONG stop_loss ({self.stop_loss}) must be below entry ({self.entry_price})"
            )
        if self.side == OrderSide.SHORT and self.stop_loss <= self.entry_price:
            raise ValueError(
                f"SHORT stop_loss ({self.stop_loss}) must be above entry ({self.entry_price})"
            )
        return self

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def total_risk(self) -> float:
        return self.risk_per_share * self.position_size
