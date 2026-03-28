"""Fill / execution model."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.models.order import OrderSide


class Fill(BaseModel):
    """Represents a single execution (fill) against an order."""

    model_config = {"frozen": True}

    fill_id: UUID = Field(default_factory=uuid4)
    order_id: UUID
    symbol: str
    side: OrderSide
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    timestamp: datetime
    commission: float = Field(default=0.0, ge=0)
    slippage: float = Field(default=0.0, ge=0)

    @property
    def notional_value(self) -> float:
        """Gross value of the fill (price * quantity)."""
        return self.price * self.quantity

    @property
    def total_cost(self) -> float:
        """Net cost including commission and slippage."""
        return self.notional_value + self.commission + self.slippage
