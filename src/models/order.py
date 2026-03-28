"""Order model with type, side, and status enums."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Order(BaseModel):
    """Represents a single order submitted to a broker or simulated engine."""

    order_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0)
    filled_quantity: float = Field(default=0.0, ge=0)
    price: Optional[float] = Field(default=None, description="Limit price or stop trigger price")
    stop_price: Optional[float] = Field(default=None, description="Stop trigger for STOP_LIMIT orders")
    status: OrderStatus = Field(default=OrderStatus.PENDING)

    created_at: datetime
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None

    parent_order_id: Optional[UUID] = None
    tag: str = Field(default="", description="Free-form label for grouping/filtering")

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL_FILL)

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED)

    def mark_filled(self, filled_qty: float, fill_time: datetime) -> None:
        """Update order state after a fill event."""
        self.filled_quantity += filled_qty
        self.updated_at = fill_time
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = fill_time
        else:
            self.status = OrderStatus.PARTIAL_FILL

    def cancel(self, cancel_time: datetime) -> None:
        """Mark the order as cancelled."""
        if self.is_terminal:
            raise ValueError(f"Cannot cancel order in terminal state: {self.status}")
        self.status = OrderStatus.CANCELLED
        self.updated_at = cancel_time

    def reject(self, reject_time: datetime) -> None:
        """Mark the order as rejected."""
        self.status = OrderStatus.REJECTED
        self.updated_at = reject_time
