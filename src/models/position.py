"""Position model with partial exit tracking and trailing stop state."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from src.models.fill import Fill
from src.models.order import OrderSide


class PositionStatus(str, Enum):
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"


class PartialExit(BaseModel):
    """Record of a partial position exit."""

    model_config = {"frozen": True}

    fill: Fill
    exit_quantity: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    realized_pnl: float
    timestamp: datetime


class TrailingStopState(BaseModel):
    """Mutable state for a trailing stop attached to a position."""

    initial_stop: float = Field(gt=0)
    current_stop: float = Field(gt=0)
    trail_points: Optional[float] = None
    trail_pct: Optional[float] = None
    highest_price: Optional[float] = None  # for LONG
    lowest_price: Optional[float] = None   # for SHORT
    activated: bool = False

    def update(self, current_price: float, side: OrderSide) -> float:
        """Advance the trailing stop and return the new stop level."""
        if side == OrderSide.LONG:
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
            if self.trail_points is not None:
                new_stop = self.highest_price - self.trail_points
            elif self.trail_pct is not None:
                new_stop = self.highest_price * (1 - self.trail_pct / 100)
            else:
                return self.current_stop
            self.current_stop = max(self.current_stop, new_stop)
        else:
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
            if self.trail_points is not None:
                new_stop = self.lowest_price + self.trail_points
            elif self.trail_pct is not None:
                new_stop = self.lowest_price * (1 + self.trail_pct / 100)
            else:
                return self.current_stop
            self.current_stop = min(self.current_stop, new_stop)

        self.activated = True
        return self.current_stop


class Position(BaseModel):
    """Tracks a position from open through partial exits to close."""

    position_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    entry_price: float = Field(gt=0)
    entry_quantity: float = Field(gt=0)
    entry_fill: Fill
    entry_timestamp: datetime

    current_quantity: float = Field(gt=0)
    status: PositionStatus = Field(default=PositionStatus.OPEN)

    partial_exits: list[PartialExit] = Field(default_factory=list)
    trailing_stop: Optional[TrailingStopState] = None

    stop_loss: Optional[float] = None
    closed_at: Optional[datetime] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exited_quantity(self) -> float:
        return sum(pe.exit_quantity for pe in self.partial_exits)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def realized_pnl(self) -> float:
        return sum(pe.realized_pnl for pe in self.partial_exits)

    def unrealized_pnl(self, current_price: float) -> float:
        """Compute mark-to-market PnL on the remaining open quantity."""
        if self.side == OrderSide.LONG:
            return (current_price - self.entry_price) * self.current_quantity
        return (self.entry_price - current_price) * self.current_quantity

    def total_pnl(self, current_price: float) -> float:
        return self.realized_pnl + self.unrealized_pnl(current_price)

    def apply_partial_exit(self, fill: Fill) -> PartialExit:
        """Record a partial exit and update position state."""
        if fill.quantity > self.current_quantity + 1e-9:
            raise ValueError(
                f"Exit qty {fill.quantity} exceeds remaining {self.current_quantity}"
            )

        if self.side == OrderSide.LONG:
            pnl = (fill.price - self.entry_price) * fill.quantity
        else:
            pnl = (self.entry_price - fill.price) * fill.quantity

        pnl -= fill.commission + fill.slippage

        partial = PartialExit(
            fill=fill,
            exit_quantity=fill.quantity,
            exit_price=fill.price,
            realized_pnl=pnl,
            timestamp=fill.timestamp,
        )
        self.partial_exits.append(partial)
        self.current_quantity -= fill.quantity

        if self.current_quantity <= 1e-9:
            self.current_quantity = 0.0
            self.status = PositionStatus.CLOSED
            self.closed_at = fill.timestamp
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED

        return partial

    def is_stopped_out(self, current_price: float) -> bool:
        """Check if current price has breached the stop loss."""
        stop = self.trailing_stop.current_stop if self.trailing_stop else self.stop_loss
        if stop is None:
            return False
        if self.side == OrderSide.LONG:
            return current_price <= stop
        return current_price >= stop
