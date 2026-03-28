"""Trade model capturing the full lifecycle from entry through partial exits to final close."""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, computed_field

from src.models.fill import Fill
from src.models.order import OrderSide
from src.models.position import PartialExit


class TradeStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Trade(BaseModel):
    """Full lifecycle record of a single trading idea from entry to exit.

    A trade is conceptually a higher-level wrapper around a :class:`Position`.
    It records the entry fill, all partial exits, and the final exit, together
    with computed PnL and duration metrics.
    """

    trade_id: UUID = Field(default_factory=uuid4)
    symbol: str
    side: OrderSide
    status: TradeStatus = Field(default=TradeStatus.OPEN)

    # Entry
    entry_fill: Fill
    entry_price: float = Field(gt=0)
    entry_quantity: float = Field(gt=0)
    entry_timestamp: datetime

    # Exits
    partial_exits: list[PartialExit] = Field(default_factory=list)
    final_exit_fill: Optional[Fill] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None

    # Risk context
    initial_stop_loss: Optional[float] = None
    initial_risk_per_share: Optional[float] = None

    tag: str = Field(default="", description="Strategy or signal label")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary trade metadata")

    # ----- computed fields -----

    @computed_field  # type: ignore[prop-decorator]
    @property
    def realized_pnl(self) -> float:
        """Total realized PnL across all partial and final exits."""
        return sum(pe.realized_pnl for pe in self.partial_exits)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_commission(self) -> float:
        total = self.entry_fill.commission
        for pe in self.partial_exits:
            total += pe.fill.commission
        return total

    @computed_field  # type: ignore[prop-decorator]
    @property
    def avg_exit_price(self) -> Optional[float]:
        if not self.partial_exits:
            return None
        total_value = sum(pe.exit_price * pe.exit_quantity for pe in self.partial_exits)
        total_qty = sum(pe.exit_quantity for pe in self.partial_exits)
        return total_value / total_qty if total_qty > 0 else None

    @property
    def duration(self) -> Optional[timedelta]:
        if self.exit_timestamp is None:
            return None
        return self.exit_timestamp - self.entry_timestamp

    @property
    def return_pct(self) -> Optional[float]:
        """Percentage return relative to entry notional."""
        notional = self.entry_price * self.entry_quantity
        if notional == 0:
            return None
        return (self.realized_pnl / notional) * 100

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """R-multiple: realized PnL / initial risk."""
        if self.initial_risk_per_share is None or self.initial_risk_per_share == 0:
            return None
        initial_risk = self.initial_risk_per_share * self.entry_quantity
        return self.realized_pnl / initial_risk

    @property
    def is_winner(self) -> Optional[bool]:
        if self.status != TradeStatus.CLOSED:
            return None
        return self.realized_pnl > 0

    def close(self, final_fill: Fill) -> None:
        """Record the final exit and mark the trade as closed."""
        pnl = self._compute_exit_pnl(final_fill)
        partial = PartialExit(
            fill=final_fill,
            exit_quantity=final_fill.quantity,
            exit_price=final_fill.price,
            realized_pnl=pnl,
            timestamp=final_fill.timestamp,
        )
        self.partial_exits.append(partial)
        self.final_exit_fill = final_fill
        self.exit_price = final_fill.price
        self.exit_timestamp = final_fill.timestamp
        self.status = TradeStatus.CLOSED

    def add_partial_exit(self, fill: Fill) -> PartialExit:
        """Record a partial exit."""
        pnl = self._compute_exit_pnl(fill)
        partial = PartialExit(
            fill=fill,
            exit_quantity=fill.quantity,
            exit_price=fill.price,
            realized_pnl=pnl,
            timestamp=fill.timestamp,
        )
        self.partial_exits.append(partial)
        return partial

    def _compute_exit_pnl(self, fill: Fill) -> float:
        if self.side == OrderSide.LONG:
            pnl = (fill.price - self.entry_price) * fill.quantity
        else:
            pnl = (self.entry_price - fill.price) * fill.quantity
        pnl -= fill.commission + fill.slippage
        return pnl
