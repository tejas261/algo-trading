"""Portfolio state model tracking equity, cash, positions, and daily returns."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from src.models.order import OrderSide
from src.models.position import Position, PositionStatus


class DailyReturn(BaseModel):
    """Snapshot of portfolio value at end-of-day."""

    model_config = {"frozen": True}

    date: date
    equity: float
    cash: float
    positions_value: float
    daily_return_pct: float
    cumulative_return_pct: float


class Portfolio(BaseModel):
    """Full portfolio state at a point in time."""

    initial_capital: float = Field(gt=0)
    cash: float
    positions: dict[str, Position] = Field(default_factory=dict)
    closed_positions: list[Position] = Field(default_factory=list)
    daily_returns: list[DailyReturn] = Field(default_factory=list)
    last_updated: Optional[datetime] = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def open_position_count(self) -> int:
        return sum(1 for p in self.positions.values() if p.status != PositionStatus.CLOSED)

    def positions_value(self, prices: dict[str, float]) -> float:
        """Mark-to-market value of all open positions.

        Args:
            prices: Mapping of symbol -> current price.
        """
        total = 0.0
        for symbol, pos in self.positions.items():
            if pos.status == PositionStatus.CLOSED:
                continue
            price = prices.get(symbol, pos.entry_price)
            if pos.side == OrderSide.LONG:
                total += price * pos.current_quantity
            else:
                # Short position value: entry_notional + unrealized P&L
                total += (2 * pos.entry_price - price) * pos.current_quantity
        return total

    def equity(self, prices: dict[str, float]) -> float:
        """Total equity = cash + positions value."""
        return self.cash + self.positions_value(prices)

    def add_position(self, position: Position) -> None:
        """Add a newly opened position."""
        if position.symbol in self.positions:
            raise ValueError(f"Position already exists for {position.symbol}")
        self.positions[position.symbol] = position

    def remove_position(self, symbol: str) -> Position:
        """Remove a closed position and archive it."""
        pos = self.positions.pop(symbol, None)
        if pos is None:
            raise KeyError(f"No open position for {symbol}")
        self.closed_positions.append(pos)
        return pos

    def record_daily_return(self, d: date, prices: dict[str, float]) -> DailyReturn:
        """Snapshot today's portfolio state and compute daily return."""
        eq = self.equity(prices)
        pos_val = self.positions_value(prices)

        if self.daily_returns:
            prev_eq = self.daily_returns[-1].equity
            daily_ret = ((eq - prev_eq) / prev_eq * 100) if prev_eq != 0 else 0.0
        else:
            daily_ret = ((eq - self.initial_capital) / self.initial_capital * 100) if self.initial_capital != 0 else 0.0

        cum_ret = ((eq - self.initial_capital) / self.initial_capital * 100) if self.initial_capital != 0 else 0.0

        snapshot = DailyReturn(
            date=d,
            equity=eq,
            cash=self.cash,
            positions_value=pos_val,
            daily_return_pct=daily_ret,
            cumulative_return_pct=cum_ret,
        )
        self.daily_returns.append(snapshot)
        return snapshot

    def drawdown_pct(self, prices: dict[str, float]) -> float:
        """Current drawdown from peak equity (percentage)."""
        eq = self.equity(prices)
        if not self.daily_returns:
            peak = self.initial_capital
        else:
            peak = max(dr.equity for dr in self.daily_returns)
        peak = max(peak, eq)
        if peak == 0:
            return 0.0
        return ((peak - eq) / peak) * 100
