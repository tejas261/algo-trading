"""Portfolio tracker for managing positions, equity, and daily snapshots."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional
from uuid import uuid4

from src.models.fill import Fill
from src.models.order import OrderSide
from src.models.portfolio import DailyReturn, Portfolio
from src.models.position import Position, PositionStatus, TrailingStopState
from src.models.trade_intent import TradeIntent
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioTracker:
    """Manages portfolio state including positions, cash, and equity tracking.

    This class wraps the :class:`Portfolio` model and provides mutation methods
    for the backtester and live execution engine.

    Args:
        initial_capital: Starting cash balance.
    """

    def __init__(self, initial_capital: float) -> None:
        self._portfolio = Portfolio(
            initial_capital=initial_capital,
            cash=initial_capital,
        )
        self._mark_prices: dict[str, float] = {}
        self._equity_curve: list[tuple[datetime, float]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def initial_capital(self) -> float:
        return self._portfolio.initial_capital

    @property
    def cash(self) -> float:
        return self._portfolio.cash

    @cash.setter
    def cash(self, value: float) -> None:
        self._portfolio.cash = value

    @property
    def positions(self) -> dict[str, Position]:
        return self._portfolio.positions

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(
        self,
        trade_intent: TradeIntent,
        fill_price: float,
        fill_qty: float,
        cost: float,
        timestamp: datetime | None = None,
    ) -> Position:
        """Open a new position based on a trade intent and its fill.

        Deducts the entry notional (for longs) or credits it (for shorts)
        from cash, and subtracts transaction costs.

        Args:
            trade_intent: The strategy's trade intent.
            fill_price: Actual fill price (after slippage).
            fill_qty: Quantity filled.
            cost: Total transaction cost (fees + tax).
            timestamp: Fill timestamp; defaults to intent's signal timestamp.

        Returns:
            The newly created :class:`Position`.

        Raises:
            ValueError: If a position already exists for the symbol.
        """
        ts = timestamp or trade_intent.signal_timestamp

        fill = Fill(
            order_id=uuid4(),
            symbol=trade_intent.symbol,
            side=trade_intent.side,
            quantity=fill_qty,
            price=fill_price,
            timestamp=ts,
            commission=cost,
            slippage=0.0,  # slippage already baked into fill_price
        )

        # Build trailing stop state if ATR is available
        # Note: the backtester now manages trailing stops directly via
        # _current_trailing_stop, but we still set up the position's
        # trailing stop state for compatibility with the execution simulator.
        trailing_stop: Optional[TrailingStopState] = None
        if trade_intent.atr is not None:
            initial_stop = trade_intent.stop_loss
            # Use ATR as trail distance (backtester may override with its own tracking)
            trail_distance = trade_intent.atr
            if trade_intent.side == OrderSide.LONG:
                trailing_stop = TrailingStopState(
                    initial_stop=initial_stop,
                    current_stop=initial_stop,
                    trail_points=trail_distance,
                    highest_price=fill_price,
                )
            else:
                trailing_stop = TrailingStopState(
                    initial_stop=initial_stop,
                    current_stop=initial_stop,
                    trail_points=trail_distance,
                    lowest_price=fill_price,
                )

        position = Position(
            symbol=trade_intent.symbol,
            side=trade_intent.side,
            entry_price=fill_price,
            entry_quantity=fill_qty,
            entry_fill=fill,
            entry_timestamp=ts,
            current_quantity=fill_qty,
            stop_loss=trade_intent.stop_loss,
            trailing_stop=trailing_stop,
        )

        # Cash accounting
        notional = fill_price * fill_qty
        if trade_intent.side == OrderSide.LONG:
            self._portfolio.cash -= notional + cost
        else:
            # Short: we receive notional but post margin equivalent
            self._portfolio.cash -= notional + cost

        self._portfolio.add_position(position)
        self._mark_prices[trade_intent.symbol] = fill_price

        logger.info(
            "Opened %s position: %s qty=%.4f @ %.4f cost=%.4f",
            trade_intent.side.value,
            trade_intent.symbol,
            fill_qty,
            fill_price,
            cost,
        )
        return position

    def apply_partial_exit(
        self,
        symbol: str,
        quantity: float,
        exit_price: float,
        cost: float,
        timestamp: datetime | None = None,
    ) -> float:
        """Apply a partial exit to an open position.

        Args:
            symbol: The position's symbol.
            quantity: Quantity to exit.
            exit_price: Exit fill price.
            cost: Transaction cost of the exit.
            timestamp: Fill timestamp.

        Returns:
            Realized PnL from this partial exit (net of costs).

        Raises:
            KeyError: If no open position exists for the symbol.
        """
        if symbol not in self._portfolio.positions:
            raise KeyError(f"No open position for {symbol}")

        position = self._portfolio.positions[symbol]
        ts = timestamp or datetime.utcnow()

        fill = Fill(
            order_id=uuid4(),
            symbol=symbol,
            side=position.side,
            quantity=quantity,
            price=exit_price,
            timestamp=ts,
            commission=cost,
            slippage=0.0,
        )

        partial = position.apply_partial_exit(fill)

        # Cash accounting: receive exit notional minus cost
        notional = exit_price * quantity
        if position.side == OrderSide.LONG:
            self._portfolio.cash += notional - cost
        else:
            # Short close: pay back notional, net the difference
            self._portfolio.cash += notional - cost

        self._mark_prices[symbol] = exit_price

        logger.info(
            "Partial exit %s: qty=%.4f @ %.4f pnl=%.4f",
            symbol, quantity, exit_price, partial.realized_pnl,
        )

        # If position is fully closed, archive it
        if position.status == PositionStatus.CLOSED:
            self._portfolio.remove_position(symbol)
            logger.info("Position %s fully closed via partial exits", symbol)

        return partial.realized_pnl

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        cost: float,
        timestamp: datetime | None = None,
    ) -> float:
        """Fully close an open position.

        This is a convenience method that exits the entire remaining quantity.

        Args:
            symbol: The position's symbol.
            exit_price: Exit fill price.
            cost: Transaction cost of the exit.
            timestamp: Fill timestamp.

        Returns:
            Realized PnL from closing the remaining position.

        Raises:
            KeyError: If no open position exists for the symbol.
        """
        if symbol not in self._portfolio.positions:
            raise KeyError(f"No open position for {symbol}")

        position = self._portfolio.positions[symbol]
        remaining = position.current_quantity
        return self.apply_partial_exit(symbol, remaining, exit_price, cost, timestamp)

    # ------------------------------------------------------------------
    # Equity & mark-to-market
    # ------------------------------------------------------------------

    def get_equity(self) -> float:
        """Calculate current total equity (cash + mark-to-market positions).

        Returns:
            Total portfolio equity.
        """
        return self._portfolio.equity(self._mark_prices)

    def update_mark_prices(self, prices_dict: dict[str, float]) -> None:
        """Update mark-to-market prices for open positions.

        Args:
            prices_dict: Mapping of symbol to current market price.
        """
        self._mark_prices.update(prices_dict)

    # ------------------------------------------------------------------
    # Snapshots & returns
    # ------------------------------------------------------------------

    def record_daily_snapshot(
        self,
        snapshot_date: date | None = None,
        timestamp: datetime | None = None,
    ) -> DailyReturn:
        """Record end-of-day equity snapshot and compute daily return.

        Args:
            snapshot_date: The date for this snapshot.
            timestamp: Optional timestamp for equity curve recording.

        Returns:
            The :class:`DailyReturn` snapshot.
        """
        d = snapshot_date or date.today()
        snapshot = self._portfolio.record_daily_return(d, self._mark_prices)

        ts = timestamp or datetime.now()
        self._equity_curve.append((ts, snapshot.equity))

        logger.debug(
            "Daily snapshot: date=%s equity=%.2f return=%.4f%%",
            d, snapshot.equity, snapshot.daily_return_pct,
        )
        return snapshot

    def get_daily_returns(self) -> list[DailyReturn]:
        """Return all recorded daily return snapshots.

        Returns:
            List of :class:`DailyReturn` objects in chronological order.
        """
        return list(self._portfolio.daily_returns)

    def get_equity_curve(self) -> list[tuple[datetime, float]]:
        """Return the equity curve as a list of (timestamp, equity) tuples.

        Returns:
            Chronologically ordered equity observations.
        """
        return list(self._equity_curve)

    def record_equity_point(self, timestamp: datetime) -> None:
        """Record a single equity observation at the given timestamp.

        Useful for intra-day equity tracking during backtesting.

        Args:
            timestamp: Observation time.
        """
        equity = self.get_equity()
        self._equity_curve.append((timestamp, equity))
