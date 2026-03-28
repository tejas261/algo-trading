"""Paper-trading execution adapter for backtesting and simulation."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from src.adapters.execution.base_execution_adapter import BaseExecutionAdapter
from src.models.fill import Fill
from src.models.order import Order, OrderSide, OrderStatus, OrderType
from src.models.position import Position
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _new_uuid() -> UUID:
    """Generate a new UUID."""
    return uuid.uuid4()


class PaperExecutionAdapter(BaseExecutionAdapter):
    """Simulated execution adapter that matches orders against candle data.

    This adapter is fully self-contained: it maintains an internal order
    book, position tracker, and fill history so that strategies can be
    back-tested without any external dependency.

    Parameters
    ----------
    initial_balance:
        Starting cash balance.
    slippage_pct:
        Percentage slippage applied to market order fills.  ``0.001``
        means 0.1 % (10 bps).
    commission_pct:
        Commission rate applied to every fill.  ``0.001`` means 0.1 %.
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        slippage_pct: float = 0.001,
        commission_pct: float = 0.001,
    ) -> None:
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._slippage_pct = slippage_pct
        self._commission_pct = commission_pct

        # Internal state --------------------------------------------------
        self._orders: dict[UUID, Order] = {}          # order_id -> Order
        self._fills: dict[UUID, list[Fill]] = {}      # order_id -> [Fill]
        self._positions: dict[str, Position] = {}     # symbol -> Position
        self._all_fills: list[Fill] = []

        # Current simulated market prices per symbol
        self._last_price: dict[str, float] = {}
        self._last_high: dict[str, float] = {}
        self._last_low: dict[str, float] = {}

        self._connected = False
        self._lock = threading.Lock()

        # Cumulative P&L
        self._realized_pnl: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        with self._lock:
            self._connected = True
        logger.info(
            "PaperExecutionAdapter connected (balance=%.2f, slippage=%.4f, commission=%.4f)",
            self._balance,
            self._slippage_pct,
            self._commission_pct,
        )

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False
        logger.info("PaperExecutionAdapter disconnected")

    # ------------------------------------------------------------------
    # Health & account
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        return self._connected

    def get_balance(self) -> float:
        with self._lock:
            return self._balance

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_positions(self) -> list[Position]:
        with self._lock:
            return list(self._positions.values())

    def get_open_orders(self) -> list[Order]:
        with self._lock:
            return [o for o in self._orders.values() if o.is_active]

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> Order:
        with self._lock:
            self._assert_connected()
            now = datetime.now(timezone.utc)
            order = Order(
                order_id=_new_uuid(),
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                status=OrderStatus.SUBMITTED,
                created_at=now,
            )
            self._orders[order.order_id] = order
            self._fills.setdefault(order.order_id, [])

            # Market orders fill immediately at last known price + slippage
            price = self._last_price.get(symbol)
            if price is not None:
                fill_price = self._apply_slippage(price, side)
                self._execute_fill(order, fill_price, quantity)
            else:
                logger.warning(
                    "No market price for %s; market order %s will fill on next update",
                    symbol,
                    order.order_id,
                )

            logger.info("Placed %s", order)
            return order

    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> Order:
        with self._lock:
            self._assert_connected()
            now = datetime.now(timezone.utc)
            order = Order(
                order_id=_new_uuid(),
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=price,
                status=OrderStatus.SUBMITTED,
                created_at=now,
            )
            self._orders[order.order_id] = order
            self._fills.setdefault(order.order_id, [])
            logger.info("Placed %s", order)

            # Check for immediate fill if limit is marketable
            self._try_fill_limit(order)

            return order

    def place_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
    ) -> Order:
        with self._lock:
            self._assert_connected()
            now = datetime.now(timezone.utc)
            order = Order(
                order_id=_new_uuid(),
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=stop_price,
                status=OrderStatus.PENDING,
                created_at=now,
            )
            self._orders[order.order_id] = order
            self._fills.setdefault(order.order_id, [])
            logger.info("Placed %s", order)
            return order

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str | UUID) -> bool:
        with self._lock:
            self._assert_connected()
            try:
                uid = order_id if isinstance(order_id, UUID) else UUID(order_id)
            except ValueError:
                logger.warning("cancel_order: invalid order_id %s", order_id)
                return False
            order = self._orders.get(uid)
            if order is None:
                logger.warning("cancel_order: unknown order_id %s", order_id)
                return False
            if not order.is_active:
                logger.warning(
                    "cancel_order: order %s is already %s",
                    order_id,
                    order.status.value,
                )
                return False
            now = datetime.now(timezone.utc)
            order.cancel(now)
            logger.info("Cancelled %s", order)
            return True

    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        with self._lock:
            self._assert_connected()
            now = datetime.now(timezone.utc)
            all_ok = True
            for order in list(self._orders.values()):
                if not order.is_active:
                    continue
                if symbol is not None and order.symbol != symbol:
                    continue
                order.cancel(now)
                logger.info("Cancelled %s", order)
            return all_ok

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def close_position(self, symbol: str) -> list[Fill]:
        with self._lock:
            self._assert_connected()
            pos = self._positions.get(symbol)
            if pos is None or pos.current_quantity == 0.0:
                logger.info("No open position in %s to close", symbol)
                return []

            # Close by submitting a market order on the opposite side
            close_side = OrderSide.SHORT if pos.side == OrderSide.LONG else OrderSide.LONG
            close_qty = pos.current_quantity

        # Release lock before calling place_market_order (which acquires it)
        order = self.place_market_order(symbol, close_side, close_qty)
        return self._fills.get(order.order_id, [])

    def close_all_positions(self) -> list[Fill]:
        with self._lock:
            symbols = list(self._positions.keys())

        all_fills: list[Fill] = []
        for sym in symbols:
            all_fills.extend(self.close_position(sym))
        return all_fills

    # ------------------------------------------------------------------
    # Fills & reconciliation
    # ------------------------------------------------------------------

    def fetch_fills(self, order_id: str | UUID) -> list[Fill]:
        with self._lock:
            uid = order_id if isinstance(order_id, UUID) else UUID(order_id)
            return list(self._fills.get(uid, []))

    def reconcile(self) -> dict:
        """In paper mode there is no external venue, so reconciliation is
        trivially consistent.  We just return current state summary."""
        with self._lock:
            return {
                "orders_synced": len(self._orders),
                "positions_synced": len(self._positions),
                "discrepancies": [],
                "balance": self._balance,
                "realized_pnl": self._realized_pnl,
                "open_orders": len([o for o in self._orders.values() if o.is_active]),
                "open_positions": len(
                    [p for p in self._positions.values() if p.current_quantity > 0]
                ),
            }

    # ------------------------------------------------------------------
    # Simulation driver
    # ------------------------------------------------------------------

    def update_market_price(
        self,
        symbol: str,
        price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> None:
        """Feed a new candle / tick into the simulator.

        This should be called once per candle during backtesting.  It updates
        the last-known price and attempts to match any resting orders.

        Parameters
        ----------
        symbol:
            The instrument whose price is being updated.
        price:
            The close (or last) price of the candle.
        high:
            The high of the candle.  Defaults to *price*.
        low:
            The low of the candle.  Defaults to *price*.
        """
        if high is None:
            high = price
        if low is None:
            low = price

        with self._lock:
            self._last_price[symbol] = price
            self._last_high[symbol] = high
            self._last_low[symbol] = low

            self._process_resting_orders(symbol, price, high, low)

    # ------------------------------------------------------------------
    # Private helpers (must be called under self._lock)
    # ------------------------------------------------------------------

    def _assert_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("Adapter is not connected. Call connect() first.")

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply slippage to *price* in the adverse direction."""
        if side == OrderSide.LONG:
            return price * (1.0 + self._slippage_pct)
        return price * (1.0 - self._slippage_pct)

    def _execute_fill(self, order: Order, fill_price: float, fill_qty: float) -> Fill:
        """Record a fill against *order* and update positions / balance."""
        now = datetime.now(timezone.utc)
        commission = fill_price * fill_qty * self._commission_pct
        slippage_cost = abs(fill_price - (self._last_price.get(order.symbol, fill_price))) * fill_qty
        fill = Fill(
            fill_id=_new_uuid(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            commission=commission,
            slippage=slippage_cost,
            timestamp=now,
        )

        # Update order state using the model's mark_filled method
        order.mark_filled(fill_qty, now)

        # Store fill
        self._fills[order.order_id].append(fill)
        self._all_fills.append(fill)

        # Update balance (deduct/add notional + commission)
        notional = fill_price * fill_qty
        if order.side == OrderSide.LONG:
            self._balance -= notional + commission
        else:
            self._balance += notional - commission

        # Update positions
        self._update_position(order.symbol, order.side, fill_qty, fill_price, fill)

        logger.debug("Executed %s", fill)
        return fill

    def _update_position(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        price: float,
        fill: Fill,
    ) -> None:
        """Adjust the position for *symbol* after a fill."""
        pos = self._positions.get(symbol)
        now = datetime.now(timezone.utc)

        if pos is None or pos.current_quantity == 0.0:
            # Open a new position
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=price,
                entry_quantity=qty,
                entry_fill=fill,
                entry_timestamp=now,
                current_quantity=qty,
            )
            return

        if pos.side == side:
            # Adding to existing position — create a fresh position that
            # aggregates the old and new quantities with a weighted avg price.
            total_qty = pos.current_quantity + qty
            new_avg = (
                (pos.entry_price * pos.current_quantity) + (price * qty)
            ) / total_qty
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                entry_price=new_avg,
                entry_quantity=total_qty,
                entry_fill=fill,
                entry_timestamp=pos.entry_timestamp,
                current_quantity=total_qty,
                partial_exits=list(pos.partial_exits),
            )
        else:
            # Reducing / flipping position
            if qty < pos.current_quantity:
                # Partial close
                partial = pos.apply_partial_exit(fill)
                pnl = partial.realized_pnl
                self._realized_pnl += pnl
            elif qty <= pos.current_quantity + 1e-9:
                # Full close
                partial = pos.apply_partial_exit(fill)
                pnl = partial.realized_pnl
                self._realized_pnl += pnl
            else:
                # Close + flip
                close_qty = pos.current_quantity
                # Create a fill for just the closing portion
                close_fill = Fill(
                    fill_id=_new_uuid(),
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    quantity=close_qty,
                    price=fill.price,
                    commission=fill.commission * (close_qty / qty),
                    slippage=fill.slippage * (close_qty / qty),
                    timestamp=fill.timestamp,
                )
                partial = pos.apply_partial_exit(close_fill)
                pnl = partial.realized_pnl
                self._realized_pnl += pnl
                remaining = qty - close_qty
                # Open new position on the other side
                self._positions[symbol] = Position(
                    symbol=symbol,
                    side=side,
                    entry_price=price,
                    entry_quantity=remaining,
                    entry_fill=fill,
                    entry_timestamp=now,
                    current_quantity=remaining,
                )

    def _process_resting_orders(
        self,
        symbol: str,
        price: float,
        high: float,
        low: float,
    ) -> None:
        """Try to fill resting limit and stop orders for *symbol*."""
        for order in list(self._orders.values()):
            if order.symbol != symbol or not order.is_active:
                continue

            if order.order_type == OrderType.MARKET and order.status == OrderStatus.SUBMITTED:
                # Unfilled market order (placed before first price update)
                fill_price = self._apply_slippage(price, order.side)
                self._execute_fill(order, fill_price, order.remaining_quantity)

            elif order.order_type == OrderType.STOP:
                self._try_trigger_stop(order, price, high, low)

            elif order.order_type == OrderType.LIMIT:
                self._try_fill_limit(order, high, low)

    def _try_trigger_stop(
        self,
        order: Order,
        price: float,
        high: float,
        low: float,
    ) -> None:
        """Check if a stop order should trigger."""
        if order.stop_price is None:
            return

        triggered = False
        if order.side == OrderSide.LONG and high >= order.stop_price:
            # Buy stop triggers when price rises to stop level
            triggered = True
        elif order.side == OrderSide.SHORT and low <= order.stop_price:
            # Sell stop triggers when price falls to stop level
            triggered = True

        if triggered:
            order.status = OrderStatus.SUBMITTED
            order.order_type = OrderType.MARKET
            fill_price = self._apply_slippage(order.stop_price, order.side)
            self._execute_fill(order, fill_price, order.remaining_quantity)
            logger.info("Stop triggered for %s at %.4f", order.order_id, order.stop_price)

    def _try_fill_limit(
        self,
        order: Order,
        high: Optional[float] = None,
        low: Optional[float] = None,
    ) -> None:
        """Check if a limit order can be filled."""
        if order.price is None or order.status != OrderStatus.SUBMITTED:
            return

        symbol = order.symbol
        if high is None:
            high = self._last_high.get(symbol, 0.0)
        if low is None:
            low = self._last_low.get(symbol, 0.0)

        filled = False
        if order.side == OrderSide.LONG and low <= order.price:
            # Buy limit fills when price drops to limit
            self._execute_fill(order, order.price, order.remaining_quantity)
            filled = True
        elif order.side == OrderSide.SHORT and high >= order.price:
            # Sell limit fills when price rises to limit
            self._execute_fill(order, order.price, order.remaining_quantity)
            filled = True

        if filled:
            logger.info("Limit order filled: %s @ %.4f", order.order_id, order.price)

    @staticmethod
    def _calc_pnl(
        side: OrderSide,
        entry_price: float,
        exit_price: float,
        quantity: float,
    ) -> float:
        """Calculate P&L for a directional trade."""
        if side == OrderSide.LONG:
            return (exit_price - entry_price) * quantity
        return (entry_price - exit_price) * quantity

    @staticmethod
    def _weighted_avg(
        old_avg: Optional[float],
        old_qty: float,
        new_price: float,
        new_qty: float,
    ) -> float:
        """Compute a weighted average price."""
        if old_avg is None or old_qty == 0.0:
            return new_price
        total = old_qty + new_qty
        return ((old_avg * old_qty) + (new_price * new_qty)) / total
