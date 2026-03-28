"""Abstract base class for all execution adapters."""

from __future__ import annotations

import abc
from typing import Optional

from src.models.fill import Fill
from src.models.order import Order, OrderSide, OrderType, OrderStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseExecutionAdapter(abc.ABC):
    """Unified interface for order execution.

    Concrete implementations include paper-trading simulators, live exchange
    connectors (e.g. Alpaca, Binance, Interactive Brokers), and replay
    engines.  All share the same public API so strategies are agnostic to
    the execution venue.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish a connection to the execution venue."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Gracefully release resources."""

    # ------------------------------------------------------------------
    # Health & account info
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` if the adapter is operational."""

    @abc.abstractmethod
    def get_balance(self) -> float:
        """Return the current account cash balance."""

    # ------------------------------------------------------------------
    # Positions & orders queries
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_positions(self) -> list:
        """Return a list of open :class:`Position` objects."""

    @abc.abstractmethod
    def get_open_orders(self) -> list:
        """Return a list of active (unfilled, uncancelled) :class:`Order` objects."""

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def place_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
    ) -> Order:
        """Submit a market order for immediate execution."""

    @abc.abstractmethod
    def place_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
    ) -> Order:
        """Submit a limit order.

        The order rests on the book until the market price reaches *price*
        or the order is cancelled.
        """

    @abc.abstractmethod
    def place_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float,
    ) -> Order:
        """Submit a stop order.

        The stop triggers when the market price crosses *stop_price*, at
        which point it converts to a market order.
        """

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order.  Return ``True`` on success."""

    @abc.abstractmethod
    def cancel_all_orders(self, symbol: Optional[str] = None) -> bool:
        """Cancel all open orders, optionally filtered by *symbol*.

        Return ``True`` if all targeted orders were successfully cancelled.
        """

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def close_position(self, symbol: str) -> list[Fill]:
        """Flatten the position in *symbol* and return resulting fills."""

    @abc.abstractmethod
    def close_all_positions(self) -> list[Fill]:
        """Flatten every open position and return all resulting fills."""

    # ------------------------------------------------------------------
    # Fills & reconciliation
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fetch_fills(self, order_id: str) -> list[Fill]:
        """Return all fills associated with *order_id*."""

    @abc.abstractmethod
    def reconcile(self) -> dict:
        """Reconcile internal state with the execution venue.

        Returns a dict summarising discrepancies (if any).  The exact
        schema is implementation-defined but should at minimum include::

            {
                "orders_synced": int,
                "positions_synced": int,
                "discrepancies": list[str],
            }
        """

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseExecutionAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.disconnect()
