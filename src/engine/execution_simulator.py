"""Candle-based execution simulator for backtesting.

Handles intra-candle logic for stop losses, partial profit targets,
and trailing stops with a conservative fill policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Sequence

from src.models.market_data import OHLCVBar
from src.models.order import OrderSide
from src.models.position import Position, PositionStatus, TrailingStopState
from src.models.trade_intent import PartialTarget
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EventType(str, Enum):
    """Types of execution events generated during candle processing."""

    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    PARTIAL_TARGET = "PARTIAL_TARGET"
    FULL_CLOSE = "FULL_CLOSE"


@dataclass
class ExecutionEvent:
    """An execution event produced by the simulator.

    Attributes:
        event_type: What triggered this event.
        fill_price: The simulated fill price.
        quantity: Number of units filled.
        timestamp: When the event occurred (candle timestamp).
        target_index: For partial targets, which target was hit (0-based).
        remaining_quantity: Position quantity after this event.
    """

    event_type: EventType
    fill_price: float
    quantity: float
    timestamp: datetime
    target_index: Optional[int] = None
    remaining_quantity: float = 0.0


class ExecutionSimulator:
    """Simulates order execution within OHLCV candles.

    The simulator uses a conservative fill policy:

    - If both stop and target could be hit within the same candle,
      the stop is assumed to hit first (worst-case assumption).
    - Stop fills occur at the stop price (no improvement assumed).
    - Target fills occur at the target price.
    - Trailing stop is updated based on candle extremes before
      checking for a stop trigger.

    Args:
        targets: Ordered list of partial profit targets from the trade intent.
        entry_price: The position's entry price.
    """

    def __init__(
        self,
        targets: list[PartialTarget] | None = None,
        entry_price: float = 0.0,
    ) -> None:
        self._targets = list(targets) if targets else []
        self._entry_price = entry_price
        self._next_target_idx: int = 0

    def process_candle(
        self,
        candle: OHLCVBar,
        position: Position,
    ) -> list[ExecutionEvent]:
        """Process a single candle against an open position.

        Checks in order:
        1. Update trailing stop using candle extremes (favourable direction first).
        2. Check stop loss / trailing stop trigger.
        3. Check partial profit targets.

        Conservative policy: if both stop and target could trigger in the
        same candle, the stop fires first and the target is skipped.

        Args:
            candle: The current OHLCV bar.
            position: The open position to evaluate.

        Returns:
            List of :class:`ExecutionEvent` objects.  May be empty if
            nothing triggered.
        """
        if position.status == PositionStatus.CLOSED:
            return []

        events: list[ExecutionEvent] = []
        is_long = position.side == OrderSide.LONG

        # --- 1. Update trailing stop with candle extreme in favourable direction ---
        if position.trailing_stop is not None:
            favourable_price = candle.high if is_long else candle.low
            position.trailing_stop.update(favourable_price, position.side)

        # --- 2. Check stop loss / trailing stop ---
        stop_price = self._get_effective_stop(position)
        stop_hit = False

        if stop_price is not None:
            if is_long and candle.low <= stop_price:
                stop_hit = True
            elif not is_long and candle.high >= stop_price:
                stop_hit = True

        # --- 3. Check targets (only if stop not hit) ---
        if stop_hit:
            # Conservative: stop fires first, close entire remaining position
            fill_price = stop_price
            # If the candle opened beyond the stop (gap), fill at the open
            if is_long and candle.open < stop_price:
                fill_price = candle.open
            elif not is_long and candle.open > stop_price:
                fill_price = candle.open

            event_type = EventType.TRAILING_STOP if (position.trailing_stop is not None and position.trailing_stop.activated) else EventType.STOP_LOSS

            events.append(ExecutionEvent(
                event_type=event_type,
                fill_price=fill_price,
                quantity=position.current_quantity,
                timestamp=candle.timestamp,
                remaining_quantity=0.0,
            ))
            logger.debug(
                "%s hit for %s @ %.4f (candle low=%.4f high=%.4f)",
                event_type.value, position.symbol, fill_price, candle.low, candle.high,
            )
            return events

        # Check partial targets
        targets_to_process = self._targets[self._next_target_idx:]
        remaining_qty = position.current_quantity

        for i, target in enumerate(targets_to_process):
            target_idx = self._next_target_idx + i

            target_hit = False
            if is_long and candle.high >= target.price:
                target_hit = True
            elif not is_long and candle.low <= target.price:
                target_hit = True

            if not target_hit:
                break  # Targets are ordered; if this one didn't hit, later ones won't either

            # Calculate quantity for this target
            exit_qty = (target.pct / 100.0) * position.entry_quantity
            exit_qty = min(exit_qty, remaining_qty)

            if exit_qty <= 1e-9:
                continue

            remaining_qty -= exit_qty
            is_final = remaining_qty <= 1e-9

            events.append(ExecutionEvent(
                event_type=EventType.FULL_CLOSE if is_final else EventType.PARTIAL_TARGET,
                fill_price=target.price,
                quantity=exit_qty,
                timestamp=candle.timestamp,
                target_index=target_idx,
                remaining_quantity=max(remaining_qty, 0.0),
            ))

            self._next_target_idx = target_idx + 1

            logger.debug(
                "Target %d hit for %s @ %.4f qty=%.4f remaining=%.4f",
                target_idx, position.symbol, target.price, exit_qty, remaining_qty,
            )

            if is_final:
                break

        return events

    def _get_effective_stop(self, position: Position) -> Optional[float]:
        """Get the current effective stop price.

        Prefers trailing stop if active, otherwise falls back to the
        static stop loss.

        Args:
            position: The position to query.

        Returns:
            The effective stop price, or ``None`` if no stop is set.
        """
        if position.trailing_stop is not None:
            return position.trailing_stop.current_stop
        return position.stop_loss

    def reset(self) -> None:
        """Reset the simulator state for a new position."""
        self._next_target_idx = 0
