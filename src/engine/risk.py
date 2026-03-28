"""Risk engine with config-driven pre-trade checks and position sizing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from src.models.order import OrderSide
from src.models.trade import Trade, TradeStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    """Outcome of a risk check pass.

    Attributes:
        approved: Whether the trade is permitted.
        reasons: Human-readable explanations for any rejections.
        position_size: Approved position size (0.0 if rejected).
    """

    approved: bool
    reasons: list[str]
    position_size: float


class RiskEngine:
    """Config-driven risk management engine.

    Provides individual check methods and a composite :meth:`run_all_checks`
    that evaluates every rule and returns a single :class:`RiskDecision`.

    Args:
        config: Dictionary of risk parameters.  Supported keys:

            - ``risk_per_trade_pct`` (float): Max equity risked per trade.
            - ``max_leverage`` (float): Maximum allowed leverage.
            - ``cooldown_bars`` (int): Bars to wait after a stop-out.
            - ``max_trades_per_day`` (int): Daily trade cap.
            - ``max_consecutive_losses`` (int): Loss streak limit.
            - ``volatility_max_ratio`` (float): ATR spike filter.
            - ``max_daily_drawdown_pct`` (float): Intraday drawdown limit.
            - ``max_exposure_pct`` (float): Portfolio exposure cap.
            - ``kill_switch`` (bool): Emergency off switch.
            - ``risk_window_mode`` (str): ``"rolling_24h"`` or ``"calendar_day"``.
            - ``risk_window_hours`` (int): Rolling window size in hours.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.risk_per_trade_pct: float = float(cfg.get("risk_per_trade_pct", 1.0))
        self.max_leverage: float = float(cfg.get("max_leverage", 1.0))
        self.cooldown_bars: int = int(cfg.get("cooldown_bars", 0))
        self.max_trades_per_day: int = int(cfg.get("max_trades_per_day", 999))
        self.max_consecutive_losses: int = int(cfg.get("max_consecutive_losses", 999))
        self.volatility_max_ratio: float = float(cfg.get("volatility_max_ratio", 999.0))
        self.max_daily_drawdown_pct: float = float(cfg.get("max_daily_drawdown_pct", 100.0))
        self.max_exposure_pct: float = float(cfg.get("max_exposure_pct", 100.0))
        self._kill_switch: bool = bool(cfg.get("kill_switch", False))
        self.risk_window_mode: str = str(cfg.get("risk_window_mode", "rolling_24h"))
        self.risk_window_hours: int = int(cfg.get("risk_window_hours", 24))

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def calculate_position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        risk_pct: float | None = None,
        leverage: float = 1.0,
    ) -> float:
        """Calculate position size based on fixed-fractional risk.

        Formula:
            risk_amount = equity * risk_pct / 100
            risk_per_unit = |entry_price - stop_price|
            raw_size = risk_amount / risk_per_unit
            leveraged_max = (equity * leverage) / entry_price
            size = min(raw_size, leveraged_max)

        Args:
            equity: Current account equity.
            entry_price: Expected entry price.
            stop_price: Stop-loss price.
            risk_pct: Percentage of equity to risk (overrides config default).
            leverage: Leverage multiplier.

        Returns:
            Position size in base units (floored to avoid fractional
            over-sizing issues).
        """
        if equity <= 0 or entry_price <= 0:
            return 0.0

        pct = risk_pct if risk_pct is not None else self.risk_per_trade_pct
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            logger.warning("Risk per unit is zero; entry=%.4f stop=%.4f", entry_price, stop_price)
            return 0.0

        risk_amount = equity * (pct / 100.0)
        raw_size = risk_amount / risk_per_unit

        effective_leverage = min(leverage, self.max_leverage)
        leveraged_max = (equity * effective_leverage) / entry_price
        size = min(raw_size, leveraged_max)

        return max(size, 0.0)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_cooldown(
        self,
        last_stopout_bar_idx: int | None,
        current_bar_idx: int,
        direction: OrderSide,
        cooldown_bars: int | None = None,
    ) -> bool:
        """Return True if cooldown period has elapsed since last stop-out.

        Args:
            last_stopout_bar_idx: Bar index of the most recent stop-out for
                this direction.  ``None`` means no prior stop-out.
            current_bar_idx: Current bar index.
            direction: Side that was stopped out (used for directional cooldown).
            cooldown_bars: Override for the engine-level cooldown setting.

        Returns:
            ``True`` if the trade is allowed (cooldown elapsed or no prior stop).
        """
        if last_stopout_bar_idx is None:
            return True
        bars = cooldown_bars if cooldown_bars is not None else self.cooldown_bars
        elapsed = current_bar_idx - last_stopout_bar_idx
        return elapsed >= bars

    def check_daily_trade_limit(
        self,
        trades_today: int,
        max_per_day: int | None = None,
    ) -> bool:
        """Return True if the daily trade count is below the limit.

        Args:
            trades_today: Number of trades already taken today.
            max_per_day: Override for the engine-level daily cap.

        Returns:
            ``True`` if another trade is permitted.
        """
        limit = max_per_day if max_per_day is not None else self.max_trades_per_day
        return trades_today < limit

    def check_consecutive_losses(
        self,
        recent_trades: list[Trade],
        max_consecutive: int | None = None,
    ) -> bool:
        """Return True if the consecutive loss streak is below the limit.

        Args:
            recent_trades: List of recent closed trades in chronological order.
            max_consecutive: Override for the engine-level streak limit.

        Returns:
            ``True`` if the streak is acceptable.
        """
        limit = max_consecutive if max_consecutive is not None else self.max_consecutive_losses
        streak = 0
        for trade in reversed(recent_trades):
            if trade.status != TradeStatus.CLOSED:
                continue
            if trade.realized_pnl <= 0:
                streak += 1
            else:
                break
        return streak < limit

    def check_volatility_filter(
        self,
        current_atr: float,
        rolling_atr_mean: float,
        max_ratio: float | None = None,
    ) -> bool:
        """Return True if current ATR is within acceptable range of its mean.

        Args:
            current_atr: Current-bar ATR value.
            rolling_atr_mean: Rolling mean of ATR (e.g. 50-bar SMA of ATR).
            max_ratio: Maximum allowed ratio of current ATR to mean.

        Returns:
            ``True`` if volatility is acceptable.
        """
        ratio = max_ratio if max_ratio is not None else self.volatility_max_ratio
        if rolling_atr_mean <= 0:
            return True
        return (current_atr / rolling_atr_mean) <= ratio

    def check_daily_drawdown(
        self,
        current_equity: float,
        day_start_equity: float,
        max_dd_pct: float | None = None,
    ) -> bool:
        """Return True if the intraday drawdown is within limits.

        Args:
            current_equity: Current portfolio equity.
            day_start_equity: Equity at the start of the trading day.
            max_dd_pct: Maximum intraday drawdown percentage allowed.

        Returns:
            ``True`` if drawdown is acceptable.
        """
        limit = max_dd_pct if max_dd_pct is not None else self.max_daily_drawdown_pct
        if day_start_equity <= 0:
            return True
        dd_pct = ((day_start_equity - current_equity) / day_start_equity) * 100.0
        return dd_pct < limit

    def check_max_exposure(
        self,
        total_exposure: float,
        equity: float,
        max_pct: float | None = None,
    ) -> bool:
        """Return True if total portfolio exposure is below the cap.

        Args:
            total_exposure: Sum of absolute notional values of all open positions.
            equity: Current portfolio equity.
            max_pct: Maximum exposure as a percentage of equity.

        Returns:
            ``True`` if exposure is acceptable.
        """
        limit = max_pct if max_pct is not None else self.max_exposure_pct
        if equity <= 0:
            return False
        exposure_pct = (total_exposure / equity) * 100.0
        return exposure_pct < limit

    def check_kill_switch(self) -> bool:
        """Return True if the kill switch is NOT engaged (trading is allowed).

        Returns:
            ``True`` if trading may proceed; ``False`` if the kill switch
            has been activated.
        """
        return not self._kill_switch

    def set_kill_switch(self, active: bool) -> None:
        """Activate or deactivate the kill switch.

        Args:
            active: ``True`` to halt all new trades.
        """
        self._kill_switch = active
        if active:
            logger.warning("Kill switch ACTIVATED -- all new trades blocked")
        else:
            logger.info("Kill switch deactivated")

    # ------------------------------------------------------------------
    # Rolling-window checks
    # ------------------------------------------------------------------

    def check_rolling_trade_limit(
        self,
        recent_trades: list[Trade],
        current_timestamp: datetime,
        window_hours: int | None = None,
        max_trades: int | None = None,
    ) -> bool:
        """Return True if the number of trades within the rolling window is under the limit.

        Args:
            recent_trades: List of trades in chronological order.
            current_timestamp: The current point in time.
            window_hours: Size of the rolling window in hours.  Falls back
                to ``self.risk_window_hours``.
            max_trades: Maximum trades allowed in the window.  Falls back
                to ``self.max_trades_per_day``.

        Returns:
            ``True`` if another trade is permitted within the window.
        """
        hours = window_hours if window_hours is not None else self.risk_window_hours
        limit = max_trades if max_trades is not None else self.max_trades_per_day
        cutoff = current_timestamp - timedelta(hours=hours)

        count = sum(
            1
            for t in recent_trades
            if t.status == TradeStatus.CLOSED
            and t.exit_timestamp is not None
            and t.exit_timestamp >= cutoff
        )
        return count < limit

    def check_rolling_consecutive_losses(
        self,
        recent_trades: list[Trade],
        current_timestamp: datetime,
        window_hours: int | None = None,
        max_consecutive: int | None = None,
    ) -> bool:
        """Return True if the consecutive loss streak within the rolling window is below the limit.

        Same logic as :meth:`check_consecutive_losses` but only considers
        trades whose exit time falls within the last *window_hours* hours.

        Args:
            recent_trades: List of trades in chronological order.
            current_timestamp: The current point in time.
            window_hours: Size of the rolling window in hours.
            max_consecutive: Maximum consecutive losses allowed.

        Returns:
            ``True`` if the streak is acceptable.
        """
        hours = window_hours if window_hours is not None else self.risk_window_hours
        limit = max_consecutive if max_consecutive is not None else self.max_consecutive_losses
        cutoff = current_timestamp - timedelta(hours=hours)

        # Filter to closed trades within the window
        windowed = [
            t
            for t in recent_trades
            if t.status == TradeStatus.CLOSED
            and t.exit_timestamp is not None
            and t.exit_timestamp >= cutoff
        ]

        streak = 0
        for trade in reversed(windowed):
            if trade.realized_pnl <= 0:
                streak += 1
            else:
                break
        return streak < limit

    # ------------------------------------------------------------------
    # Position validation
    # ------------------------------------------------------------------

    def validate_position_size(
        self,
        quantity: float,
        price: float,
        instrument_spec: dict | None = None,
    ) -> tuple[float, list[str]]:
        """Validate and adjust a position size against instrument specifications.

        Checks minimum notional value, minimum quantity, and rounds to the
        instrument's quantity step size.

        Args:
            quantity: Desired position size in base units.
            price: Expected execution price.
            instrument_spec: Optional dictionary with instrument constraints.
                Supported keys:

                - ``min_notional`` (float): Minimum order notional value.
                - ``min_qty`` (float): Minimum order quantity.
                - ``qty_step`` (float): Quantity step/tick size for rounding.

        Returns:
            A tuple of ``(adjusted_quantity, warnings)`` where *warnings*
            is a list of human-readable strings describing any adjustments
            or issues.
        """
        warnings: list[str] = []
        spec = instrument_spec or {}

        min_notional = float(spec.get("min_notional", 0.0))
        min_qty = float(spec.get("min_qty", 0.0))
        qty_step = float(spec.get("qty_step", 0.0))

        adjusted = quantity

        # Round to qty_step first (before other checks)
        if qty_step > 0:
            adjusted = (adjusted // qty_step) * qty_step
            if adjusted != quantity:
                warnings.append(
                    f"Quantity rounded from {quantity} to {adjusted} "
                    f"(qty_step={qty_step})"
                )

        # Check minimum quantity
        if min_qty > 0 and adjusted < min_qty:
            warnings.append(
                f"Quantity {adjusted} below minimum {min_qty}; "
                f"adjusted up to min_qty"
            )
            adjusted = min_qty

        # Check minimum notional
        if min_notional > 0 and price > 0:
            notional = adjusted * price
            if notional < min_notional:
                required_qty = min_notional / price
                # Round required_qty up to qty_step
                if qty_step > 0:
                    import math as _math
                    required_qty = _math.ceil(required_qty / qty_step) * qty_step
                warnings.append(
                    f"Notional {notional:.2f} below minimum {min_notional:.2f}; "
                    f"adjusted quantity from {adjusted} to {required_qty}"
                )
                adjusted = required_qty

        return (adjusted, warnings)

    # ------------------------------------------------------------------
    # Composite check
    # ------------------------------------------------------------------

    def run_all_checks(self, context: dict[str, Any]) -> RiskDecision:
        """Evaluate all risk checks against the provided context.

        The *context* dictionary should contain whichever keys are relevant.
        Missing keys cause the corresponding check to be skipped (assumed
        passing).

        Supported context keys:
            - ``equity`` (float): Current equity.
            - ``entry_price`` (float): Expected entry price.
            - ``stop_price`` (float): Stop-loss price.
            - ``leverage`` (float): Desired leverage.
            - ``last_stopout_bar_idx`` (int | None): Bar index of last stop-out.
            - ``current_bar_idx`` (int): Current bar index.
            - ``direction`` (OrderSide): Trade direction.
            - ``trades_today`` (int): Number of trades taken today.
            - ``recent_trades`` (list[Trade]): Recent closed trades.
            - ``current_atr`` (float): Current ATR.
            - ``rolling_atr_mean`` (float): Rolling mean ATR.
            - ``day_start_equity`` (float): Equity at start of day.
            - ``total_exposure`` (float): Current total exposure.
            - ``current_timestamp`` (datetime): Current timestamp (used
              for rolling-window checks).

        Returns:
            A :class:`RiskDecision` with approval status, rejection reasons,
            and the approved position size.
        """
        reasons: list[str] = []
        position_size = 0.0

        # Kill switch
        if not self.check_kill_switch():
            reasons.append("Kill switch is active")

        # Cooldown
        if "current_bar_idx" in context and "direction" in context:
            if not self.check_cooldown(
                context.get("last_stopout_bar_idx"),
                context["current_bar_idx"],
                context["direction"],
            ):
                reasons.append(
                    f"Cooldown active: {self.cooldown_bars} bars required after stop-out"
                )

        # Daily trade limit / rolling trade limit
        if self.risk_window_mode == "rolling_24h" and "recent_trades" in context and "current_timestamp" in context:
            if not self.check_rolling_trade_limit(
                context["recent_trades"],
                context["current_timestamp"],
            ):
                reasons.append(
                    f"Rolling {self.risk_window_hours}h trade limit reached: "
                    f"max {self.max_trades_per_day}"
                )
        elif "trades_today" in context:
            if not self.check_daily_trade_limit(context["trades_today"]):
                reasons.append(
                    f"Daily trade limit reached: {context['trades_today']}/{self.max_trades_per_day}"
                )

        # Consecutive losses (rolling or calendar)
        if self.risk_window_mode == "rolling_24h" and "recent_trades" in context and "current_timestamp" in context:
            if not self.check_rolling_consecutive_losses(
                context["recent_trades"],
                context["current_timestamp"],
            ):
                reasons.append(
                    f"Consecutive loss limit reached within {self.risk_window_hours}h window: "
                    f"{self.max_consecutive_losses}"
                )
        elif "recent_trades" in context:
            if not self.check_consecutive_losses(context["recent_trades"]):
                reasons.append(
                    f"Consecutive loss limit reached: {self.max_consecutive_losses}"
                )

        # Volatility filter
        if "current_atr" in context and "rolling_atr_mean" in context:
            if not self.check_volatility_filter(
                context["current_atr"],
                context["rolling_atr_mean"],
            ):
                reasons.append(
                    f"Volatility too high: ATR ratio exceeds {self.volatility_max_ratio:.2f}"
                )

        # Daily drawdown
        if "day_start_equity" in context and "equity" in context:
            if not self.check_daily_drawdown(context["equity"], context["day_start_equity"]):
                reasons.append(
                    f"Daily drawdown limit breached: max {self.max_daily_drawdown_pct:.1f}%"
                )

        # Max exposure
        if "total_exposure" in context and "equity" in context:
            if not self.check_max_exposure(context["total_exposure"], context["equity"]):
                reasons.append(
                    f"Exposure limit breached: max {self.max_exposure_pct:.1f}% of equity"
                )

        # Position sizing (only compute if no blockers so far)
        if not reasons:
            equity = context.get("equity", 0.0)
            entry_price = context.get("entry_price", 0.0)
            stop_price = context.get("stop_price", 0.0)
            leverage = context.get("leverage", 1.0)

            if equity > 0 and entry_price > 0 and stop_price != entry_price:
                position_size = self.calculate_position_size(
                    equity=equity,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    leverage=leverage,
                )
                if position_size <= 0:
                    reasons.append("Calculated position size is zero or negative")

        approved = len(reasons) == 0
        if not approved:
            logger.info("Risk checks FAILED: %s", "; ".join(reasons))

        return RiskDecision(
            approved=approved,
            reasons=reasons,
            position_size=position_size,
        )
