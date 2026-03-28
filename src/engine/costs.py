"""Transaction cost model for backtesting and live trading.

Handles maker/taker fees, dynamic slippage simulation, tax, perpetual funding
costs, and instrument-level order constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.models.order import OrderSide
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InstrumentSpec:
    """Constraints and metadata for a tradeable instrument.

    Attributes:
        instrument_type: ``"spot"`` or ``"perpetual"``.
        min_notional: Minimum order value in quote currency.
        min_qty: Minimum order quantity in base currency.
        qty_step: Smallest allowed quantity increment.
        price_tick: Smallest allowed price increment.
        assumed_daily_volume: Estimated daily volume used for size-based
            slippage scaling (quote currency).
    """

    instrument_type: str = "perpetual"
    min_notional: float = 10.0
    min_qty: float = 0.00001
    qty_step: float = 0.00001
    price_tick: float = 0.01
    assumed_daily_volume: float = 1_000_000_000.0


@dataclass
class CostModel:
    """Configurable transaction cost model with dynamic slippage.

    All fee / slippage parameters are expressed in basis points
    (1 bps = 0.01 %).

    Attributes:
        maker_fee_bps: Fee charged when providing liquidity (limit orders).
        taker_fee_bps: Fee charged when taking liquidity (market orders).
        slippage_bps: Legacy flat slippage — kept for backward compatibility.
            Overridden by *base_slippage_bps* when the dynamic model is used.
        tax_bps: Per-trade tax (e.g. STT, CTT) in basis points.
        funding_bps_per_8h: Perpetual futures funding rate per 8-hour period.
        base_slippage_bps: Starting slippage before dynamic adjustments.
        volatility_slippage_multiplier: Multiplier for the volatility term.
        size_slippage_multiplier: Multiplier for the order-size term.
        stress_slippage_multiplier: Extra factor applied on stop-loss fills.
        liquidation_slippage_multiplier: Extra factor applied on liquidation
            fills.
        min_slippage_bps: Floor for computed dynamic slippage.
        max_slippage_bps: Cap for computed dynamic slippage.
    """

    # --- existing fields (unchanged) ---
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    slippage_bps: float = 0.0
    tax_bps: float = 0.0
    funding_bps_per_8h: float = 0.0

    # --- dynamic slippage fields ---
    base_slippage_bps: float = 3.0
    volatility_slippage_multiplier: float = 1.0
    size_slippage_multiplier: float = 0.5
    stress_slippage_multiplier: float = 1.5
    liquidation_slippage_multiplier: float = 2.0
    min_slippage_bps: float = 1.0
    max_slippage_bps: float = 100.0

    # -- internal constants (not exposed as config) --
    _BASELINE_VOL: float = field(default=0.005, init=False, repr=False)

    # ------------------------------------------------------------------ #
    #  helpers
    # ------------------------------------------------------------------ #

    def _bps_to_rate(self, bps: float) -> float:
        """Convert basis points to a decimal rate."""
        return bps / 10_000.0

    # ------------------------------------------------------------------ #
    #  entry / exit costs (unchanged public API)
    # ------------------------------------------------------------------ #

    def calculate_entry_cost(
        self,
        price: float,
        quantity: float,
        is_maker: bool = False,
    ) -> float:
        """Calculate the total cost of entering a position.

        Includes trading fee and tax.  Slippage is applied separately via
        :meth:`apply_slippage` so that the fill price itself is adjusted.

        Args:
            price: Fill price (after any slippage adjustment).
            quantity: Number of units filled.
            is_maker: Whether the order was a maker (limit) fill.

        Returns:
            Total entry cost in quote currency.
        """
        notional = price * quantity
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee = notional * self._bps_to_rate(fee_bps)
        tax = notional * self._bps_to_rate(self.tax_bps)
        total = fee + tax
        logger.debug(
            "Entry cost: notional=%.4f fee=%.4f tax=%.4f total=%.4f",
            notional, fee, tax, total,
        )
        return total

    def calculate_exit_cost(
        self,
        price: float,
        quantity: float,
        is_maker: bool = False,
    ) -> float:
        """Calculate the total cost of exiting a position.

        Args:
            price: Fill price (after any slippage adjustment).
            quantity: Number of units filled.
            is_maker: Whether the order was a maker (limit) fill.

        Returns:
            Total exit cost in quote currency.
        """
        notional = price * quantity
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        fee = notional * self._bps_to_rate(fee_bps)
        tax = notional * self._bps_to_rate(self.tax_bps)
        total = fee + tax
        logger.debug(
            "Exit cost: notional=%.4f fee=%.4f tax=%.4f total=%.4f",
            notional, fee, tax, total,
        )
        return total

    # ------------------------------------------------------------------ #
    #  dynamic slippage
    # ------------------------------------------------------------------ #

    def _compute_dynamic_slippage_bps(
        self,
        price: float,
        atr: float | None = None,
        order_notional: float | None = None,
        assumed_daily_volume: float | None = None,
        is_stop_fill: bool = False,
        is_liquidation: bool = False,
    ) -> float:
        """Compute dynamic slippage in bps, applying vol / size / stress
        multipliers and clamping to [min, max].
        """
        slippage = self.base_slippage_bps

        # --- volatility factor ---
        vol_factor = 1.0
        if atr is not None and price > 0.0:
            vol_ratio = atr / price
            vol_factor = 1.0 + self.volatility_slippage_multiplier * max(
                0.0, vol_ratio / self._BASELINE_VOL - 1.0,
            )
        slippage *= vol_factor

        # --- size factor ---
        size_factor = 1.0
        if order_notional is not None:
            adv = assumed_daily_volume if assumed_daily_volume is not None else 1_000_000_000.0
            if adv > 0.0:
                size_factor = 1.0 + self.size_slippage_multiplier * max(
                    0.0, order_notional / adv - 0.01,
                ) * 100.0
        slippage *= size_factor

        # --- stress / liquidation multipliers ---
        if is_liquidation:
            slippage *= self.liquidation_slippage_multiplier
        elif is_stop_fill:
            slippage *= self.stress_slippage_multiplier

        # --- clamp ---
        slippage = max(self.min_slippage_bps, min(slippage, self.max_slippage_bps))

        logger.debug(
            "Dynamic slippage: base=%.2f vol_factor=%.3f size_factor=%.3f "
            "stop=%s liq=%s -> %.2f bps",
            self.base_slippage_bps, vol_factor, size_factor,
            is_stop_fill, is_liquidation, slippage,
        )
        return slippage

    def apply_slippage(
        self,
        price: float,
        side: OrderSide,
        atr: float | None = None,
        order_notional: float | None = None,
        candle_range: float | None = None,
        is_stop_fill: bool = False,
        is_liquidation: bool = False,
        assumed_daily_volume: float | None = None,
    ) -> float:
        """Adjust a price for estimated slippage.

        When called **without** any of the new optional parameters the
        behaviour is identical to the original flat-slippage model (using
        ``base_slippage_bps``).

        For LONG entries (buying), the fill price is moved *up* (worse).
        For SHORT entries (selling), the fill price is moved *down* (worse).

        Args:
            price: Theoretical fill price.
            side: Order side indicating direction.
            atr: Average True Range for volatility scaling (optional).
            order_notional: Order size in quote currency for size scaling
                (optional).
            candle_range: Reserved for future use (currently unused).
            is_stop_fill: ``True`` when the fill is triggered by a stop order.
            is_liquidation: ``True`` when the fill is a forced liquidation.
            assumed_daily_volume: Override the instrument's assumed daily
                volume for size-impact calculation.

        Returns:
            Slippage-adjusted fill price.
        """
        slippage_bps = self._compute_dynamic_slippage_bps(
            price=price,
            atr=atr,
            order_notional=order_notional,
            assumed_daily_volume=assumed_daily_volume,
            is_stop_fill=is_stop_fill,
            is_liquidation=is_liquidation,
        )
        slip_rate = self._bps_to_rate(slippage_bps)

        if side == OrderSide.LONG:
            adjusted = price * (1.0 + slip_rate)
        else:
            adjusted = price * (1.0 - slip_rate)
        return adjusted

    # ------------------------------------------------------------------ #
    #  funding (direction-aware)
    # ------------------------------------------------------------------ #

    def calculate_funding(
        self,
        notional: float,
        hours_held: float,
        side: OrderSide | None = None,
        funding_rate_bps: float | None = None,
    ) -> float:
        """Calculate cumulative perpetual funding cost or income.

        Funding is charged every 8 hours on perpetual futures.  This method
        pro-rates the amount based on the actual holding duration.

        The return value is **signed**:
        * Positive  -> the position holder pays funding.
        * Negative  -> the position holder receives funding.

        When *side* is ``None`` the legacy behaviour is preserved (always
        returns a non-negative cost).

        Args:
            notional: Absolute notional value of the position.
            hours_held: Total hours the position was held.
            side: ``OrderSide.LONG`` or ``OrderSide.SHORT``.  When provided
                the sign of the funding rate determines who pays.
            funding_rate_bps: Override the instance-level funding rate for
                this calculation.  If ``None``, ``self.funding_bps_per_8h``
                is used.

        Returns:
            Funding amount in quote currency (signed when *side* given).
        """
        rate_bps = funding_rate_bps if funding_rate_bps is not None else self.funding_bps_per_8h

        if rate_bps == 0.0 or hours_held <= 0.0:
            return 0.0

        periods = hours_held / 8.0
        raw = abs(notional) * self._bps_to_rate(rate_bps) * periods

        # Legacy path: no side provided -> always non-negative cost.
        if side is None:
            return abs(raw)

        # Direction-aware: positive rate means longs pay / shorts receive.
        if rate_bps > 0.0:
            return raw if side == OrderSide.LONG else -raw
        else:
            # Negative rate: shorts pay, longs receive.
            # *raw* is already negative because rate_bps < 0.
            return raw if side == OrderSide.LONG else -raw

    # ------------------------------------------------------------------ #
    #  instrument helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def round_quantity(qty: float, spec: InstrumentSpec) -> float:
        """Round *qty* down to the nearest ``spec.qty_step``.

        Args:
            qty: Desired quantity.
            spec: Instrument specification with step constraints.

        Returns:
            Quantity rounded to the closest valid step (toward zero).
        """
        if spec.qty_step <= 0:
            return qty
        # Use floor to avoid exceeding the requested quantity.
        steps = math.floor(abs(qty) / spec.qty_step)
        rounded = steps * spec.qty_step
        # Preserve the sign of the original qty.
        rounded = math.copysign(rounded, qty) if qty != 0.0 else 0.0
        # Avoid floating-point dust.
        decimals = max(0, -int(math.floor(math.log10(spec.qty_step)))) if spec.qty_step < 1 else 0
        rounded = round(rounded, decimals)
        return rounded

    @staticmethod
    def validate_order(
        qty: float,
        price: float,
        spec: InstrumentSpec,
    ) -> tuple[bool, str]:
        """Check whether an order satisfies instrument constraints.

        Args:
            qty: Order quantity (base currency).
            price: Intended order price.
            spec: Instrument specification.

        Returns:
            ``(True, "")`` if valid, ``(False, reason)`` otherwise.
        """
        abs_qty = abs(qty)
        if abs_qty < spec.min_qty:
            return False, (
                f"Quantity {abs_qty} below minimum {spec.min_qty}"
            )
        notional = abs_qty * price
        if notional < spec.min_notional:
            return False, (
                f"Notional {notional:.2f} below minimum {spec.min_notional:.2f}"
            )
        return True, ""

    # ------------------------------------------------------------------ #
    #  construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config: dict) -> CostModel:
        """Construct a CostModel from a configuration dictionary.

        Supports all new dynamic-slippage keys.  ``slippage_bps`` is used as
        a fallback for ``base_slippage_bps`` when the latter is absent, so
        legacy configs continue to work.

        Args:
            config: Dictionary with cost parameters.

        Returns:
            Configured CostModel instance.
        """
        legacy_slippage = float(config.get("slippage_bps", 0.0))
        base_slippage = float(
            config.get("base_slippage_bps", legacy_slippage if legacy_slippage else 3.0),
        )

        return cls(
            maker_fee_bps=float(config.get("maker_fee_bps", 0.0)),
            taker_fee_bps=float(config.get("taker_fee_bps", 0.0)),
            slippage_bps=legacy_slippage,
            tax_bps=float(config.get("tax_bps", 0.0)),
            funding_bps_per_8h=float(config.get("funding_bps_per_8h", 0.0)),
            base_slippage_bps=base_slippage,
            volatility_slippage_multiplier=float(
                config.get("volatility_slippage_multiplier", 1.0),
            ),
            size_slippage_multiplier=float(
                config.get("size_slippage_multiplier", 0.5),
            ),
            stress_slippage_multiplier=float(
                config.get("stress_slippage_multiplier", 1.5),
            ),
            liquidation_slippage_multiplier=float(
                config.get("liquidation_slippage_multiplier", 2.0),
            ),
            min_slippage_bps=float(config.get("min_slippage_bps", 1.0)),
            max_slippage_bps=float(config.get("max_slippage_bps", 100.0)),
        )
