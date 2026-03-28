"""Backtester engine that ties together signals, risk, costs, and execution.

Iterates candle-by-candle with no look-ahead bias: signals are read from
the current candle's close, and entries are filled at the next candle's open.

Supports two exit modes (controlled by ``exit_mode`` in strategy config):
  - ``"trend_following"`` (default for v2): exit on EMA crossover OR trailing stop
  - ``"partial_targets"``: legacy partial profit ladder with trailing stop

Realism features:
  - Trailing stop wick ordering: check OLD stop against wick before updating
  - 8-hour funding cost tracking (constant / series / regime_approx modes)
  - Liquidation modeling for perpetual futures
  - Rolling 24-hour risk windows
  - Dynamic slippage (ATR-aware, stop-fill stress, liquidation stress)
  - Per-trade metadata (funding paid, exit reason, liquidation price, slippage)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd

from src.engine.costs import CostModel
from src.engine.metrics import compute_all_metrics
from src.engine.portfolio import PortfolioTracker
from src.engine.risk import RiskEngine
from src.models.market_data import OHLCVBar
from src.models.order import OrderSide
from src.models.position import Position
from src.models.results import BacktestResults, PerformanceMetrics
from src.models.trade import Trade, TradeStatus
from src.models.trade_intent import PartialTarget, TradeIntent
from src.strategy.signals import SignalType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Backtester:
    """Candle-by-candle backtesting engine.

    Execution flow per candle:
        1. If a pending entry exists from the previous bar's signal,
           fill it at the current bar's open.
        2. Check for exits on the current candle (liquidation, stop loss,
           trailing stop, EMA crossover, or opposite breakout depending on
           exit_mode).  Trailing stop is updated ONLY if no exit triggered.
        3. Apply funding costs every 8 hours for perpetual positions.
        4. Read the signal column for the current candle.
        5. If the signal indicates a new entry and no position is open,
           run risk checks and queue the entry for next-bar fill.

    At end-of-data, any remaining open position is force-closed at the
    last candle's close.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        strategy_config: dict[str, Any],
        risk_config: dict[str, Any] | None = None,
        cost_config: dict[str, Any] | None = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self._df = df.copy().reset_index(drop=True)
        self._strategy_config = strategy_config
        self._risk_engine = RiskEngine(risk_config)
        self._cost_model = CostModel.from_config(cost_config or {})
        self._portfolio = PortfolioTracker(initial_capital)
        self._initial_capital = initial_capital

        # Exit mode: "trend_following" (EMA cross + trailing) or "partial_targets" (legacy)
        self._exit_mode = strategy_config.get("exit_mode", "trend_following")
        self._trailing_stop_atr_mult = float(strategy_config.get("trailing_stop_atr_mult", 3.0))

        # Instrument / liquidation config
        self._instrument_type: str = strategy_config.get("instrument_type", "perpetual")
        self._leverage: float = float(strategy_config.get("leverage", 1.0))
        self._initial_margin_fraction: float = float(
            strategy_config.get("initial_margin_fraction", 1.0 / self._leverage)
        )
        self._maintenance_margin_fraction: float = float(
            strategy_config.get("maintenance_margin_fraction", 0.005)
        )
        self._liquidation_fee_bps: float = float(
            strategy_config.get("liquidation_fee_bps", 50)
        )

        # Funding config
        self._funding_mode: str = strategy_config.get("funding_mode", "constant")
        self._funding_column_name: str = strategy_config.get("funding_column_name", "funding_rate")

        # State
        self._trades: list[Trade] = []
        self._active_trade: Optional[Trade] = None
        self._pending_intent: Optional[TradeIntent] = None
        self._current_trailing_stop: Optional[float] = None
        self._liquidation_price: Optional[float] = None
        self._last_stopout_bar: dict[str, Optional[int]] = {
            OrderSide.LONG.value: None,
            OrderSide.SHORT.value: None,
        }

        # Rolling 24h risk window (Fix 4)
        self._recent_trade_timestamps: list[datetime] = []

        # Day boundary tracking (kept for daily snapshots)
        self._current_day: Optional[date] = None
        self._day_start_equity: float = initial_capital

        # Funding tracking (Fix 2)
        self._last_funding_ts: Optional[datetime] = None
        self._total_funding_paid: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResults:
        """Execute the backtest over all candles."""
        n = len(self._df)
        if n == 0:
            logger.warning("Empty DataFrame; returning empty results")
            return self._build_results()

        symbol = self._strategy_config.get("symbol", "UNKNOWN")
        logger.info(
            "Starting backtest: %d candles, initial_capital=%.2f, symbol=%s",
            n, self._initial_capital, symbol,
        )

        for i in range(n):
            row = self._df.iloc[i]
            candle = self._build_candle(row, symbol)
            candle_ts = candle.timestamp

            # Track daily boundaries
            self._handle_day_boundary(candle_ts)

            # Step 1: Fill any pending entry at this candle's open
            if self._pending_intent is not None:
                self._fill_pending_entry(candle, row, i)

            # Step 2: Process exits on current candle
            if self._active_trade is not None:
                self._check_exits(row, candle, symbol, candle_ts, i)

            # Step 3: Apply funding costs (perpetual only, every 8h)
            if self._active_trade is not None and self._instrument_type == "perpetual":
                self._apply_funding_if_due(row, candle, symbol, candle_ts)

            # Update mark prices for equity tracking
            self._portfolio.update_mark_prices({symbol: candle.close})

            # Step 4: Check signal for new entry (only if no open position)
            if self._active_trade is None and i < n - 1:
                signal = self._read_signal(row)
                if signal != SignalType.NO_SIGNAL:
                    self._evaluate_entry(row, signal, symbol, candle_ts, i)

            # Record equity point
            self._portfolio.record_equity_point(candle_ts)

        # Force-close at end of data
        if self._active_trade is not None:
            last_row = self._df.iloc[-1]
            last_candle = self._build_candle(last_row, symbol)
            self._force_close(symbol, last_candle.close, last_candle.timestamp)

        # Final daily snapshot
        if self._current_day is not None:
            last_row = self._df.iloc[-1]
            last_ts = self._parse_timestamp(last_row)
            self._portfolio.record_daily_snapshot(last_ts.date(), last_ts)

        logger.info(
            "Backtest complete: %d trades, final equity=%.2f",
            len(self._trades), self._portfolio.get_equity(),
        )

        return self._build_results()

    # ------------------------------------------------------------------
    # Internals: entry handling
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self,
        row: pd.Series,
        signal: SignalType,
        symbol: str,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Evaluate a signal and queue a trade intent if risk checks pass."""
        side = OrderSide.LONG if signal == SignalType.LONG else OrderSide.SHORT
        close = float(row["close"])

        atr = self._get_atr(row)
        if atr is None or atr <= 0:
            logger.debug("Skipping signal at bar %d: ATR unavailable", bar_idx)
            return

        # Calculate stop loss
        stop_mult = float(self._strategy_config.get("stop_loss_atr_mult", 2.0))
        if side == OrderSide.LONG:
            stop_loss = close - (atr * stop_mult)
        else:
            stop_loss = close + (atr * stop_mult)

        if stop_loss <= 0:
            logger.debug("Skipping signal at bar %d: invalid stop_loss=%.4f", bar_idx, stop_loss)
            return

        # Build risk context
        equity = self._portfolio.get_equity()
        context: dict[str, Any] = {
            "equity": equity,
            "entry_price": close,
            "stop_price": stop_loss,
            "leverage": self._leverage,
            "direction": side,
            "current_bar_idx": bar_idx,
            "last_stopout_bar_idx": self._last_stopout_bar.get(side.value),
            "recent_trades": self._trades[-20:] if self._trades else [],
            "day_start_equity": self._day_start_equity,
            "total_exposure": 0.0,
            # Rolling 24h risk window (Fix 4)
            "current_timestamp": timestamp,
        }

        # Volatility filter
        rolling_atr_mean = self._get_rolling_atr_mean(row)
        if atr is not None and rolling_atr_mean is not None and rolling_atr_mean > 0:
            context["current_atr"] = atr
            context["rolling_atr_mean"] = rolling_atr_mean

        decision = self._risk_engine.run_all_checks(context)
        if not decision.approved:
            logger.debug(
                "Signal rejected at bar %d: %s",
                bar_idx, "; ".join(decision.reasons),
            )
            return

        # Single target at a far-away price (trailing stop governs actual exit)
        far_mult = 10.0
        if side == OrderSide.LONG:
            target_price = close + (atr * far_mult)
        else:
            target_price = max(close - (atr * far_mult), atr * 0.01)

        targets = [PartialTarget(price=target_price, pct=100.0)]

        intent = TradeIntent(
            symbol=symbol,
            side=side,
            entry_price=close,
            stop_loss=stop_loss,
            targets=targets,
            position_size=decision.position_size,
            atr=atr,
            signal_timestamp=timestamp,
            metadata={"bar_idx": bar_idx, "signal": signal.value},
        )
        self._pending_intent = intent
        logger.debug(
            "Trade intent queued: %s %s size=%.4f entry~%.4f stop=%.4f",
            side.value, symbol, decision.position_size, close, stop_loss,
        )

    def _fill_pending_entry(self, candle: OHLCVBar, row: pd.Series, bar_idx: int) -> None:
        """Fill a pending trade intent at the current candle's open."""
        intent = self._pending_intent
        assert intent is not None
        self._pending_intent = None

        atr = intent.atr
        order_notional = candle.open * intent.position_size

        # Dynamic slippage on entry (Fix 5)
        fill_price = self._cost_model.apply_slippage(
            candle.open,
            intent.side,
            atr=atr,
            order_notional=order_notional,
        )
        cost = self._cost_model.calculate_entry_cost(fill_price, intent.position_size)

        # Check affordability
        notional = fill_price * intent.position_size
        if notional + cost > self._portfolio.cash:
            max_affordable = (self._portfolio.cash - cost) / fill_price
            if max_affordable <= 0:
                logger.debug("Insufficient cash for entry at bar %d", bar_idx)
                return
            adjusted_size = max_affordable * 0.99
            cost = self._cost_model.calculate_entry_cost(fill_price, adjusted_size)
        else:
            adjusted_size = intent.position_size

        if adjusted_size != intent.position_size:
            far_target = intent.targets[0]
            intent = TradeIntent(
                symbol=intent.symbol,
                side=intent.side,
                entry_price=intent.entry_price,
                stop_loss=intent.stop_loss,
                targets=[PartialTarget(price=far_target.price, pct=100.0)],
                position_size=adjusted_size,
                atr=intent.atr,
                signal_timestamp=intent.signal_timestamp,
                metadata=intent.metadata,
            )

        # Recalculate stop relative to actual fill price
        atr_val = intent.atr or 0.0
        stop_mult = float(self._strategy_config.get("stop_loss_atr_mult", 2.0))
        if intent.side == OrderSide.LONG:
            actual_stop = fill_price - (atr_val * stop_mult)
        else:
            actual_stop = fill_price + (atr_val * stop_mult)

        # Update intent with fill-price-based stop
        intent = TradeIntent(
            symbol=intent.symbol,
            side=intent.side,
            entry_price=fill_price,
            stop_loss=max(actual_stop, 0.01),
            targets=intent.targets,
            position_size=intent.position_size,
            atr=intent.atr,
            signal_timestamp=intent.signal_timestamp,
            metadata=intent.metadata,
        )

        position = self._portfolio.open_position(
            trade_intent=intent,
            fill_price=fill_price,
            fill_qty=intent.position_size,
            cost=cost,
            timestamp=candle.timestamp,
        )

        from src.models.fill import Fill
        from uuid import uuid4

        entry_fill = Fill(
            order_id=uuid4(),
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.position_size,
            price=fill_price,
            timestamp=candle.timestamp,
            commission=cost,
        )

        self._active_trade = Trade(
            symbol=intent.symbol,
            side=intent.side,
            entry_fill=entry_fill,
            entry_price=fill_price,
            entry_quantity=intent.position_size,
            entry_timestamp=candle.timestamp,
            initial_stop_loss=intent.stop_loss,
            initial_risk_per_share=abs(fill_price - intent.stop_loss),
        )

        # Initialize trailing stop at the initial stop level
        self._current_trailing_stop = intent.stop_loss

        # Calculate liquidation price (Fix 3)
        self._liquidation_price = None
        if self._instrument_type == "perpetual":
            self._liquidation_price = self._calculate_liquidation_price(
                fill_price, intent.side,
            )

        # Initialize funding tracking (Fix 2)
        self._last_funding_ts = candle.timestamp
        self._total_funding_paid = 0.0

        # Record trade timestamp for rolling window (Fix 4)
        self._recent_trade_timestamps.append(candle.timestamp)

        logger.info(
            "Filled entry: %s %s qty=%.4f @ %.4f stop=%.4f liq=%.4f (bar %d)",
            intent.side.value, intent.symbol, intent.position_size,
            fill_price, intent.stop_loss,
            self._liquidation_price or 0.0, bar_idx,
        )

    # ------------------------------------------------------------------
    # Internals: liquidation (Fix 3)
    # ------------------------------------------------------------------

    def _calculate_liquidation_price(self, entry_price: float, side: OrderSide) -> float:
        """Calculate the liquidation price for a leveraged perpetual position.

        For LONG:  liq = entry * (1 - 1/leverage * (1 - maint_margin))
        For SHORT: liq = entry * (1 + 1/leverage * (1 - maint_margin))
        """
        inv_lev = 1.0 / self._leverage
        margin_factor = 1.0 - self._maintenance_margin_fraction

        if side == OrderSide.LONG:
            return entry_price * (1.0 - inv_lev * margin_factor)
        else:
            return entry_price * (1.0 + inv_lev * margin_factor)

    def _check_liquidation(self, candle: OHLCVBar, is_long: bool) -> Optional[float]:
        """Check if the liquidation price was breached on this candle.

        Returns the exit price if liquidated, else None.
        Only applies to perpetual instruments.
        """
        if self._instrument_type != "perpetual":
            return None
        if self._liquidation_price is None:
            return None

        if is_long and candle.low <= self._liquidation_price:
            # Liquidated — exit at liquidation price (or gap-open if worse)
            if candle.open < self._liquidation_price:
                return candle.open
            return self._liquidation_price
        elif not is_long and candle.high >= self._liquidation_price:
            if candle.open > self._liquidation_price:
                return candle.open
            return self._liquidation_price

        return None

    # ------------------------------------------------------------------
    # Internals: funding (Fix 2)
    # ------------------------------------------------------------------

    def _apply_funding_if_due(
        self,
        row: pd.Series,
        candle: OHLCVBar,
        symbol: str,
        timestamp: datetime,
    ) -> None:
        """Apply funding cost if 8+ hours have passed since last charge."""
        if self._last_funding_ts is None:
            return

        elapsed = timestamp - self._last_funding_ts
        if elapsed < timedelta(hours=8):
            return

        trade = self._active_trade
        assert trade is not None

        position = self._get_open_position(symbol)
        if position is None:
            return

        notional = candle.close * position.current_quantity
        hours_elapsed = elapsed.total_seconds() / 3600.0

        # Determine funding rate based on mode
        funding_rate_bps = self._resolve_funding_rate(row, candle)

        funding_cost = self._cost_model.calculate_funding(
            notional=notional,
            hours_held=hours_elapsed,
            side=trade.side,
            funding_rate_bps=funding_rate_bps,
        )

        # Deduct from portfolio cash (positive = pay, negative = receive)
        self._portfolio.cash -= funding_cost
        self._total_funding_paid += funding_cost
        self._last_funding_ts = timestamp

        logger.debug(
            "Funding applied: %.4f (cumulative: %.4f) at %s, rate=%.2f bps",
            funding_cost, self._total_funding_paid, timestamp, funding_rate_bps or 0.0,
        )

    def _resolve_funding_rate(self, row: pd.Series, candle: OHLCVBar) -> Optional[float]:
        """Resolve the funding rate in bps based on the configured mode."""
        if self._funding_mode == "series":
            # Read from a column in the DataFrame
            col = self._funding_column_name
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
            return self._cost_model.funding_bps_per_8h

        if self._funding_mode == "regime_approx":
            # Approximate based on trend direction relative to EMA-200
            ema_200 = row.get("ema_200")
            configured_rate = self._cost_model.funding_bps_per_8h
            if pd.notna(ema_200) and configured_rate != 0.0:
                if candle.close > ema_200:
                    # Uptrend: longs pay positive funding
                    return abs(configured_rate)
                else:
                    # Downtrend: shorts receive, lower magnitude
                    return -abs(configured_rate) * 0.5
            return configured_rate

        # "constant" mode (default) — use the CostModel's configured rate
        return None  # calculate_funding will use self.funding_bps_per_8h

    # ------------------------------------------------------------------
    # Internals: exit handling
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        row: pd.Series,
        candle: OHLCVBar,
        symbol: str,
        timestamp: datetime,
        bar_idx: int,
    ) -> None:
        """Check all exit conditions for the current candle.

        Exit priority (Fix 1 + Fix 3):
            1. Liquidation check (perpetuals only) — exchange liquidates first
            2. Stop / trailing stop hit using OLD stop (before any update)
            3. EMA crossover (trend_following) or opposite breakout (donchian)
            4. If no exit, THEN update trailing stop using candle extreme
        """
        trade = self._active_trade
        assert trade is not None

        position = self._get_open_position(symbol)
        if position is None:
            return

        is_long = trade.side == OrderSide.LONG
        atr = self._get_atr(row)
        exit_reason: Optional[str] = None
        exit_price: Optional[float] = None

        # --- 1. Check liquidation FIRST (Fix 3) ---
        liq_exit = self._check_liquidation(candle, is_long)
        if liq_exit is not None:
            exit_price = liq_exit
            exit_reason = "LIQUIDATION"

        # --- 2. Check initial stop / trailing stop hit using OLD stop (Fix 1) ---
        if exit_reason is None:
            stop = self._current_trailing_stop
            if stop is not None:
                if is_long and candle.low <= stop:
                    exit_price = stop
                    # Gap down: fill at open if it opened below stop
                    if candle.open < stop:
                        exit_price = candle.open
                    exit_reason = "STOP"
                elif not is_long and candle.high >= stop:
                    exit_price = stop
                    if candle.open > stop:
                        exit_price = candle.open
                    exit_reason = "STOP"

        # --- 3. Check EMA crossover exit (trend_following mode) ---
        if exit_reason is None and self._exit_mode == "trend_following":
            ema_cross_exit = self._check_ema_crossover_exit(row, is_long)
            if ema_cross_exit:
                exit_price = candle.close
                exit_reason = "EMA_CROSS"

        # --- 3b. Check opposite breakout exit (donchian mode) ---
        if exit_reason is None and self._exit_mode == "donchian":
            opp_exit = self._check_opposite_breakout_exit(row, is_long)
            if opp_exit:
                exit_price = candle.close
                exit_reason = "OPPOSITE_BREAKOUT"

        # --- Execute exit ---
        if exit_reason is not None and exit_price is not None:
            self._close_trade(
                symbol=symbol,
                exit_price=exit_price,
                timestamp=timestamp,
                bar_idx=bar_idx,
                reason=exit_reason,
                atr=atr,
            )
            return

        # --- 4. No exit triggered: update trailing stop using candle extreme (Fix 1) ---
        if atr is not None and atr > 0 and self._current_trailing_stop is not None:
            trail_distance = atr * self._trailing_stop_atr_mult
            if is_long:
                # Use candle HIGH for long (favorable extreme), not close
                new_trail = candle.high - trail_distance
                self._current_trailing_stop = max(self._current_trailing_stop, new_trail)
            else:
                # Use candle LOW for short (favorable extreme), not close
                new_trail = candle.low + trail_distance
                self._current_trailing_stop = min(self._current_trailing_stop, new_trail)

    def _check_ema_crossover_exit(self, row: pd.Series, is_long: bool) -> bool:
        """Check if EMA(20) has crossed EMA(50) against the trade direction."""
        ema20 = row.get("ema_20")
        ema50 = row.get("ema_50")
        if pd.isna(ema20) or pd.isna(ema50):
            return False

        if is_long and ema20 < ema50:
            return True
        if not is_long and ema20 > ema50:
            return True
        return False

    def _check_opposite_breakout_exit(self, row: pd.Series, is_long: bool) -> bool:
        """Check if close has broken through the opposite 20-bar channel (donchian exit)."""
        if is_long:
            ll20 = row.get("lowest_low_20_close")
            if pd.notna(ll20) and row["close"] < ll20:
                return True
        else:
            hh20 = row.get("highest_high_20_close")
            if pd.notna(hh20) and row["close"] > hh20:
                return True
        return False

    def _close_trade(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        bar_idx: int,
        reason: str,
        atr: Optional[float] = None,
    ) -> None:
        """Close the active trade at the given price."""
        trade = self._active_trade
        assert trade is not None

        position = self._get_open_position(symbol)
        if position is None:
            return

        remaining = position.current_quantity

        # Dynamic slippage based on exit reason (Fix 5)
        is_stop = reason == "STOP"
        is_liq = reason == "LIQUIDATION"
        slipped_price = self._cost_model.apply_slippage(
            exit_price,
            trade.side,
            atr=atr,
            is_stop_fill=is_stop,
            is_liquidation=is_liq,
        )

        # Calculate effective slippage in bps for metadata
        if exit_price > 0:
            effective_slippage_bps = abs(slipped_price - exit_price) / exit_price * 10_000
        else:
            effective_slippage_bps = 0.0

        exit_cost = self._cost_model.calculate_exit_cost(slipped_price, remaining)

        # Add liquidation fee if applicable (Fix 3)
        if is_liq:
            liq_fee = abs(slipped_price * remaining) * (self._liquidation_fee_bps / 10_000.0)
            exit_cost += liq_fee

        self._portfolio.apply_partial_exit(
            symbol=symbol,
            quantity=remaining,
            exit_price=slipped_price,
            cost=exit_cost,
            timestamp=timestamp,
        )

        from src.models.fill import Fill
        from uuid import uuid4

        exit_fill = Fill(
            order_id=uuid4(),
            symbol=symbol,
            side=trade.side,
            quantity=remaining,
            price=slipped_price,
            timestamp=timestamp,
            commission=exit_cost,
        )

        # Store metadata before closing (Fix 6)
        trade.metadata = {
            "funding_paid": self._total_funding_paid,
            "exit_reason": reason,
            "effective_slippage_bps": round(effective_slippage_bps, 4),
        }
        if self._liquidation_price is not None:
            trade.metadata["liquidation_price"] = self._liquidation_price

        trade.close(exit_fill)
        self._trades.append(trade)

        logger.info(
            "Trade closed (%s): %s %s pnl=%.4f duration=%s at bar %d "
            "(funding=%.4f, slippage=%.2f bps)",
            reason, trade.side.value, symbol,
            trade.realized_pnl,
            trade.duration,
            bar_idx,
            self._total_funding_paid,
            effective_slippage_bps,
        )

        # Track stop-out for cooldown
        if reason == "STOP":
            self._last_stopout_bar[trade.side.value] = bar_idx

        self._active_trade = None
        self._current_trailing_stop = None
        self._liquidation_price = None
        self._total_funding_paid = 0.0
        self._last_funding_ts = None

    def _force_close(self, symbol: str, price: float, timestamp: datetime) -> None:
        """Force-close an open position at end of data."""
        if self._active_trade is None:
            return

        position = self._get_open_position(symbol)
        if position is None:
            return

        remaining = position.current_quantity
        atr_val = None  # No ATR context at force-close; use base slippage
        exit_price = self._cost_model.apply_slippage(price, self._active_trade.side, atr=atr_val)
        exit_cost = self._cost_model.calculate_exit_cost(exit_price, remaining)

        # Calculate effective slippage
        if price > 0:
            effective_slippage_bps = abs(exit_price - price) / price * 10_000
        else:
            effective_slippage_bps = 0.0

        self._portfolio.close_position(
            symbol=symbol,
            exit_price=exit_price,
            cost=exit_cost,
            timestamp=timestamp,
        )

        from src.models.fill import Fill
        from uuid import uuid4

        exit_fill = Fill(
            order_id=uuid4(),
            symbol=symbol,
            side=self._active_trade.side,
            quantity=remaining,
            price=exit_price,
            timestamp=timestamp,
            commission=exit_cost,
        )

        # Store metadata (Fix 6)
        self._active_trade.metadata = {
            "funding_paid": self._total_funding_paid,
            "exit_reason": "FORCE_CLOSE",
            "effective_slippage_bps": round(effective_slippage_bps, 4),
        }
        if self._liquidation_price is not None:
            self._active_trade.metadata["liquidation_price"] = self._liquidation_price

        self._active_trade.close(exit_fill)
        self._trades.append(self._active_trade)

        logger.info(
            "Force-closed at end of data: %s pnl=%.4f (funding=%.4f)",
            symbol, self._active_trade.realized_pnl, self._total_funding_paid,
        )
        self._active_trade = None
        self._current_trailing_stop = None
        self._liquidation_price = None
        self._total_funding_paid = 0.0
        self._last_funding_ts = None

    # ------------------------------------------------------------------
    # Internals: helpers
    # ------------------------------------------------------------------

    def _build_candle(self, row: pd.Series, symbol: str) -> OHLCVBar:
        return OHLCVBar(
            symbol=symbol,
            timestamp=self._parse_timestamp(row),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )

    def _parse_timestamp(self, row: pd.Series) -> datetime:
        if "timestamp" in row.index:
            ts = row["timestamp"]
        elif "datetime" in row.index:
            ts = row["datetime"]
        elif "date" in row.index:
            ts = row["date"]
        else:
            ts = row.name

        if isinstance(ts, str):
            return datetime.fromisoformat(ts)
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        if isinstance(ts, datetime):
            return ts
        return datetime.utcnow()

    def _read_signal(self, row: pd.Series) -> SignalType:
        signal_col = self._strategy_config.get("signal_column", "signal")
        if signal_col not in row.index:
            return SignalType.NO_SIGNAL

        val = row[signal_col]

        if isinstance(val, SignalType):
            return val
        if isinstance(val, str):
            try:
                return SignalType(val)
            except ValueError:
                return SignalType.NO_SIGNAL
        return SignalType.NO_SIGNAL

    def _get_atr(self, row: pd.Series) -> Optional[float]:
        atr_col = self._strategy_config.get("atr_column", "atr_14")
        if atr_col in row.index:
            val = row[atr_col]
            if pd.notna(val) and float(val) > 0:
                return float(val)
        return None

    def _get_rolling_atr_mean(self, row: pd.Series) -> Optional[float]:
        col = self._strategy_config.get("atr_mean_column", "atr_14_sma_50")
        if col in row.index:
            val = row[col]
            if pd.notna(val) and float(val) > 0:
                return float(val)
        return None

    def _get_open_position(self, symbol: str) -> Optional[Position]:
        return self._portfolio.positions.get(symbol)

    def _handle_day_boundary(self, timestamp: datetime) -> None:
        current_date = timestamp.date()
        if self._current_day is None:
            self._current_day = current_date
            self._day_start_equity = self._portfolio.get_equity()
            return

        if current_date != self._current_day:
            self._portfolio.record_daily_snapshot(self._current_day, timestamp)
            self._current_day = current_date
            self._day_start_equity = self._portfolio.get_equity()

    def _build_results(self) -> BacktestResults:
        equity_curve = self._portfolio.get_equity_curve()
        daily_returns_snapshots = self._portfolio.get_daily_returns()
        daily_return_pcts = [dr.daily_return_pct / 100.0 for dr in daily_returns_snapshots]

        if len(self._df) > 0:
            first_row = self._df.iloc[0]
            last_row = self._df.iloc[-1]
            start_date = self._parse_timestamp(first_row).date()
            end_date = self._parse_timestamp(last_row).date()
        else:
            start_date = date.today()
            end_date = date.today()

        symbol = self._strategy_config.get("symbol", "UNKNOWN")

        results = BacktestResults(
            strategy_name=self._strategy_config.get("strategy_name", "unnamed"),
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self._initial_capital,
            final_equity=self._portfolio.get_equity(),
            trade_log=self._trades,
            equity_curve=daily_returns_snapshots,
            daily_returns=daily_return_pcts,
        )

        if self._trades:
            equity_values = [eq for _, eq in equity_curve] if equity_curve else [self._initial_capital]

            # Pass timeframe and market_type for correct annualization (Fix 7)
            timeframe = self._strategy_config.get("timeframe", "1h")
            market_type = self._strategy_config.get("market_type", "crypto")

            metrics_dict = compute_all_metrics(
                trades=self._trades,
                equity_curve=equity_values,
                daily_returns=daily_return_pcts,
                timeframe=timeframe,
                market_type=market_type,
            )

            num_days = (end_date - start_date).days or 1
            years = num_days / 365.25
            annualized = 0.0
            if years > 0 and self._initial_capital > 0:
                final = self._portfolio.get_equity()
                annualized = ((final / self._initial_capital) ** (1.0 / years) - 1.0) * 100.0

            results.metrics = PerformanceMetrics(
                total_return_pct=metrics_dict["total_return_pct"],
                annualized_return_pct=round(annualized, 4),
                max_drawdown_pct=metrics_dict["max_drawdown_pct"],
                sharpe_ratio=metrics_dict["sharpe_ratio"] or None,
                sortino_ratio=metrics_dict["sortino_ratio"] or None,
                calmar_ratio=metrics_dict["calmar_ratio"] or None,
                total_trades=metrics_dict["total_trades"],
                winning_trades=metrics_dict["winning_trades"],
                losing_trades=metrics_dict["losing_trades"],
                win_rate_pct=metrics_dict["win_rate_pct"],
                avg_win_pct=metrics_dict["avg_win_pct"],
                avg_loss_pct=metrics_dict["avg_loss_pct"],
                profit_factor=metrics_dict.get("profit_factor"),
                expectancy=metrics_dict["expectancy"],
                avg_trade_duration_days=metrics_dict.get("avg_trade_duration_days"),
                max_consecutive_wins=metrics_dict["max_consecutive_wins"],
                max_consecutive_losses=metrics_dict["max_consecutive_losses"],
            )

        return results
