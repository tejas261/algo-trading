"""Node: Build a deterministic TradeIntent from signal + risk + market data."""

from __future__ import annotations

from datetime import datetime, timezone

from src.agents.state import TradingState
from src.engine.risk import RiskDecision
from src.models.order import OrderSide
from src.models.trade_intent import PartialTarget, TradeIntent
from src.strategy.signals import SignalType
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_trade_intent_node(state: TradingState) -> dict:
    """Build a TradeIntent from the current signal, risk decision, and market data.

    Entry, stop-loss, and profit targets are calculated deterministically
    using ATR-based offsets.  No LLM involvement.

    Returns a partial state update with:
        trade_intent
    """
    signal = state.get("signal", SignalType.NO_SIGNAL)
    risk_decision: RiskDecision | None = state.get("risk_decision")
    indicators = state.get("indicators")
    symbol = state.get("symbol", "")
    config = state.get("config", {})
    strategy_config = config.get("strategy", {})

    # Guard: no signal or risk rejected
    if signal == SignalType.NO_SIGNAL:
        logger.info("No signal; skipping trade intent construction")
        return {"trade_intent": None}

    if risk_decision is None or not risk_decision.approved:
        logger.info("Risk not approved; skipping trade intent construction")
        return {"trade_intent": None}

    if indicators is None or indicators.empty:
        logger.warning("No indicator data available for trade intent")
        return {"trade_intent": None}

    latest = indicators.iloc[-1]
    close = float(latest["close"])
    atr_value = float(latest.get("atr_14", 0.0))

    if atr_value <= 0:
        logger.warning("ATR is zero or negative; cannot build trade intent")
        return {"trade_intent": None}

    # Direction
    side = OrderSide.LONG if signal == SignalType.LONG else OrderSide.SHORT

    # ATR multipliers (configurable)
    atr_stop_mult = float(strategy_config.get("atr_stop_multiplier", 1.5))
    atr_tp1_mult = float(strategy_config.get("atr_tp1_multiplier", 2.0))
    atr_tp2_mult = float(strategy_config.get("atr_tp2_multiplier", 3.0))
    atr_tp3_mult = float(strategy_config.get("atr_tp3_multiplier", 5.0))

    # Partial target allocation (configurable)
    tp1_pct = float(strategy_config.get("tp1_pct", 40.0))
    tp2_pct = float(strategy_config.get("tp2_pct", 35.0))
    tp3_pct = float(strategy_config.get("tp3_pct", 25.0))

    # Calculate levels
    if side == OrderSide.LONG:
        entry_price = close
        stop_loss = close - (atr_value * atr_stop_mult)
        tp1 = close + (atr_value * atr_tp1_mult)
        tp2 = close + (atr_value * atr_tp2_mult)
        tp3 = close + (atr_value * atr_tp3_mult)
    else:
        entry_price = close
        stop_loss = close + (atr_value * atr_stop_mult)
        tp1 = close - (atr_value * atr_tp1_mult)
        tp2 = close - (atr_value * atr_tp2_mult)
        tp3 = close - (atr_value * atr_tp3_mult)

    # Build targets
    targets = [
        PartialTarget(price=tp1, pct=tp1_pct),
        PartialTarget(price=tp2, pct=tp2_pct),
        PartialTarget(price=tp3, pct=tp3_pct),
    ]

    # Timestamp from the latest bar, or now
    signal_ts = latest.get("timestamp")
    if signal_ts is None:
        signal_ts = datetime.now(tz=timezone.utc)

    try:
        intent = TradeIntent(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            targets=targets,
            position_size=risk_decision.position_size,
            atr=atr_value,
            signal_timestamp=signal_ts,
            metadata={
                "signal": signal.value,
                "close": close,
                "atr_14": atr_value,
                "atr_stop_mult": atr_stop_mult,
            },
        )
    except Exception as exc:
        logger.error("Failed to build TradeIntent: %s", exc, exc_info=True)
        return {"trade_intent": None, "error": str(exc)}

    logger.info(
        "Built TradeIntent: %s %s @ %.4f, stop=%.4f, size=%.4f, targets=%d",
        side.value, symbol, entry_price, stop_loss,
        risk_decision.position_size, len(targets),
    )

    return {"trade_intent": intent}
