"""Node: Run pre-trade risk checks."""

from __future__ import annotations

from src.agents.state import TradingState
from src.engine.risk import RiskDecision, RiskEngine
from src.models.order import OrderSide
from src.strategy.signals import SignalType
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_risk_checks_node(state: TradingState) -> dict:
    """Evaluate risk checks against the current signal and portfolio state.

    If there is no actionable signal, or if risk checks reject the trade,
    sets should_continue=False to skip downstream trade execution.

    Returns a partial state update with:
        risk_decision, should_continue
    """
    signal = state.get("signal", SignalType.NO_SIGNAL)
    config = state.get("config", {})
    risk_config = config.get("risk", {})
    indicators = state.get("indicators")
    portfolio_state = state.get("portfolio_state", {})

    # No signal means nothing to risk-check
    if signal == SignalType.NO_SIGNAL:
        logger.info("No signal to evaluate; skipping risk checks")
        return {
            "risk_decision": RiskDecision(
                approved=False,
                reasons=["no_actionable_signal"],
                position_size=0.0,
            ),
            "should_continue": False,
        }

    # Build the risk context from available state
    risk_context: dict = {}

    # Map signal to OrderSide
    direction = OrderSide.LONG if signal == SignalType.LONG else OrderSide.SHORT
    risk_context["direction"] = direction

    # Extract pricing info from indicators (latest bar)
    if indicators is not None and not indicators.empty:
        latest = indicators.iloc[-1]
        close = float(latest["close"])
        atr_value = float(latest.get("atr_14", 0.0))

        risk_context["entry_price"] = close

        # Stop price based on ATR
        atr_multiplier = float(risk_config.get("atr_stop_multiplier", 1.5))
        if direction == OrderSide.LONG:
            risk_context["stop_price"] = close - (atr_value * atr_multiplier)
        else:
            risk_context["stop_price"] = close + (atr_value * atr_multiplier)

        # Volatility filter context
        if "atr_14" in latest.index:
            risk_context["current_atr"] = atr_value
            # Use ATR SMA from the last 50 bars as rolling mean
            if len(indicators) >= 50:
                risk_context["rolling_atr_mean"] = float(
                    indicators["atr_14"].tail(50).mean()
                )

    # Portfolio context
    risk_context["equity"] = float(portfolio_state.get("equity", risk_config.get("initial_equity", 100_000.0)))
    risk_context["day_start_equity"] = float(portfolio_state.get("day_start_equity", risk_context["equity"]))
    risk_context["total_exposure"] = float(portfolio_state.get("total_exposure", 0.0))
    risk_context["trades_today"] = int(portfolio_state.get("trades_today", 0))
    risk_context["leverage"] = float(risk_config.get("leverage", 1.0))

    # Recent trades for consecutive loss check
    recent_trades = portfolio_state.get("recent_trades", [])
    if recent_trades:
        risk_context["recent_trades"] = recent_trades

    try:
        engine = RiskEngine(config=risk_config)
        decision = engine.run_all_checks(risk_context)
    except Exception as exc:
        logger.error("Risk engine failed: %s", exc, exc_info=True)
        decision = RiskDecision(
            approved=False,
            reasons=[f"risk_engine_error: {exc}"],
            position_size=0.0,
        )

    should_continue = decision.approved
    if not should_continue:
        logger.info("Risk checks rejected trade: %s", "; ".join(decision.reasons))
    else:
        logger.info(
            "Risk checks passed. Position size=%.4f",
            decision.position_size,
        )

    return {
        "risk_decision": decision,
        "should_continue": should_continue,
    }
