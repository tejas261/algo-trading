"""Node: Run the TrendBreakout strategy on market data."""

from __future__ import annotations

from src.agents.state import TradingState
from src.strategy.signals import SignalType
from src.strategy.strategy import TrendBreakoutStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_strategy_node(state: TradingState) -> dict:
    """Run TrendBreakoutStrategy on market data and extract the latest signal.

    This is a pure deterministic node -- no LLM involvement.  It computes
    all technical indicators and generates signals across the full
    DataFrame, then extracts the signal from the latest bar.

    Returns a partial state update with:
        indicators, signal
    """
    market_data = state.get("market_data")
    config = state.get("config", {})
    strategy_config = config.get("strategy", {})

    if market_data is None or market_data.empty:
        logger.warning("No market data available; returning NO_SIGNAL")
        return {
            "indicators": None,
            "signal": SignalType.NO_SIGNAL,
        }

    try:
        strategy = TrendBreakoutStrategy(config=strategy_config)
        result_df = strategy.run(market_data)
    except Exception as exc:
        logger.error("Strategy execution failed: %s", exc, exc_info=True)
        return {
            "indicators": None,
            "signal": SignalType.NO_SIGNAL,
            "error": str(exc),
        }

    # Extract the signal from the latest bar
    latest_signal = result_df["signal"].iloc[-1]
    if not isinstance(latest_signal, SignalType):
        latest_signal = SignalType.NO_SIGNAL

    logger.info(
        "Strategy produced signal=%s on latest bar (close=%.4f)",
        latest_signal.value,
        result_df["close"].iloc[-1],
    )

    return {
        "indicators": result_df,
        "signal": latest_signal,
    }
