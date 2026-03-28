"""Node: Load market data from CSV adapter."""

from __future__ import annotations

from src.agents.state import TradingState
from src.adapters.data.csv_data_adapter import CsvDataAdapter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_market_data_node(state: TradingState) -> dict:
    """Load OHLCV data for the configured symbol and timeframe.

    Reads from the CsvDataAdapter, validates the data, and produces a
    latest-bar snapshot.  Sets anomaly_flags if the data is empty, has
    gaps, or raises during loading.

    Returns a partial state update with:
        market_data, latest_snapshot, anomaly_flags
    """
    symbol = state.get("symbol", "")
    timeframe = state.get("timeframe", "1h")
    config = state.get("config", {})
    anomaly_flags: list[str] = list(state.get("anomaly_flags", []))

    data_dir = config.get("data_dir", "data")
    column_mapping = config.get("column_mapping")
    file_map = config.get("file_map")

    try:
        adapter = CsvDataAdapter(
            data_dir=data_dir,
            column_mapping=column_mapping,
            file_map=file_map,
        )
        adapter.connect()
        df = adapter.fetch_ohlcv(symbol, timeframe)
        adapter.disconnect()
    except FileNotFoundError as exc:
        logger.error("Market data file not found: %s", exc)
        anomaly_flags.append(f"data_file_not_found: {exc}")
        return {
            "market_data": None,
            "latest_snapshot": {},
            "anomaly_flags": anomaly_flags,
            "error": str(exc),
        }
    except Exception as exc:
        logger.error("Failed to load market data: %s", exc, exc_info=True)
        anomaly_flags.append(f"data_load_error: {exc}")
        return {
            "market_data": None,
            "latest_snapshot": {},
            "anomaly_flags": anomaly_flags,
            "error": str(exc),
        }

    if df.empty:
        logger.warning("Loaded market data is empty for %s/%s", symbol, timeframe)
        anomaly_flags.append("empty_market_data")
        return {
            "market_data": df,
            "latest_snapshot": {},
            "anomaly_flags": anomaly_flags,
        }

    # Check for data quality issues
    null_count = df[["open", "high", "low", "close", "volume"]].isnull().sum().sum()
    if null_count > 0:
        logger.warning("Market data contains %d null values", null_count)
        anomaly_flags.append(f"null_values_in_ohlcv: {null_count}")

    # Build latest-bar snapshot
    latest = df.iloc[-1]
    latest_snapshot = {
        "timestamp": latest.get("timestamp"),
        "open": float(latest["open"]),
        "high": float(latest["high"]),
        "low": float(latest["low"]),
        "close": float(latest["close"]),
        "volume": float(latest["volume"]),
        "bar_count": len(df),
    }

    logger.info(
        "Loaded %d bars for %s/%s. Latest close=%.4f",
        len(df), symbol, timeframe, latest_snapshot["close"],
    )

    return {
        "market_data": df,
        "latest_snapshot": latest_snapshot,
        "anomaly_flags": anomaly_flags,
    }
