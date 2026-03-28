"""Abstract base class for all data adapters."""

from __future__ import annotations

import abc
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

OHLCV_COLUMNS: list[str] = ["timestamp", "open", "high", "low", "close", "volume"]


class DataValidationError(Exception):
    """Raised when fetched data does not conform to the OHLCV schema."""


class BaseDataAdapter(abc.ABC):
    """Abstract interface that every data adapter must implement.

    Concrete subclasses provide data from CSV files, databases, REST APIs,
    websocket streams, etc.  All of them must return DataFrames that conform
    to the canonical OHLCV schema defined by ``OHLCV_COLUMNS``.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish a connection to the data source.

        Implementations may open file handles, authenticate with APIs,
        create database connections, etc.
        """

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Release resources held by the adapter."""

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data for *symbol* at the given *timeframe*.

        Parameters
        ----------
        symbol:
            Trading pair / ticker, e.g. ``"BTC/USDT"`` or ``"AAPL"``.
        timeframe:
            Candle interval, e.g. ``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``.
        start:
            Inclusive lower bound on the timestamp.  ``None`` means "from
            the earliest available record".
        end:
            Inclusive upper bound on the timestamp.  ``None`` means "up to
            the most recent record".

        Returns
        -------
        pd.DataFrame
            A DataFrame whose columns match ``OHLCV_COLUMNS``, sorted by
            *timestamp* ascending.
        """

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Validate (and lightly coerce) a DataFrame to the OHLCV schema.

        Checks
        ------
        1. All required columns are present.
        2. ``timestamp`` is a ``datetime64`` type.
        3. OHLCV numeric columns are numeric.
        4. No null values in required columns.
        5. Data is sorted by timestamp ascending.

        Returns the validated DataFrame (potentially with coerced dtypes).

        Raises
        ------
        DataValidationError
            If the DataFrame cannot be made conformant.
        """
        if df.empty:
            logger.warning("validate_ohlcv received an empty DataFrame")
            return df

        # 1. Column presence
        missing = set(OHLCV_COLUMNS) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )

        # Restrict to canonical columns (keep order)
        df = df[OHLCV_COLUMNS].copy()

        # 2. Timestamp dtype
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            except Exception as exc:
                raise DataValidationError(
                    f"Could not convert 'timestamp' column to datetime: {exc}"
                ) from exc

        # 3. Numeric columns
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception as exc:
                    raise DataValidationError(
                        f"Column '{col}' cannot be converted to numeric: {exc}"
                    ) from exc

        # 4. Null check
        null_counts = df[OHLCV_COLUMNS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise DataValidationError(
                f"Null values detected in columns: "
                f"{dict(cols_with_nulls)}"
            )

        # 5. Sort by timestamp
        if not df["timestamp"].is_monotonic_increasing:
            logger.debug("Sorting DataFrame by timestamp")
            df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseDataAdapter":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.disconnect()
