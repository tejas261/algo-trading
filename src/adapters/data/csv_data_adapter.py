"""CSV-based data adapter for backtesting and research."""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd

from src.adapters.data.base_data_adapter import (
    BaseDataAdapter,
    DataValidationError,
    OHLCV_COLUMNS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Common timestamp formats tried during auto-detection, ordered from most
# specific to most general so the first successful parse wins.
_TIMESTAMP_FORMATS: list[str] = [
    "%Y-%m-%d %H:%M:%S%z",          # 2024-01-15 09:30:00+00:00
    "%Y-%m-%dT%H:%M:%S%z",          # ISO-8601 with tz
    "%Y-%m-%dT%H:%M:%S.%f%z",       # ISO-8601 with fractional seconds + tz
    "%Y-%m-%d %H:%M:%S.%f",         # with fractional seconds
    "%Y-%m-%d %H:%M:%S",            # standard datetime
    "%Y-%m-%dT%H:%M:%S",            # ISO-8601 no tz
    "%Y-%m-%dT%H:%M:%S.%f",        # ISO-8601 fractional no tz
    "%m/%d/%Y %H:%M:%S",            # US style
    "%d/%m/%Y %H:%M:%S",            # EU style
    "%Y-%m-%d",                      # date only
    "%m/%d/%Y",                      # US date only
    "%d/%m/%Y",                      # EU date only
]

# Commonly-seen column name variants mapped to canonical names.
_DEFAULT_COLUMN_ALIASES: dict[str, list[str]] = {
    "timestamp": ["timestamp", "date", "datetime", "time", "dt", "Date", "Datetime", "Timestamp"],
    "open":      ["open", "Open", "o", "OPEN"],
    "high":      ["high", "High", "h", "HIGH"],
    "low":       ["low", "Low", "l", "LOW"],
    "close":     ["close", "Close", "c", "CLOSE"],
    "volume":    ["volume", "Volume", "vol", "Vol", "v", "VOLUME"],
}

# Timeframe string pattern: one or more digits followed by a unit letter.
_TIMEFRAME_RE = re.compile(r"^(\d+)([mhd])$", re.IGNORECASE)

_TIMEFRAME_UNITS: dict[str, str] = {
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def _parse_timeframe(timeframe: str) -> timedelta:
    """Parse '1h', '4h', '1d', '15m', etc. to timedelta."""
    match = _TIMEFRAME_RE.match(timeframe)
    if not match:
        raise ValueError(
            f"Cannot parse timeframe string '{timeframe}'. "
            f"Expected format like '1m', '5m', '15m', '1h', '4h', '1d'."
        )
    value = int(match.group(1))
    unit_char = match.group(2).lower()
    unit_name = _TIMEFRAME_UNITS[unit_char]
    return timedelta(**{unit_name: value})


def _is_equity_market_gap(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Return True if the gap between *start* and *end* falls within expected
    equity market closure windows (overnight, weekends, holidays).

    Heuristic rules (US markets, broadly applicable):
    - Weekend gaps: start is Friday and end is Monday (or later due to holiday).
    - Overnight gaps: start and end are on consecutive trading days and the gap
      is less than ~18 hours (market closes ~16:00, reopens ~09:30 next day).
    """
    start_day = start.weekday()  # 0=Mon ... 6=Sun
    end_day = end.weekday()

    gap_hours = (end - start).total_seconds() / 3600.0

    # Weekend: Friday -> Monday (or Tuesday if Monday holiday)
    if start_day == 4 and end_day in (0, 1):  # Fri -> Mon/Tue
        return True

    # Overnight gap on consecutive weekdays (gap < 20h is typical)
    if gap_hours <= 20 and start_day != end_day:
        return True

    # Multi-day holiday gap (e.g. Thu close -> Mon open): up to ~4 calendar
    # days gap that starts/ends on a weekday
    if gap_hours <= 96 and end_day < 5 and start_day < 5:
        return True

    return False


class CsvDataAdapter(BaseDataAdapter):
    """Read OHLCV data from local CSV files.

    Parameters
    ----------
    data_dir:
        Root directory that contains CSV files.  Files are resolved as
        ``{data_dir}/{symbol}_{timeframe}.csv`` unless *file_map* is given.
    column_mapping:
        Optional explicit mapping ``{csv_column_name: canonical_name}``.
        When provided, auto-detection of column names is skipped.
    file_map:
        Optional mapping ``{(symbol, timeframe): "/absolute/path.csv"}``.
        Overrides the default filename convention.
    csv_read_kwargs:
        Extra keyword arguments forwarded to ``pd.read_csv``.
    gap_detection:
        How to handle detected data gaps:
        - ``"ignore"``: do nothing (skip gap detection entirely).
        - ``"warn"``: log a warning for every gap found, but continue.
        - ``"fail"``: raise ``DataValidationError`` if any gaps are detected.
        Defaults to ``"warn"``.
    market_type:
        ``"crypto"`` (24/7 markets, any gap is suspicious) or ``"equity"``
        (gaps during overnight/weekend market closures are expected and
        ignored).  Defaults to ``"crypto"``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        column_mapping: Optional[dict[str, str]] = None,
        file_map: Optional[dict[tuple[str, str], str | Path]] = None,
        csv_read_kwargs: Optional[dict[str, Any]] = None,
        gap_detection: Literal["ignore", "warn", "fail"] = "warn",
        market_type: Literal["crypto", "equity"] = "crypto",
    ) -> None:
        self._data_dir = Path(data_dir)
        self._column_mapping = column_mapping
        self._file_map = file_map or {}
        self._csv_read_kwargs = csv_read_kwargs or {}
        self._connected = False
        self._gap_detection = gap_detection
        self._market_type = market_type
        self._last_gaps: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gap_report(self) -> list[dict[str, Any]]:
        """Return the list of gaps detected during the last ``fetch_ohlcv`` call.

        Each entry is a dict with keys:
        ``start``, ``end``, ``missing_candles``, ``gap_hours``.
        """
        return list(self._last_gaps)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if not self._data_dir.is_dir():
            raise FileNotFoundError(
                f"Data directory does not exist: {self._data_dir}"
            )
        self._connected = True
        logger.info("CsvDataAdapter connected (data_dir=%s)", self._data_dir)

    def disconnect(self) -> None:
        self._connected = False
        logger.info("CsvDataAdapter disconnected")

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if not self._connected:
            raise RuntimeError("Adapter is not connected. Call connect() first.")

        csv_path = self._resolve_path(symbol, timeframe)
        logger.info("Reading %s", csv_path)

        df = pd.read_csv(csv_path, **self._csv_read_kwargs)
        df = self._normalize_columns(df)
        df = self._parse_timestamps(df)
        df = self._filter_by_date(df, start, end)
        df = self.validate_ohlcv(df)

        # --- Gap detection ---
        if self._gap_detection != "ignore" and not df.empty:
            gaps = self.detect_gaps(df, timeframe, self._gap_detection)
            self._last_gaps = gaps
        else:
            self._last_gaps = []

        return df

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def detect_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str,
        mode: Literal["ignore", "warn", "fail"] = "warn",
    ) -> list[dict[str, Any]]:
        """Detect gaps in OHLCV data based on expected candle intervals.

        Parameters
        ----------
        df:
            DataFrame with a ``timestamp`` column (datetime64, sorted).
        timeframe:
            Candle interval string, e.g. ``"1m"``, ``"5m"``, ``"1h"``, ``"1d"``.
        mode:
            ``"ignore"`` returns an empty list immediately.
            ``"warn"`` logs each gap as a warning.
            ``"fail"`` raises ``DataValidationError`` if any gaps exist.

        Returns
        -------
        list[dict]
            Each dict contains ``start``, ``end``, ``missing_candles``, and
            ``gap_hours``.
        """
        if mode == "ignore" or len(df) < 2:
            return []

        expected_td = _parse_timeframe(timeframe)
        threshold = expected_td * 1.5

        timestamps = df["timestamp"]
        diffs = timestamps.diff().iloc[1:]  # first diff is NaT

        gaps: list[dict[str, Any]] = []
        for idx in diffs.index:
            actual_delta = diffs[idx]
            if actual_delta > threshold:
                gap_start = timestamps[idx - 1]
                gap_end = timestamps[idx]

                # For equity markets, skip gaps that fall during expected
                # market closure windows.
                if self._market_type == "equity" and _is_equity_market_gap(
                    gap_start, gap_end
                ):
                    continue

                gap_seconds = actual_delta.total_seconds()
                expected_seconds = expected_td.total_seconds()
                missing_candles = int(gap_seconds / expected_seconds) - 1
                gap_hours = gap_seconds / 3600.0

                gap_info: dict[str, Any] = {
                    "start": gap_start,
                    "end": gap_end,
                    "missing_candles": missing_candles,
                    "gap_hours": round(gap_hours, 4),
                }
                gaps.append(gap_info)

        # Act on gaps according to mode
        if gaps and mode == "warn":
            for g in gaps:
                logger.warning(
                    "Data gap detected: %s -> %s (%.2f hours, ~%d missing candles)",
                    g["start"],
                    g["end"],
                    g["gap_hours"],
                    g["missing_candles"],
                )
        elif gaps and mode == "fail":
            summary_lines = [
                f"  {g['start']} -> {g['end']} "
                f"({g['gap_hours']:.2f}h, ~{g['missing_candles']} missing)"
                for g in gaps
            ]
            raise DataValidationError(
                f"Detected {len(gaps)} data gap(s):\n"
                + "\n".join(summary_lines)
            )

        return gaps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, symbol: str, timeframe: str) -> Path:
        """Resolve the CSV file path for a given symbol and timeframe."""
        key = (symbol, timeframe)
        if key in self._file_map:
            path = Path(self._file_map[key])
        else:
            # Sanitise symbol for filesystem (e.g. BTC/USDT -> BTC_USDT)
            safe_symbol = symbol.replace("/", "_").replace("\\", "_")
            filename = f"{safe_symbol}_{timeframe}.csv"
            path = self._data_dir / filename

        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename CSV columns to canonical OHLCV names.

        If an explicit ``column_mapping`` was provided at init, use that.
        Otherwise auto-detect by matching against known aliases.
        """
        if self._column_mapping:
            # Explicit mapping: csv_col -> canonical_col
            df = df.rename(columns=self._column_mapping)
        else:
            rename_map: dict[str, str] = {}
            existing_cols = set(df.columns)
            for canonical, aliases in _DEFAULT_COLUMN_ALIASES.items():
                if canonical in existing_cols:
                    # Already has the canonical name
                    continue
                for alias in aliases:
                    if alias in existing_cols and alias not in rename_map:
                        rename_map[alias] = canonical
                        break
            if rename_map:
                df = df.rename(columns=rename_map)

        # Verify all required columns are now present
        missing = set(OHLCV_COLUMNS) - set(df.columns)
        if missing:
            raise DataValidationError(
                f"After column normalisation the following columns are still "
                f"missing: {sorted(missing)}.  Available columns: "
                f"{sorted(df.columns.tolist())}.  Consider providing an "
                f"explicit column_mapping."
            )
        return df

    def _parse_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the *timestamp* column to ``datetime64[ns, UTC]``.

        Tries the caller-provided format first (via ``csv_read_kwargs``), then
        falls back to auto-detection against ``_TIMESTAMP_FORMATS``.
        """
        col = df["timestamp"]
        if pd.api.types.is_datetime64_any_dtype(col):
            # Already parsed (e.g. via parse_dates in csv_read_kwargs)
            if col.dt.tz is None:
                df["timestamp"] = col.dt.tz_localize("UTC")
            return df

        # If the column looks like epoch seconds / milliseconds
        if pd.api.types.is_numeric_dtype(col):
            sample = col.iloc[0]
            # Heuristic: epoch millis are > 1e12
            unit = "ms" if sample > 1e12 else "s"
            df["timestamp"] = pd.to_datetime(col, unit=unit, utc=True)
            logger.debug("Parsed timestamp as epoch (%s)", unit)
            return df

        # String-based: try each format
        sample_str = str(col.iloc[0]).strip()
        for fmt in _TIMESTAMP_FORMATS:
            try:
                datetime.strptime(sample_str, fmt)
                # If the sample parses, apply to the entire column.
                df["timestamp"] = pd.to_datetime(col, format=fmt, utc=True)
                logger.debug("Parsed timestamp with format '%s'", fmt)
                return df
            except (ValueError, TypeError):
                continue

        # Last resort: let pandas infer
        try:
            df["timestamp"] = pd.to_datetime(col, utc=True)
            logger.debug("Parsed timestamp via pandas inference")
            return df
        except Exception as exc:
            raise DataValidationError(
                f"Unable to parse timestamp column. Sample value: "
                f"'{sample_str}'. Error: {exc}"
            ) from exc

    @staticmethod
    def _filter_by_date(
        df: pd.DataFrame,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> pd.DataFrame:
        """Filter rows to the ``[start, end]`` inclusive date range."""
        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            df = df.loc[df["timestamp"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC")
            df = df.loc[df["timestamp"] <= end_ts]
        return df.reset_index(drop=True)
