"""Data validation helpers for OHLCV bars, configuration dicts, and general checks."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence


class ValidationError(Exception):
    """Raised when data fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation failed with {len(errors)} error(s): {'; '.join(errors)}")


# ---------------------------------------------------------------------------
# OHLCV validation
# ---------------------------------------------------------------------------

def validate_ohlcv_bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    timestamp: datetime | None = None,
    *,
    strict: bool = True,
) -> list[str]:
    """Validate a single OHLCV bar.

    Returns a list of error strings (empty means valid).  When *strict* is
    ``True``, additional range checks are applied.
    """
    errors: list[str] = []

    for name, val in [("open", open_), ("high", high), ("low", low), ("close", close)]:
        if val is None or (isinstance(val, float) and (val != val)):  # NaN check
            errors.append(f"{name} is missing or NaN")
        elif val <= 0:
            errors.append(f"{name} must be positive, got {val}")

    if volume is not None and volume < 0:
        errors.append(f"volume must be non-negative, got {volume}")

    if not errors:
        if high < low:
            errors.append(f"high ({high}) < low ({low})")
        if high < open_ or high < close:
            errors.append(f"high ({high}) is not the highest value")
        if low > open_ or low > close:
            errors.append(f"low ({low}) is not the lowest value")

    if strict and not errors:
        spread_pct = ((high - low) / low) * 100 if low > 0 else 0
        if spread_pct > 50:
            errors.append(f"Suspicious bar: {spread_pct:.1f}% spread (high={high}, low={low})")

    return errors


def validate_ohlcv_series(
    bars: Sequence[dict[str, Any]],
    *,
    required_fields: tuple[str, ...] = ("open", "high", "low", "close", "volume", "timestamp"),
    check_monotonic_time: bool = True,
) -> list[str]:
    """Validate a sequence of OHLCV bar dicts.

    Checks for missing fields, OHLCV consistency, and optionally monotonic
    timestamps.
    """
    errors: list[str] = []

    if not bars:
        errors.append("Empty bar series")
        return errors

    prev_ts: datetime | None = None

    for i, bar in enumerate(bars):
        prefix = f"bar[{i}]"

        missing = [f for f in required_fields if f not in bar]
        if missing:
            errors.append(f"{prefix}: missing fields {missing}")
            continue

        bar_errors = validate_ohlcv_bar(
            bar["open"], bar["high"], bar["low"], bar["close"],
            bar.get("volume", 0),
            bar.get("timestamp"),
        )
        errors.extend(f"{prefix}: {e}" for e in bar_errors)

        if check_monotonic_time and "timestamp" in bar:
            ts = bar["timestamp"]
            if isinstance(ts, datetime) and prev_ts is not None and ts <= prev_ts:
                errors.append(f"{prefix}: timestamp {ts} is not after previous {prev_ts}")
            if isinstance(ts, datetime):
                prev_ts = ts

    return errors


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

_CONFIG_SCHEMA: dict[str, dict[str, Any]] = {
    "initial_capital": {"type": float, "min": 0, "required": True},
    "risk_per_trade_pct": {"type": float, "min": 0.01, "max": 100.0, "required": True},
    "max_open_positions": {"type": int, "min": 1, "required": False, "default": 5},
    "commission_pct": {"type": float, "min": 0, "required": False, "default": 0.0},
    "slippage_pct": {"type": float, "min": 0, "required": False, "default": 0.0},
}


def validate_config(config: dict[str, Any], schema: dict[str, dict[str, Any]] | None = None) -> list[str]:
    """Validate a backtest / strategy configuration dict against a schema.

    Uses the built-in ``_CONFIG_SCHEMA`` when *schema* is ``None``.
    """
    schema = schema or _CONFIG_SCHEMA
    errors: list[str] = []

    for key, rules in schema.items():
        if key not in config:
            if rules.get("required", False):
                errors.append(f"Missing required config key: '{key}'")
            continue

        value = config[key]

        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            # Allow int where float is expected
            if expected_type is float and isinstance(value, int):
                pass
            else:
                errors.append(f"'{key}' must be {expected_type.__name__}, got {type(value).__name__}")
                continue

        if "min" in rules and value < rules["min"]:
            errors.append(f"'{key}' must be >= {rules['min']}, got {value}")
        if "max" in rules and value > rules["max"]:
            errors.append(f"'{key}' must be <= {rules['max']}, got {value}")

    return errors


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def require_positive(value: float, name: str = "value") -> float:
    """Return *value* if positive, otherwise raise ``ValueError``."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def require_non_negative(value: float, name: str = "value") -> float:
    """Return *value* if non-negative, otherwise raise ``ValueError``."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def require_in_range(
    value: float, low: float, high: float, name: str = "value",
) -> float:
    """Return *value* if ``low <= value <= high``, otherwise raise ``ValueError``."""
    if not (low <= value <= high):
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")
    return value


def assert_valid(errors: list[str], context: str = "") -> None:
    """Raise :class:`ValidationError` if *errors* is non-empty."""
    if errors:
        raise ValidationError(
            [f"[{context}] {e}" if context else e for e in errors]
        )
