"""Timestamp parsing, timezone handling, and market-hours utilities."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from enum import Enum
from typing import Union
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Common timezone references
# ---------------------------------------------------------------------------
UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")
EST = ZoneInfo("US/Eastern")

# Indian equity market hours (NSE / BSE)
MARKET_OPEN_IST = time(9, 15)
MARKET_CLOSE_IST = time(15, 30)


class MarketSession(str, Enum):
    PRE_OPEN = "PRE_OPEN"
    REGULAR = "REGULAR"
    POST_CLOSE = "POST_CLOSE"
    CLOSED = "CLOSED"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_timestamp(value: Union[str, int, float, datetime]) -> datetime:
    """Normalise various timestamp representations into a tz-aware ``datetime``.

    Accepts:
    * ISO-8601 strings (with or without timezone)
    * Unix epoch seconds (int / float)
    * ``datetime`` objects (made tz-aware if naive)

    Returns:
        A timezone-aware ``datetime`` in UTC.
    """
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        # Try ISO-8601 first (handles most broker / exchange formats)
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            # Fallback: common formats from Indian brokers
            for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%d-%m-%Y %H:%M:%S",
                "%Y%m%d%H%M%S",
                "%Y-%m-%dT%H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(value, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unable to parse timestamp: {value!r}")
        return _ensure_utc(dt)
    raise TypeError(f"Unsupported timestamp type: {type(value)}")


def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as a UTC-aware datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Timezone conversion
# ---------------------------------------------------------------------------

def to_ist(dt: datetime) -> datetime:
    """Convert a datetime to IST."""
    return parse_timestamp(dt).astimezone(IST)


def to_utc(dt: datetime) -> datetime:
    """Convert a datetime to UTC."""
    return parse_timestamp(dt).astimezone(UTC)


# ---------------------------------------------------------------------------
# Market-session helpers
# ---------------------------------------------------------------------------

def market_session(dt: datetime | None = None) -> MarketSession:
    """Determine the current NSE market session for a given timestamp.

    Args:
        dt: Timestamp to check (defaults to ``now`` in IST).

    Returns:
        The :class:`MarketSession` enum value.
    """
    ist_dt = to_ist(dt) if dt else datetime.now(IST)
    t = ist_dt.time()

    if ist_dt.weekday() >= 5:
        return MarketSession.CLOSED
    if t < time(9, 0):
        return MarketSession.CLOSED
    if t < MARKET_OPEN_IST:
        return MarketSession.PRE_OPEN
    if t <= MARKET_CLOSE_IST:
        return MarketSession.REGULAR
    return MarketSession.POST_CLOSE


def is_market_open(dt: datetime | None = None) -> bool:
    """Return ``True`` if the NSE regular session is active."""
    return market_session(dt) == MarketSession.REGULAR


def trading_day_range(
    d: date | None = None,
) -> tuple[datetime, datetime]:
    """Return (market_open, market_close) as UTC datetimes for a given date."""
    d = d or date.today()
    open_dt = datetime.combine(d, MARKET_OPEN_IST, tzinfo=IST).astimezone(UTC)
    close_dt = datetime.combine(d, MARKET_CLOSE_IST, tzinfo=IST).astimezone(UTC)
    return open_dt, close_dt


def epoch_ms(dt: datetime) -> int:
    """Convert a datetime to Unix epoch milliseconds."""
    return int(parse_timestamp(dt).timestamp() * 1000)


def from_epoch_ms(ms: int) -> datetime:
    """Convert Unix epoch milliseconds to a UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=UTC)


def floor_to_interval(dt: datetime, interval_minutes: int) -> datetime:
    """Floor a datetime to the nearest lower candle boundary.

    Example: ``floor_to_interval(09:23, 5)`` -> ``09:20``.
    """
    dt = parse_timestamp(dt)
    total_minutes = dt.hour * 60 + dt.minute
    floored = (total_minutes // interval_minutes) * interval_minutes
    return dt.replace(
        hour=floored // 60,
        minute=floored % 60,
        second=0,
        microsecond=0,
    )
