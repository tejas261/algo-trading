"""Notification service with pluggable channels (console, file, webhook)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


class NotificationChannel(str, Enum):
    """Supported notification delivery channels."""

    CONSOLE = "CONSOLE"
    FILE = "FILE"
    WEBHOOK = "WEBHOOK"


# ---------------------------------------------------------------------------
# Alert levels (kept as plain strings for flexibility, but these are the
# canonical values used throughout the framework).
# ---------------------------------------------------------------------------
ALERT_INFO = "INFO"
ALERT_WARNING = "WARNING"
ALERT_ERROR = "ERROR"
ALERT_CRITICAL = "CRITICAL"


def _serialize(obj: Any) -> Any:
    """Best-effort JSON serialiser for Pydantic models, datetimes, UUIDs, etc."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__str__"):
        return str(obj)
    return obj


def _make_payload(
    *,
    alert_type: str,
    level: str,
    message: str,
    data: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    ts = timestamp or datetime.now(tz=timezone.utc)
    return {
        "timestamp": ts.isoformat(),
        "alert_type": alert_type,
        "level": level,
        "message": message,
        "data": data,
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Notifier(ABC):
    """Abstract notification sender."""

    @abstractmethod
    def _deliver(self, payload: dict[str, Any]) -> None:
        """Channel-specific delivery. Subclasses must implement."""

    # -- public API ---------------------------------------------------------

    def send_alert(
        self,
        level: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Send a generic alert."""
        payload = _make_payload(alert_type="alert", level=level, message=message, data=data)
        self._deliver(payload)

    def send_trade_notification(self, trade_intent: Any) -> None:
        """Notify about a new trade intent produced by the strategy."""
        data = _serialize(trade_intent)
        payload = _make_payload(
            alert_type="trade_intent",
            level=ALERT_INFO,
            message=f"Trade intent: {getattr(trade_intent, 'side', '?')} "
                    f"{getattr(trade_intent, 'symbol', '?')} "
                    f"@ {getattr(trade_intent, 'entry_price', '?')}",
            data=data if isinstance(data, dict) else {"raw": data},
        )
        self._deliver(payload)

    def send_risk_alert(self, risk_decision: Any) -> None:
        """Notify about a risk-manager decision (approve / reject / modify)."""
        data = _serialize(risk_decision)
        payload = _make_payload(
            alert_type="risk_decision",
            level=ALERT_WARNING,
            message=f"Risk decision: {risk_decision}",
            data=data if isinstance(data, dict) else {"raw": data},
        )
        self._deliver(payload)

    def send_daily_report(self, report: Any) -> None:
        """Send an end-of-day summary report."""
        data = _serialize(report)
        payload = _make_payload(
            alert_type="daily_report",
            level=ALERT_INFO,
            message="Daily report generated",
            data=data if isinstance(data, dict) else {"raw": data},
        )
        self._deliver(payload)

    def send_emergency_stop(self, reason: str) -> None:
        """Send a critical emergency-stop notification."""
        payload = _make_payload(
            alert_type="emergency_stop",
            level=ALERT_CRITICAL,
            message=f"EMERGENCY STOP: {reason}",
            data={"reason": reason},
        )
        self._deliver(payload)
        logger.critical("Emergency stop notification sent: %s", reason)


# ---------------------------------------------------------------------------
# Console implementation
# ---------------------------------------------------------------------------

_LEVEL_COLORS = {
    ALERT_INFO: "\033[32m",       # green
    ALERT_WARNING: "\033[33m",    # yellow
    ALERT_ERROR: "\033[31m",      # red
    ALERT_CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class ConsoleNotifier(Notifier):
    """Prints formatted, coloured alerts to stdout."""

    def _deliver(self, payload: dict[str, Any]) -> None:
        level = payload.get("level", ALERT_INFO)
        color = _LEVEL_COLORS.get(level, "")
        ts = payload["timestamp"]
        alert_type = payload["alert_type"]
        message = payload["message"]

        header = f"{color}[{ts}] [{level:<8}] [{alert_type}]{_RESET}"
        print(f"{header} {message}")

        data = payload.get("data")
        if data:
            print(f"  {json.dumps(data, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# File implementation (JSONL)
# ---------------------------------------------------------------------------

class FileNotifier(Notifier):
    """Appends each notification as a JSON line to a file."""

    def __init__(self, file_path: str | Path = "alerts.jsonl") -> None:
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _deliver(self, payload: dict[str, Any]) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        logger.debug("Alert written to %s", self._path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_notifier(channel: NotificationChannel | str, **kwargs: Any) -> Notifier:
    """Instantiate a :class:`Notifier` for the requested channel.

    Args:
        channel: One of :class:`NotificationChannel` values (or its string
            representation).
        **kwargs: Forwarded to the channel-specific constructor.

    Returns:
        A ready-to-use :class:`Notifier` instance.

    Raises:
        ValueError: If *channel* is not supported.
    """
    if isinstance(channel, str):
        channel = NotificationChannel(channel.upper())

    if channel is NotificationChannel.CONSOLE:
        return ConsoleNotifier(**kwargs)

    if channel is NotificationChannel.FILE:
        return FileNotifier(**kwargs)

    if channel is NotificationChannel.WEBHOOK:
        raise NotImplementedError(
            "Webhook notifier is not yet implemented. "
            "Provide a WebhookNotifier subclass or use FILE/CONSOLE."
        )

    raise ValueError(f"Unsupported notification channel: {channel}")
