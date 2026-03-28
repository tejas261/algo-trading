"""Thread-safe audit logger that writes structured events to a JSONL file."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _serialize(obj: Any) -> Any:
    """Convert Pydantic models, datetimes, UUIDs, etc. to JSON-safe values."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    return obj


class AuditLogger:
    """Append-only audit log with monotonic sequence numbers.

    Every event is a JSON object with the shape::

        {
            "timestamp": "2026-03-28T14:00:00+00:00",
            "event_type": "order_submitted",
            "data": { ... },
            "sequence_number": 42
        }

    All public methods are thread-safe.

    Args:
        log_path: Filesystem path for the JSONL audit log.
        buffer_size: Number of events to buffer before auto-flushing.
            Set to ``1`` for immediate writes (safest but slowest).
    """

    def __init__(
        self,
        log_path: str | Path = "audit_log.jsonl",
        buffer_size: int = 1,
    ) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer_size = max(1, buffer_size)

        self._lock = threading.Lock()
        self._sequence: int = 0
        self._buffer: list[dict[str, Any]] = []

    # -- core ---------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        data: dict[str, Any],
        timestamp: datetime | None = None,
    ) -> int:
        """Record an audit event.

        Args:
            event_type: A short, dot-free identifier (e.g. ``"order_submitted"``).
            data: Arbitrary payload associated with the event.
            timestamp: Event time. Defaults to *now* (UTC).

        Returns:
            The monotonic sequence number assigned to this event.
        """
        ts = timestamp or datetime.now(tz=timezone.utc)
        safe_data = _serialize(data)

        with self._lock:
            self._sequence += 1
            seq = self._sequence
            event = {
                "timestamp": ts.isoformat(),
                "event_type": event_type,
                "data": safe_data,
                "sequence_number": seq,
            }
            self._buffer.append(event)
            if len(self._buffer) >= self._buffer_size:
                self._flush_unlocked()

        logger.debug("Audit event #%d: %s", seq, event_type)
        return seq

    # -- convenience helpers ------------------------------------------------

    def log_order_submitted(self, order: Any) -> int:
        """Log an order submission event."""
        return self.log_event("order_submitted", {"order": order})

    def log_order_filled(self, order: Any, fill: Any) -> int:
        """Log an order fill event."""
        return self.log_event("order_filled", {"order": order, "fill": fill})

    def log_order_cancelled(self, order: Any) -> int:
        """Log an order cancellation event."""
        return self.log_event("order_cancelled", {"order": order})

    def log_risk_check(self, decision: Any) -> int:
        """Log the result of a risk-manager check."""
        return self.log_event("risk_check", {"decision": decision})

    def log_signal(self, signal_type: str, bar_data: Any) -> int:
        """Log a strategy signal event."""
        return self.log_event("signal", {"signal_type": signal_type, "bar_data": bar_data})

    def log_state_transition(self, from_state: str, to_state: str) -> int:
        """Log a state-machine transition."""
        return self.log_event(
            "state_transition",
            {"from_state": from_state, "to_state": to_state},
        )

    # -- query --------------------------------------------------------------

    def get_events(
        self,
        event_type: str | None = None,
        start: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read events from the log file, with optional filtering.

        Args:
            event_type: If given, only return events matching this type.
            start: If given, only return events at or after this timestamp.

        Returns:
            A list of event dicts ordered by sequence number.
        """
        # Flush any buffered events first so the query is consistent.
        self.flush()

        events: list[dict[str, Any]] = []
        if not self._path.exists():
            return events

        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed audit log line: %s", line[:120])
                    continue

                if event_type is not None and event.get("event_type") != event_type:
                    continue

                if start is not None:
                    event_ts = datetime.fromisoformat(event["timestamp"])
                    if event_ts < start:
                        continue

                events.append(event)

        return events

    # -- flush / lifecycle --------------------------------------------------

    def flush(self) -> None:
        """Write any buffered events to disk immediately."""
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Flush without acquiring the lock (caller must already hold it)."""
        if not self._buffer:
            return
        with self._path.open("a", encoding="utf-8") as fh:
            for event in self._buffer:
                fh.write(json.dumps(event, default=str) + "\n")
        self._buffer.clear()

    def __del__(self) -> None:
        """Best-effort flush on garbage collection."""
        try:
            self.flush()
        except Exception:  # noqa: BLE001
            pass
