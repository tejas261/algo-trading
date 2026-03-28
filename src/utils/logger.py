"""Structured logging setup with module-level logger factory."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for machine-readable output."""

    def __init__(self, include_extras: bool = True) -> None:
        super().__init__()
        self._include_extras = include_extras

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        if self._include_extras:
            standard_attrs = logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
            for key, value in record.__dict__.items():
                if key not in standard_attrs and key not in log_entry:
                    try:
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",     # cyan
        "INFO": "\033[32m",      # green
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[35m",  # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"{color}{ts} [{record.levelname:<8}] "
            f"{record.name}: {record.getMessage()}{self.RESET}"
        )


def get_logger(
    name: str,
    *,
    level: int | str = logging.INFO,
    log_dir: str | Path | None = None,
    console: bool = True,
    structured_file: bool = True,
) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        level: Logging level (int or string like ``"DEBUG"``).
        log_dir: Directory for log files. ``None`` disables file logging.
        console: Whether to attach a console (stderr) handler.
        structured_file: If ``True`` and ``log_dir`` is set, write JSON-structured
            logs to ``<log_dir>/<name>.log``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(level)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(ConsoleFormatter())
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    if log_dir is not None and structured_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        safe_name = name.replace(".", "_")
        file_handler = logging.FileHandler(log_path / f"{safe_name}.log")
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
