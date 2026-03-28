"""Local-first storage service for JSON, CSV, and JSONL persistence."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class StorageBackend(str, Enum):
    """Supported storage back-ends."""

    LOCAL = "LOCAL"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Storage(ABC):
    """Unified interface for persisting and loading data artifacts."""

    @abstractmethod
    def save_json(self, data: Any, path: str | Path) -> None:
        """Serialise *data* to a JSON file at *path*."""

    @abstractmethod
    def load_json(self, path: str | Path) -> dict[str, Any]:
        """Deserialise a JSON file and return the resulting dict."""

    @abstractmethod
    def save_csv(self, df: pd.DataFrame, path: str | Path) -> None:
        """Write a DataFrame to a CSV file."""

    @abstractmethod
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        """Read a CSV file into a DataFrame."""

    @abstractmethod
    def append_jsonl(self, data: Any, path: str | Path) -> None:
        """Append a single JSON object as a line to a JSONL file."""

    @abstractmethod
    def ensure_dir(self, path: str | Path) -> Path:
        """Create the directory (and parents) if it does not exist.

        Returns:
            The resolved :class:`Path` object.
        """


# ---------------------------------------------------------------------------
# Local filesystem implementation
# ---------------------------------------------------------------------------

class LocalStorage(Storage):
    """Reads and writes files on the local filesystem via :mod:`pathlib`."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base = Path(base_dir).resolve() if base_dir else None

    def _resolve(self, path: str | Path) -> Path:
        """Resolve *path* relative to *base_dir* (if configured)."""
        p = Path(path)
        if not p.is_absolute() and self._base is not None:
            p = self._base / p
        return p.resolve()

    # -- JSON ---------------------------------------------------------------

    def save_json(self, data: Any, path: str | Path) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.debug("JSON saved: %s", target)

    def load_json(self, path: str | Path) -> dict[str, Any]:
        target = self._resolve(path)
        with target.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        logger.debug("JSON loaded: %s", target)
        return data

    # -- CSV ----------------------------------------------------------------

    def save_csv(self, df: pd.DataFrame, path: str | Path) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target, index=False)
        logger.debug("CSV saved (%d rows): %s", len(df), target)

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        target = self._resolve(path)
        df = pd.read_csv(target)
        logger.debug("CSV loaded (%d rows): %s", len(df), target)
        return df

    # -- JSONL --------------------------------------------------------------

    def append_jsonl(self, data: Any, path: str | Path) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, default=str) + "\n")
        logger.debug("JSONL line appended: %s", target)

    # -- Directory ----------------------------------------------------------

    def ensure_dir(self, path: str | Path) -> Path:
        target = self._resolve(path)
        target.mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ensured: %s", target)
        return target


# ---------------------------------------------------------------------------
# Factory (mirrors notifier pattern for consistency)
# ---------------------------------------------------------------------------

def create_storage(backend: StorageBackend | str = StorageBackend.LOCAL, **kwargs: Any) -> Storage:
    """Instantiate a :class:`Storage` implementation.

    Args:
        backend: One of :class:`StorageBackend` values.
        **kwargs: Forwarded to the backend constructor (e.g. ``base_dir``).

    Returns:
        A ready-to-use :class:`Storage` instance.
    """
    if isinstance(backend, str):
        backend = StorageBackend(backend.upper())

    if backend is StorageBackend.LOCAL:
        return LocalStorage(**kwargs)

    raise ValueError(f"Unsupported storage backend: {backend}")
