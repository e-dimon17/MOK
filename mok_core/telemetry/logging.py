"""Structured JSON logging. Strictly outside the deterministic path — nothing
here may influence training state."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

_context: ContextVar[dict[str, Any] | None] = ContextVar("mok_log_context", default=None)


def bind(**fields: Any) -> None:
    """Attach fields (uid, window, role, rank...) to all subsequent log lines
    in this task/thread context."""
    _context.set({**(_context.get() or {}), **fields})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            **(_context.get() or {}),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            entry.update(extra)
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", *, stream=None) -> logging.Logger:
    root = logging.getLogger("mok")
    root.setLevel(level.upper())
    root.propagate = False
    if not root.handlers:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    return root


def get_logger(name: str) -> FieldLogger:
    return FieldLogger(logging.getLogger(f"mok.{name}"))


class FieldLogger:
    """Thin wrapper allowing `log.info("msg", window=3, loss=2.1)`."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, fields: dict[str, Any], exc_info: bool = False) -> None:
        self._logger.log(level, msg, extra={"fields": fields}, exc_info=exc_info)

    def debug(self, msg: str, **fields: Any) -> None:
        self._log(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._log(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._log(logging.WARNING, msg, fields)

    def error(self, msg: str, exc_info: bool = False, **fields: Any) -> None:
        self._log(logging.ERROR, msg, fields, exc_info=exc_info)
