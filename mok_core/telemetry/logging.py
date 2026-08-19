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


class ConsoleFormatter(logging.Formatter):
    """Operator-facing one-line format:
    ``HH:MM:SS LEVEL logger  message  key=value ...`` with bound context first.
    Values are compacted (hex digests shortened to 12 chars, floats to 4 sig.)."""

    _LEVEL_PAD = 7

    @staticmethod
    def _compact(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4g}"
        if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v):
            return v[:12] + "…"
        s = str(v)
        return s if len(s) <= 96 else s[:93] + "..."

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        ctx = _context.get() or {}
        extra = getattr(record, "fields", None)
        fields = {**ctx, **(extra if isinstance(extra, dict) else {})}
        kv = " ".join(f"{k}={self._compact(v)}" for k, v in fields.items())
        name = record.name.removeprefix("mok.")
        line = f"{ts} {record.levelname:<{self._LEVEL_PAD}} {name:<16} {record.getMessage()}"
        if kv:
            line += f"  {kv}"
        if record.exc_info and record.exc_info[0] is not None:
            line += "\n" + self.formatException(record.exc_info)
        return line


LOG_FORMAT_ENV = "MOK_LOG_FORMAT"      # "console" | "json"; default: console on a TTY, else json
LOG_LEVEL_ENV = "MOK_LOG_LEVEL"        # overrides the config level when set


def _pick_format(stream: Any) -> str:
    import os  # noqa: PLC0415

    fmt = os.environ.get(LOG_FORMAT_ENV, "").lower()
    if fmt in ("console", "json"):
        return fmt
    return "console" if getattr(stream, "isatty", lambda: False)() else "json"


def setup_logging(level: str = "INFO", *, stream=None, fmt: str | None = None) -> logging.Logger:
    """Configure the 'mok' logger once. Format: `fmt`, else $MOK_LOG_FORMAT, else
    console when stderr is a TTY (operators) and JSON otherwise (log shippers).
    $MOK_LOG_LEVEL overrides `level`."""
    import os  # noqa: PLC0415

    root = logging.getLogger("mok")
    root.setLevel(os.environ.get(LOG_LEVEL_ENV, level).upper())
    root.propagate = False
    if not root.handlers:
        out = stream or sys.stderr
        handler = logging.StreamHandler(out)
        chosen = fmt or _pick_format(out)
        handler.setFormatter(ConsoleFormatter() if chosen == "console" else JsonFormatter())
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
