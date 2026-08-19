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


# The configuration last applied by setup_logging(): (level, handler). Kept so
# the 'mok' logger can be RE-ARMED if a third party strips it (see _ensure_armed).
_ARMED: dict[str, Any] = {}


def _make_handler(stream: Any, fmt: str | None) -> logging.Handler:
    out = stream or sys.stderr
    handler = logging.StreamHandler(out)
    chosen = fmt or _pick_format(out)
    handler.setFormatter(ConsoleFormatter() if chosen == "console" else JsonFormatter())
    return handler


def _ensure_armed() -> None:
    """Restore the 'mok' logger's handler/level if something stripped them."""
    if not _ARMED:
        return
    root = logging.getLogger("mok")
    handler = _ARMED["handler"]
    if handler not in root.handlers:
        root.addHandler(handler)
    if root.level != _ARMED["level"]:
        root.setLevel(_ARMED["level"])
    if root.disabled:
        root.disabled = False


def setup_logging(level: str = "INFO", *, stream=None, fmt: str | None = None) -> logging.Logger:
    """Configure the 'mok' logger once. Format: `fmt`, else $MOK_LOG_FORMAT, else
    console when stderr is a TTY (operators) and JSON otherwise (log shippers).
    $MOK_LOG_LEVEL overrides `level`.

    Robust against the bittensor SDK: its LoggingMachine, on import, REMOVES every
    handler from every logger it does not own and sets them to CRITICAL (its
    ``enable_third_party_loggers=False`` default). We do not register with it
    (that attaches its own colored handler and duplicates every line); instead
    FieldLogger re-arms the 'mok' handler/level on every emit (see _ensure_armed)."""
    import os  # noqa: PLC0415

    root = logging.getLogger("mok")
    lvl = logging.getLevelName(os.environ.get(LOG_LEVEL_ENV, level).upper())
    root.setLevel(lvl)
    root.propagate = False
    if not root.handlers:
        root.addHandler(_make_handler(stream, fmt))
    _ARMED.update(level=root.level, handler=root.handlers[0])
    # Quiet known-noisy third-party loggers that bypass sensible defaults: the
    # substrate websocket's keepalive thread logs a full traceback (via the
    # last-resort stderr handler) every time an idle RPC connection is culled
    # by the endpoint's load balancer — routine, auto-reconnected, and already
    # surfaced meaningfully by ChainClient's read retries when it matters.
    for noisy in ("websockets", "websockets.client", "websockets.sync"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    return root


def get_logger(name: str) -> FieldLogger:
    return FieldLogger(logging.getLogger(f"mok.{name}"))


class FieldLogger:
    """Thin wrapper allowing `log.info("msg", window=3, loss=2.1)`."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, msg: str, fields: dict[str, Any], exc_info: bool = False) -> None:
        _ensure_armed()
        if self._logger.disabled:            # bittensor's reset also disables loggers
            self._logger.disabled = False
        if self._logger.level and self._logger.level > logging.getLogger("mok").level:
            self._logger.setLevel(logging.NOTSET)  # inherit from 'mok' again
        self._logger.log(level, msg, extra={"fields": fields}, exc_info=exc_info)

    def debug(self, msg: str, **fields: Any) -> None:
        self._log(logging.DEBUG, msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        self._log(logging.INFO, msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        self._log(logging.WARNING, msg, fields)

    def error(self, msg: str, exc_info: bool = False, **fields: Any) -> None:
        self._log(logging.ERROR, msg, fields, exc_info=exc_info)
