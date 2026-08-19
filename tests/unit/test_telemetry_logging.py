"""mok_core/telemetry/logging.py — console vs JSON formats, env overrides, context binding."""

from __future__ import annotations

import io
import json
import logging

from mok_core.telemetry.logging import (
    LOG_FORMAT_ENV,
    LOG_LEVEL_ENV,
    ConsoleFormatter,
    JsonFormatter,
    bind,
    get_logger,
    setup_logging,
)


def _record(msg: str, **fields):
    rec = logging.LogRecord("mok.core.window_runner", logging.INFO, __file__, 1, msg, (), None)
    rec.fields = fields
    return rec


def test_console_format_is_one_line_with_compact_values() -> None:
    line = ConsoleFormatter().format(_record(
        "training done", window=3, final_loss=3.14159265, theta_end="ab" * 32, uids=[0, 2],
    ))
    assert "\n" not in line
    assert " INFO " in line and "core.window_runner" in line and "training done" in line
    assert "window=3" in line
    assert "final_loss=3.142" in line                 # 4 significant digits
    assert "theta_end=abababababab…" in line          # 64-hex digest shortened
    assert "uids=[0, 2]" in line
    assert "mok." not in line.split("training done")[0]   # logger prefix stripped


def test_console_format_includes_bound_context() -> None:
    bind(role="miner", uid=7)
    try:
        line = ConsoleFormatter().format(_record("window start", window=1))
        assert "role=miner" in line and "uid=7" in line and "window=1" in line
    finally:
        bind(role=None, uid=None)  # leave context neutral for other tests


def test_json_format_still_structured() -> None:
    entry = json.loads(JsonFormatter().format(_record("x", window=2)))
    assert entry["msg"] == "x" and entry["window"] == 2 and entry["level"] == "INFO"


def _isolated_setup(monkeypatch, level: str, stream) -> logging.Logger:
    """setup_logging against a pristine 'mok' logger (the suite shares the real one)."""
    root = logging.getLogger("mok")
    monkeypatch.setattr(root, "handlers", [])
    monkeypatch.setattr(root, "disabled", False)
    return setup_logging(level, stream=stream)


def _emit_via(root: logging.Logger, level: int, msg: str, **fields) -> None:
    rec = logging.LogRecord("mok.app.miner", level, __file__, 1, msg, (), None)
    rec.fields = fields
    for h in root.handlers:
        h.handle(rec)


def test_setup_logging_picks_console_via_env(monkeypatch) -> None:
    monkeypatch.setenv(LOG_FORMAT_ENV, "console")
    stream = io.StringIO()
    root = _isolated_setup(monkeypatch, "INFO", stream)
    assert isinstance(root.handlers[0].formatter, ConsoleFormatter)
    _emit_via(root, logging.INFO, "miner ready", start_window=0)
    out = stream.getvalue()
    assert "miner ready" in out and "start_window=0" in out and not out.lstrip().startswith("{")


def test_setup_logging_defaults_to_json_on_non_tty(monkeypatch) -> None:
    monkeypatch.delenv(LOG_FORMAT_ENV, raising=False)
    stream = io.StringIO()                       # no isatty -> json
    root = _isolated_setup(monkeypatch, "INFO", stream)
    assert isinstance(root.handlers[0].formatter, JsonFormatter)
    _emit_via(root, logging.INFO, "hello", k=1)
    assert json.loads(stream.getvalue().strip())["msg"] == "hello"


def test_log_level_env_overrides_config(monkeypatch) -> None:
    monkeypatch.setenv(LOG_LEVEL_ENV, "WARNING")
    root = _isolated_setup(monkeypatch, "INFO", io.StringIO())   # config INFO, env WARNING
    assert root.level == logging.WARNING


def test_survives_third_party_logger_reset(monkeypatch) -> None:
    """The bittensor SDK strips every non-primary logger's handlers and sets them
    to CRITICAL on import. Our loggers must keep emitting afterwards."""
    stream = io.StringIO()
    root = _isolated_setup(monkeypatch, "INFO", stream)
    lg = get_logger("app.miner")
    lg.info("before reset")
    # Simulate LoggingMachine.disable_third_party_loggers + before_enable_default:
    for name in ("mok", "mok.app.miner"):
        lgr = logging.getLogger(name)
        for h in list(lgr.handlers):
            lgr.removeHandler(h)
        lgr.setLevel(logging.CRITICAL)
    assert root.handlers == [] and root.level == logging.CRITICAL
    lg.info("after reset")                        # must re-arm and print
    out = stream.getvalue()
    assert "before reset" in out and "after reset" in out
    assert out.count("after reset") == 1          # single handler, no duplicates
