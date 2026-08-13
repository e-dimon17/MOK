"""Metric emission: local JSONL always; wandb and the public dashboard bucket
optionally. All fire-and-forget — a telemetry failure must never stall a window."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

from ..config.schemas import TelemetryConfig
from .logging import get_logger

log = get_logger("telemetry")


class MetricSink(Protocol):
    def emit(self, kind: str, payload: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")  # noqa: SIM115 — long-lived sink handle

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        self._fh.write(json.dumps({"ts": time.time(), "kind": kind, **payload}, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class WandbSink:
    def __init__(self, cfg: TelemetryConfig, run_name: str) -> None:
        import wandb  # noqa: PLC0415

        self._run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            resume="allow",
            settings=wandb.Settings(start_method="thread"),
        )

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        numeric = {f"{kind}/{k}": v for k, v in payload.items() if isinstance(v, (int, float))}
        if numeric:
            self._run.log(numeric)

    def close(self) -> None:
        self._run.finish()


class Metrics:
    """Fan-out emitter. Every window emits: loss, grad_norm, router entropy,
    expert-load histogram, capacity utilization, window latency, audit results."""

    def __init__(self, cfg: TelemetryConfig, *, run_name: str, out_dir: str | Path = "telemetry") -> None:
        self._sinks: list[MetricSink] = [JsonlSink(Path(out_dir) / f"{run_name}.jsonl")]
        if cfg.wandb_project and os.environ.get("WANDB_API_KEY"):
            try:
                self._sinks.append(WandbSink(cfg, run_name))
            except Exception:  # noqa: BLE001 — telemetry must never be fatal
                log.warning("wandb init failed; continuing with local sink only")

    def emit(self, kind: str, **payload: Any) -> None:
        for sink in self._sinks:
            try:
                sink.emit(kind, payload)
            except Exception:  # noqa: BLE001
                log.warning("metric sink emit failed", kind=kind)

    def close(self) -> None:
        import contextlib  # noqa: PLC0415

        for sink in self._sinks:
            with contextlib.suppress(Exception):
                sink.close()
