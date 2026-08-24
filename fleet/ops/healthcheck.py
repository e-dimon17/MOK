"""Periodic node health checks — the operator's watchdog.

Checks GPU vitals (ECC/temperature/power), NVLink topology, wall-clock sync
(the upload gate is a WALL-CLOCK protocol — a drifting clock silently turns
into late uploads), R2 reachability, and disk headroom. Every probe input is
injectable (canned text in tests, files via CLI flags for debugging); the CLI
loop emits one JSON report per interval on stdout — structured telemetry a
supervisor or the dashboard can tail.

All checks are report-only: the watchdog never kills the miner (the protocol
already prices failures via scoring); it exists so the operator sees trouble
before the chain does.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fleet.onboarding.preflight import parse_topo_matrix
from mok_core.config.schemas import DataConfig, FrozenModel

__all__ = [
    "GPU_QUERY_FIELDS",
    "MAX_CLOCK_OFFSET_S",
    "MAX_GPU_TEMP_C",
    "MIN_DISK_FREE_BYTES",
    "HealthCheck",
    "HealthReport",
    "build_parser",
    "clock_sync",
    "disk_space",
    "gpu_health",
    "main",
    "nvlink_health",
    "run_healthchecks",
    "storage_reachability",
]

GPU_QUERY_FIELDS = "index,name,temperature.gpu,power.draw,ecc.errors.uncorrected.volatile.total"
MAX_GPU_TEMP_C = 88.0
MAX_CLOCK_OFFSET_S = 1.0
MIN_DISK_FREE_BYTES = 500 * 10**9  # running headroom (preflight demands 3 TB up front)

Runner = Callable[[list[str]], str]


class HealthCheck(FrozenModel):
    name: str
    ok: bool
    detail: str


class HealthReport(FrozenModel):
    ts: float
    checks: tuple[HealthCheck, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_json(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "ok": self.ok,
            "checks": [c.model_dump() for c in self.checks],
        }


# --------------------------------------------------------------------------- #
# Individual checks (pure over injected text)
# --------------------------------------------------------------------------- #


def gpu_health(query_csv: str, *, max_temp_c: float = MAX_GPU_TEMP_C) -> HealthCheck:
    """Parse ``nvidia-smi --query-gpu=<GPU_QUERY_FIELDS> --format=csv,noheader,nounits``.

    Fails on any uncorrected volatile ECC error or a GPU above ``max_temp_c``.
    ``[N/A]`` ECC fields (ECC disabled/unsupported reporting) are noted but do
    not fail — the fleet spec runs ECC on, so preflight/ops policy catches it.
    """
    problems: list[str] = []
    notes: list[str] = []
    rows = 0
    for line in query_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            problems.append(f"unparseable row: {line.strip()!r}")
            continue
        rows += 1
        idx, _name, temp_s, power_s, ecc_s = parts[:5]
        try:
            temp = float(temp_s)
        except ValueError:
            problems.append(f"gpu{idx}: bad temperature {temp_s!r}")
            continue
        if temp > max_temp_c:
            problems.append(f"gpu{idx}: {temp:.0f}C > {max_temp_c:.0f}C")
        if ecc_s.upper() in ("[N/A]", "N/A"):
            notes.append(f"gpu{idx}: ECC counter N/A")
        else:
            try:
                if int(ecc_s) > 0:
                    problems.append(f"gpu{idx}: {ecc_s} uncorrected ECC errors")
            except ValueError:
                problems.append(f"gpu{idx}: bad ECC field {ecc_s!r}")
        try:
            float(power_s)
        except ValueError:
            problems.append(f"gpu{idx}: bad power field {power_s!r}")
    if rows == 0:
        problems.append("no GPU rows in query output")
    detail = "; ".join(problems) if problems else f"{rows} GPUs nominal"
    if notes:
        detail += " (" + "; ".join(notes) + ")"
    return HealthCheck(name="gpu", ok=not problems, detail=detail)


def nvlink_health(topo_text: str) -> HealthCheck:
    """All-pairs NVLink, via the preflight topology parser."""
    try:
        links = parse_topo_matrix(topo_text)
    except ValueError as e:
        return HealthCheck(name="nvlink", ok=False, detail=f"topology unparseable: {e}")
    bad = sorted({f"{i}-{j}:{v}" for (i, j), v in links.items() if not v.startswith("NV")})
    return HealthCheck(
        name="nvlink",
        ok=not bad,
        detail="all GPU pairs NVLink" if not bad else f"non-NVLink pairs: {bad[:8]}",
    )


def clock_sync(status_text: str, *, max_offset_s: float = MAX_CLOCK_OFFSET_S) -> HealthCheck:
    """Wall-clock sync from ``chronyc tracking`` or ``timedatectl`` output
    (auto-detected by content — parse whichever the host has)."""
    text = status_text.strip()
    if not text:
        return HealthCheck(name="clock", ok=False, detail="empty clock status output")
    if "System time" in text:  # chronyc tracking
        for line in text.splitlines():
            if line.strip().startswith("System time"):
                # "System time     : 0.000000012 seconds fast of NTP time"
                try:
                    offset = abs(float(line.split(":", 1)[1].split()[0]))
                except (IndexError, ValueError):
                    return HealthCheck(name="clock", ok=False, detail=f"bad chronyc line {line!r}")
                ok = offset <= max_offset_s
                return HealthCheck(
                    name="clock",
                    ok=ok,
                    detail=f"chrony offset {offset:.6f}s (max {max_offset_s}s)",
                )
        return HealthCheck(name="clock", ok=False, detail="chronyc output missing 'System time'")
    if "System clock synchronized" in text:  # timedatectl
        synced = any(
            line.strip().lower() == "system clock synchronized: yes" for line in text.splitlines()
        )
        return HealthCheck(
            name="clock",
            ok=synced,
            detail="timedatectl: synchronized" if synced else "timedatectl: NOT synchronized",
        )
    return HealthCheck(name="clock", ok=False, detail="unrecognized clock status format")


async def storage_reachability(storage: Any, bucket: Any, key: str = "manifest.json") -> HealthCheck:
    """Async HEAD through the storage client. A missing object is still a
    REACHABLE endpoint — only transport/auth failures fail this check."""
    try:
        exists = await storage.object_exists(bucket, key)
    except Exception as e:  # noqa: BLE001 — any transport failure is the finding
        return HealthCheck(
            name="storage", ok=False, detail=f"HEAD {key} failed: {type(e).__name__}: {e}"
        )
    return HealthCheck(
        name="storage", ok=True, detail=f"HEAD {key}: {'present' if exists else 'absent'} (reachable)"
    )


def disk_space(
    path: str | Path,
    *,
    min_free_bytes: int = MIN_DISK_FREE_BYTES,
    disk_usage: Callable[[str], Any] | None = None,
) -> HealthCheck:
    usage = disk_usage if disk_usage is not None else shutil.disk_usage
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = usage(str(probe)).free
    except OSError as e:
        return HealthCheck(name="disk", ok=False, detail=f"disk_usage({probe}): {e}")
    return HealthCheck(
        name="disk",
        ok=free >= min_free_bytes,
        detail=f"{free / 1e9:.0f} GB free at {probe} (min {min_free_bytes / 1e9:.0f} GB)",
    )


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #


def _default_runner(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout


def _probe(runner: Runner, cmds: list[list[str]]) -> str | None:
    for cmd in cmds:
        try:
            return runner(cmd)
        except Exception:  # noqa: BLE001 — try the next probe
            continue
    return None


async def run_healthchecks(
    *,
    gpu_csv: str | None = None,
    topo_text: str | None = None,
    clock_text: str | None = None,
    storage: Any | None = None,
    bucket: Any | None = None,
    cache_dir: str | Path | None = None,
    min_free_bytes: int = MIN_DISK_FREE_BYTES,
    runner: Runner | None = None,
    disk_usage: Callable[[str], Any] | None = None,
    clock: Callable[[], float] = time.time,
) -> HealthReport:
    """Assemble one report. ``None`` inputs are probed live via ``runner``;
    a failed probe becomes a failed check (the report always renders)."""
    run = runner if runner is not None else _default_runner
    cache = Path(cache_dir) if cache_dir is not None else Path(DataConfig().shard_cache_dir).expanduser()
    checks: list[HealthCheck] = []

    if gpu_csv is None:
        gpu_csv = _probe(
            run, [["nvidia-smi", f"--query-gpu={GPU_QUERY_FIELDS}", "--format=csv,noheader,nounits"]]
        )
    checks.append(
        gpu_health(gpu_csv)
        if gpu_csv is not None
        else HealthCheck(name="gpu", ok=False, detail="nvidia-smi query failed")
    )

    if topo_text is None:
        topo_text = _probe(run, [["nvidia-smi", "topo", "-m"]])
    checks.append(
        nvlink_health(topo_text)
        if topo_text is not None
        else HealthCheck(name="nvlink", ok=False, detail="nvidia-smi topo -m failed")
    )

    if clock_text is None:
        clock_text = _probe(run, [["chronyc", "tracking"], ["timedatectl", "status"], ["timedatectl"]])
    checks.append(
        clock_sync(clock_text)
        if clock_text is not None
        else HealthCheck(name="clock", ok=False, detail="no chronyc/timedatectl available")
    )

    if storage is not None and bucket is not None:
        checks.append(await storage_reachability(storage, bucket))

    checks.append(disk_space(cache, min_free_bytes=min_free_bytes, disk_usage=disk_usage))
    return HealthReport(ts=clock(), checks=tuple(checks))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m fleet.ops.healthcheck",
        description="Periodic node health watchdog; one JSON report per interval on stdout.",
    )
    p.add_argument("--interval", type=float, default=300.0, help="seconds between reports")
    p.add_argument("--iterations", type=int, default=0, help="0 = run forever")
    p.add_argument("--cache-dir", default=None, help="shard-cache path for the disk check")
    p.add_argument(
        "--min-free-gb", type=float, default=MIN_DISK_FREE_BYTES / 1e9, help="disk free threshold"
    )
    p.add_argument("--gpu-csv", default=None, help="canned nvidia-smi query output file (debug/test)")
    p.add_argument("--topo", default=None, help="canned nvidia-smi topo -m output file")
    p.add_argument("--clock-status", default=None, help="canned chronyc/timedatectl output file")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def read(value: str | None) -> str | None:
        return Path(value).read_text(encoding="utf-8") if value is not None else None

    iteration = 0
    while True:
        report = asyncio.run(
            run_healthchecks(
                gpu_csv=read(args.gpu_csv),
                topo_text=read(args.topo),
                clock_text=read(args.clock_status),
                cache_dir=args.cache_dir,
                min_free_bytes=int(args.min_free_gb * 1e9),
            )
        )
        print(json.dumps(report.to_json(), sort_keys=True), flush=True)
        iteration += 1
        if args.iterations and iteration >= args.iterations:
            return 0 if report.ok else 1
        time.sleep(args.interval)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
