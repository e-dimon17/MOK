"""Hardware preflight — refuse to onboard a node that cannot possibly keep up.

Checks the full Tier-A spec (playbook step B): 8× B300 (SM103, compute
capability 10.3), ≥280 GB HBM per GPU, all-pairs NVLink, ≥1.2 TB system RAM,
≥3 TB free NVMe under the shard cache, and the blessed-container digest env.
Every probe output is injectable so the parsing and the verdicts are fully
CPU-testable with canned fixtures; on a real node the defaults shell out to
``nvidia-smi`` / ``/proc/meminfo`` / ``shutil.disk_usage``.

Compute capability is read from BOTH supported sources ("parse both paths"):
``nvidia-smi --query-gpu=compute_cap`` (preferred, present on current
drivers) and a ``<compute_cap>`` element in ``nvidia-smi -q -x`` output when
the driver includes one — whichever yields values wins; neither available is
a failed check (attestation would catch it anyway, but failing fast is the
kind option).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mok_core.config.schemas import DataConfig, FrozenModel
from mok_core.determinism import CONTAINER_DIGEST_ENV

__all__ = [
    "GPU_NAME_TOKEN",
    "MIN_NVME_FREE_BYTES",
    "MIN_RAM_BYTES",
    "MIN_VRAM_BYTES",
    "REQUIRED_COMPUTE_CAP",
    "REQUIRED_GPUS",
    "GpuInfo",
    "PreflightCheck",
    "PreflightError",
    "PreflightReport",
    "parse_compute_caps_csv",
    "parse_meminfo_bytes",
    "parse_smi_xml",
    "parse_topo_matrix",
    "run_preflight",
]

REQUIRED_GPUS = 8
GPU_NAME_TOKEN = "B300"
REQUIRED_COMPUTE_CAP = "10.3"  # SM103 — the only architecture the MoK kernel builds for
MIN_VRAM_BYTES = 280 * 10**9  # per GPU
MIN_RAM_BYTES = int(1.2 * 10**12)
MIN_NVME_FREE_BYTES = 3 * 10**12

Runner = Callable[[list[str]], str]

_MEM_UNITS = {"KiB": 1024, "MiB": 1024**2, "GiB": 1024**3, "TiB": 1024**4}


class PreflightError(RuntimeError):
    pass


class PreflightCheck(FrozenModel):
    name: str
    ok: bool
    detail: str


class PreflightReport(FrozenModel):
    checks: tuple[PreflightCheck, ...]

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(c for c in self.checks if not c.ok)

    def strict(self) -> None:
        """Raise ``PreflightError`` listing every failed check (no-op if all pass)."""
        failed = self.failures()
        if failed:
            raise PreflightError(
                "preflight failed: " + "; ".join(f"[{c.name}] {c.detail}" for c in failed)
            )


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_bytes: int
    compute_cap: str | None  # from the XML, when the driver includes it


# --------------------------------------------------------------------------- #
# Parsers (pure)
# --------------------------------------------------------------------------- #


def _parse_mem_field(text: str) -> int:
    """'289536 MiB' -> bytes. Raises ValueError on garbage."""
    parts = text.split()
    if len(parts) != 2 or parts[1] not in _MEM_UNITS:
        raise ValueError(f"unparseable memory field {text!r}")
    return int(float(parts[0]) * _MEM_UNITS[parts[1]])


def parse_smi_xml(xml_text: str) -> list[GpuInfo]:
    """GPU name / VRAM / (optional) compute cap from ``nvidia-smi -q -x`` output."""
    root = ET.fromstring(xml_text)
    gpus: list[GpuInfo] = []
    for gpu in root.iter("gpu"):
        name = (gpu.findtext("product_name") or "").strip()
        total = gpu.findtext("fb_memory_usage/total") or ""
        cap = gpu.findtext("compute_cap")
        gpus.append(
            GpuInfo(
                name=name,
                vram_bytes=_parse_mem_field(total.strip()) if total.strip() else 0,
                compute_cap=cap.strip() if cap and cap.strip() else None,
            )
        )
    return gpus


def parse_compute_caps_csv(csv_text: str) -> list[str]:
    """Per-GPU capability strings from ``nvidia-smi --query-gpu=compute_cap
    --format=csv,noheader`` output (one value per line)."""
    return [line.strip() for line in csv_text.splitlines() if line.strip()]


def parse_topo_matrix(topo_text: str) -> dict[tuple[int, int], str]:
    """GPU-pair link types from ``nvidia-smi topo -m`` output.

    Returns ``{(i, j): link}`` for every ordered off-diagonal GPU pair. Non-GPU
    columns (NICs, CPU/NUMA affinity) are ignored.
    """
    rows: dict[int, list[str]] = {}
    header_gpus = 0
    for line in topo_text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0].startswith("GPU") and tokens[0][3:].isdigit():
            rows[int(tokens[0][3:])] = tokens[1:]
        elif header_gpus == 0:
            header_gpus = sum(1 for t in tokens if t.startswith("GPU") and t[3:].isdigit())
    n = len(rows)
    if n == 0:
        raise ValueError("no GPU rows found in topology matrix")
    links: dict[tuple[int, int], str] = {}
    for i, cells in rows.items():
        if len(cells) < n:
            raise ValueError(f"GPU{i} topology row has {len(cells)} cells, expected >= {n}")
        for j in range(n):
            if i != j:
                links[(i, j)] = cells[j]
    return links


def parse_meminfo_bytes(meminfo_text: str) -> int:
    """MemTotal bytes from ``/proc/meminfo`` content."""
    for line in meminfo_text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "kB":
                return int(parts[1]) * 1024
            raise ValueError(f"unparseable MemTotal line {line!r}")
    raise ValueError("MemTotal not found in meminfo")


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #


def _check_gpu_count(gpus: list[GpuInfo]) -> PreflightCheck:
    return PreflightCheck(
        name="gpu_count",
        ok=len(gpus) == REQUIRED_GPUS,
        detail=f"{len(gpus)} GPUs visible, need exactly {REQUIRED_GPUS}",
    )


def _check_gpu_names(gpus: list[GpuInfo]) -> PreflightCheck:
    bad = [g.name for g in gpus if GPU_NAME_TOKEN not in g.name]
    return PreflightCheck(
        name="gpu_name",
        ok=not bad,
        detail=(
            f"all GPUs report {GPU_NAME_TOKEN!r}"
            if not bad
            else f"non-{GPU_NAME_TOKEN} GPUs present: {sorted(set(bad))}"
        ),
    )


def _check_vram(gpus: list[GpuInfo]) -> PreflightCheck:
    low = [g.vram_bytes for g in gpus if g.vram_bytes < MIN_VRAM_BYTES]
    return PreflightCheck(
        name="gpu_vram",
        ok=not low,
        detail=(
            f"all GPUs have >= {MIN_VRAM_BYTES / 1e9:.0f} GB HBM"
            if not low
            else f"{len(low)} GPUs below {MIN_VRAM_BYTES / 1e9:.0f} GB: {sorted(low)}"
        ),
    )


def _check_compute_cap(query_caps: list[str] | None, gpus: list[GpuInfo]) -> PreflightCheck:
    caps = query_caps if query_caps else [g.compute_cap for g in gpus if g.compute_cap]
    if not caps:
        return PreflightCheck(
            name="compute_cap",
            ok=False,
            detail="compute capability unavailable from both --query-gpu and -q -x output",
        )
    bad = [c for c in caps if c != REQUIRED_COMPUTE_CAP]
    return PreflightCheck(
        name="compute_cap",
        ok=not bad and len(caps) >= len(gpus),
        detail=(
            f"all GPUs report compute capability {REQUIRED_COMPUTE_CAP} (SM103)"
            if not bad and len(caps) >= len(gpus)
            else f"caps {caps} != {REQUIRED_COMPUTE_CAP} for all {len(gpus)} GPUs"
        ),
    )


def _check_nvlink(topo_text: str) -> PreflightCheck:
    try:
        links = parse_topo_matrix(topo_text)
    except ValueError as e:
        return PreflightCheck(name="nvlink", ok=False, detail=f"topology unparseable: {e}")
    non_nv = sorted({f"{i}-{j}:{v}" for (i, j), v in links.items() if not v.startswith("NV")})
    return PreflightCheck(
        name="nvlink",
        ok=not non_nv,
        detail="all GPU pairs NVLink" if not non_nv else f"non-NVLink pairs: {non_nv[:8]}",
    )


def _check_ram(meminfo_text: str) -> PreflightCheck:
    try:
        total = parse_meminfo_bytes(meminfo_text)
    except ValueError as e:
        return PreflightCheck(name="ram", ok=False, detail=str(e))
    return PreflightCheck(
        name="ram",
        ok=total >= MIN_RAM_BYTES,
        detail=f"{total / 1e12:.2f} TB RAM (need >= {MIN_RAM_BYTES / 1e12:.1f} TB)",
    )


def _check_nvme(cache_dir: Path, disk_usage: Callable[[str], Any]) -> PreflightCheck:
    probe = cache_dir
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = disk_usage(str(probe)).free
    except OSError as e:
        return PreflightCheck(name="nvme_free", ok=False, detail=f"disk_usage({probe}): {e}")
    return PreflightCheck(
        name="nvme_free",
        ok=free >= MIN_NVME_FREE_BYTES,
        detail=(
            f"{free / 1e12:.2f} TB free at {probe} "
            f"(need >= {MIN_NVME_FREE_BYTES / 1e12:.1f} TB for the shard cache)"
        ),
    )


def _check_container_digest(env: Mapping[str, str]) -> PreflightCheck:
    digest = env.get(CONTAINER_DIGEST_ENV, "")
    return PreflightCheck(
        name="container_digest",
        ok=bool(digest),
        detail=(
            f"{CONTAINER_DIGEST_ENV}={digest}"
            if digest
            else f"{CONTAINER_DIGEST_ENV} unset — run inside the blessed container"
        ),
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _default_runner(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout


def run_preflight(
    *,
    smi_xml: str | None = None,
    compute_caps_csv: str | None = None,
    topo_text: str | None = None,
    meminfo_text: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    runner: Runner | None = None,
    disk_usage: Callable[[str], Any] | None = None,
) -> PreflightReport:
    """Run every check; ``None`` inputs are probed live via ``runner``.

    A probe failure (nvidia-smi missing, etc.) becomes a failed check rather
    than an exception — the report always covers all checks.
    """
    run = runner if runner is not None else _default_runner
    env = env if env is not None else os.environ
    usage = disk_usage if disk_usage is not None else shutil.disk_usage
    cache = Path(cache_dir) if cache_dir is not None else Path(DataConfig().shard_cache_dir).expanduser()

    checks: list[PreflightCheck] = []

    if smi_xml is None:
        try:
            smi_xml = run(["nvidia-smi", "-q", "-x"])
        except Exception as e:  # noqa: BLE001 — any probe failure is a finding
            smi_xml = ""
            checks.append(PreflightCheck(name="gpu_probe", ok=False, detail=f"nvidia-smi -q -x: {e}"))
    gpus: list[GpuInfo] = []
    if smi_xml:
        try:
            gpus = parse_smi_xml(smi_xml)
        except (ET.ParseError, ValueError) as e:
            checks.append(PreflightCheck(name="gpu_probe", ok=False, detail=f"XML unparseable: {e}"))
    checks.append(_check_gpu_count(gpus))
    checks.append(_check_gpu_names(gpus))
    checks.append(_check_vram(gpus))

    if compute_caps_csv is None:
        try:
            compute_caps_csv = run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
        except Exception:  # noqa: BLE001 — older drivers lack the field; XML path may cover it
            compute_caps_csv = None
    caps = parse_compute_caps_csv(compute_caps_csv) if compute_caps_csv else None
    checks.append(_check_compute_cap(caps, gpus))

    if topo_text is None:
        try:
            topo_text = run(["nvidia-smi", "topo", "-m"])
        except Exception as e:  # noqa: BLE001
            topo_text = ""
            checks.append(PreflightCheck(name="nvlink", ok=False, detail=f"nvidia-smi topo -m: {e}"))
    if topo_text:
        checks.append(_check_nvlink(topo_text))

    if meminfo_text is None:
        try:
            meminfo_text = Path("/proc/meminfo").read_text(encoding="utf-8")
        except OSError as e:
            meminfo_text = ""
            checks.append(PreflightCheck(name="ram", ok=False, detail=f"/proc/meminfo: {e}"))
    if meminfo_text:
        checks.append(_check_ram(meminfo_text))

    checks.append(_check_nvme(cache, usage))
    checks.append(_check_container_digest(env))
    return PreflightReport(checks=tuple(checks))
