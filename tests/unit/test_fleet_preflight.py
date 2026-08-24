"""Tests for fleet/onboarding/preflight.py — canned probes, pass + every failure mode."""

from __future__ import annotations

from typing import Any

import pytest

from fleet.onboarding.preflight import (
    MIN_NVME_FREE_BYTES,
    MIN_RAM_BYTES,
    PreflightError,
    PreflightReport,
    parse_compute_caps_csv,
    parse_meminfo_bytes,
    parse_smi_xml,
    parse_topo_matrix,
    run_preflight,
)

# --------------------------------------------------------------------------- #
# Canned fixtures
# --------------------------------------------------------------------------- #


def smi_xml(n_gpus: int = 8, *, name: str = "NVIDIA B300", vram: str = "289536 MiB",
            cap: str | None = None) -> str:
    cap_el = f"<compute_cap>{cap}</compute_cap>" if cap else ""
    gpus = "".join(
        f'<gpu id="00000000:0{i}:00.0"><product_name>{name}</product_name>{cap_el}'
        f"<fb_memory_usage><total>{vram}</total><used>0 MiB</used></fb_memory_usage></gpu>"
        for i in range(n_gpus)
    )
    return f'<?xml version="1.0"?><nvidia_smi_log><attached_gpus>{n_gpus}</attached_gpus>{gpus}</nvidia_smi_log>'


def topo_text(n: int = 8, *, link: str = "NV18", break_pair: tuple[int, int] | None = None) -> str:
    header = "\t" + "\t".join(f"GPU{i}" for i in range(n)) + "\tCPU Affinity\tNUMA Affinity"
    lines = [header]
    for i in range(n):
        cells = []
        for j in range(n):
            if i == j:
                cells.append("X")
            elif break_pair is not None and {i, j} == set(break_pair):
                cells.append("SYS")
            else:
                cells.append(link)
        lines.append(f"GPU{i}\t" + "\t".join(cells) + "\t0-95\t0")
    return "\n".join(lines)


MEMINFO_OK = "MemTotal:       1585000000 kB\nMemFree:        1000000000 kB\n"
MEMINFO_LOW = "MemTotal:        528000000 kB\n"
CAPS_OK = "10.3\n" * 8
ENV_OK = {"MOK_CONTAINER_DIGEST": "sha256:" + "ab" * 32}


class FakeUsage:
    def __init__(self, free: int) -> None:
        self.free = free
        self.total = free * 2
        self.used = free


def good_kwargs(tmp_path, **overrides: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "smi_xml": smi_xml(),
        "compute_caps_csv": CAPS_OK,
        "topo_text": topo_text(),
        "meminfo_text": MEMINFO_OK,
        "cache_dir": tmp_path,
        "env": ENV_OK,
        "disk_usage": lambda _p: FakeUsage(4 * 10**12),
    }
    kw.update(overrides)
    return kw


def by_name(report: PreflightReport) -> dict[str, bool]:
    return {c.name: c.ok for c in report.checks}


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #


def test_parse_smi_xml() -> None:
    gpus = parse_smi_xml(smi_xml(2, cap="10.3"))
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA B300"
    assert gpus[0].vram_bytes == 289536 * 1024**2
    assert gpus[0].compute_cap == "10.3"
    assert parse_smi_xml(smi_xml(1))[0].compute_cap is None


def test_parse_compute_caps_csv() -> None:
    assert parse_compute_caps_csv("10.3\n10.3\n\n") == ["10.3", "10.3"]


def test_parse_topo_matrix_ignores_non_gpu_columns() -> None:
    links = parse_topo_matrix(topo_text(4))
    assert len(links) == 4 * 3
    assert links[(0, 1)] == "NV18"
    with pytest.raises(ValueError, match="no GPU rows"):
        parse_topo_matrix("nothing here")


def test_parse_meminfo() -> None:
    assert parse_meminfo_bytes(MEMINFO_OK) == 1585000000 * 1024
    with pytest.raises(ValueError, match="MemTotal"):
        parse_meminfo_bytes("MemFree: 12 kB")


# --------------------------------------------------------------------------- #
# Pass + each failure mode
# --------------------------------------------------------------------------- #


def test_full_pass(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path))
    assert report.ok, [c for c in report.checks if not c.ok]
    report.strict()  # no raise
    assert set(by_name(report)) == {
        "gpu_count", "gpu_name", "gpu_vram", "compute_cap", "nvlink", "ram",
        "nvme_free", "container_digest",
    }


def test_fail_gpu_count(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, smi_xml=smi_xml(7), compute_caps_csv="10.3\n" * 7))
    assert by_name(report)["gpu_count"] is False
    with pytest.raises(PreflightError, match="gpu_count"):
        report.strict()


def test_fail_gpu_name(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, smi_xml=smi_xml(name="NVIDIA H100 SXM")))
    assert by_name(report)["gpu_name"] is False


def test_fail_vram(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, smi_xml=smi_xml(vram="141520 MiB")))
    assert by_name(report)["gpu_vram"] is False


def test_fail_compute_cap_wrong(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, compute_caps_csv="9.0\n" * 8))
    assert by_name(report)["compute_cap"] is False


def _no_smi(cmd):
    # Simulates a host without nvidia-smi — without this, compute_caps_csv=None
    # falls through to a LIVE probe, which succeeds on GPU-equipped test hosts.
    raise FileNotFoundError("nvidia-smi not present")


def test_compute_cap_falls_back_to_xml(tmp_path) -> None:
    report = run_preflight(
        **good_kwargs(tmp_path, smi_xml=smi_xml(cap="10.3"), compute_caps_csv=None, runner=_no_smi)
    )
    assert by_name(report)["compute_cap"] is True


def test_compute_cap_unavailable_from_both_paths_fails(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, compute_caps_csv=None, runner=_no_smi))
    check = {c.name: c for c in report.checks}["compute_cap"]
    assert not check.ok
    assert "unavailable" in check.detail


def test_fail_nvlink_pair(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, topo_text=topo_text(break_pair=(2, 5))))
    check = {c.name: c for c in report.checks}["nvlink"]
    assert not check.ok
    assert "2-5:SYS" in check.detail


def test_fail_ram(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, meminfo_text=MEMINFO_LOW))
    assert by_name(report)["ram"] is False
    assert int(1.2e12) == MIN_RAM_BYTES


def test_fail_nvme_free(tmp_path) -> None:
    report = run_preflight(
        **good_kwargs(tmp_path, disk_usage=lambda _p: FakeUsage(MIN_NVME_FREE_BYTES - 1))
    )
    assert by_name(report)["nvme_free"] is False


def test_fail_container_digest(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, env={}))
    assert by_name(report)["container_digest"] is False


def test_strict_lists_every_failure(tmp_path) -> None:
    report = run_preflight(**good_kwargs(tmp_path, env={}, meminfo_text=MEMINFO_LOW))
    with pytest.raises(PreflightError) as ei:
        report.strict()
    assert "ram" in str(ei.value) and "container_digest" in str(ei.value)


def test_probe_failure_becomes_failed_check(tmp_path) -> None:
    def broken_runner(cmd):
        raise FileNotFoundError("nvidia-smi not found")

    report = run_preflight(
        smi_xml=None,
        compute_caps_csv=None,
        topo_text=None,
        meminfo_text=MEMINFO_OK,
        cache_dir=tmp_path,
        env=ENV_OK,
        runner=broken_runner,
        disk_usage=lambda _p: FakeUsage(4 * 10**12),
    )
    assert not report.ok
    names = by_name(report)
    assert names["gpu_count"] is False  # zero GPUs parsed
    assert "gpu_probe" in names and names["gpu_probe"] is False
