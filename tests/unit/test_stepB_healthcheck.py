"""Tests for B/ops/healthcheck.py — canned probe outputs, assembly, CLI loop."""

from __future__ import annotations

import json
from pathlib import Path

from test_stepB_preflight import FakeUsage, topo_text

from B.ops.healthcheck import (
    HealthCheck,
    clock_sync,
    disk_space,
    gpu_health,
    main,
    nvlink_health,
    run_healthchecks,
    storage_reachability,
)

GPU_OK = "\n".join(f"{i}, NVIDIA B300, 45, 612.50, 0" for i in range(8))
CHRONY_OK = (
    "Reference ID    : A9FEA9FE (time.example.com)\n"
    "Stratum         : 2\n"
    "System time     : 0.000012340 seconds fast of NTP time\n"
    "Leap status     : Normal\n"
)
CHRONY_DRIFT = "System time     : 3.20000 seconds slow of NTP time\n"
TIMEDATECTL_OK = "  Local time: ...\nSystem clock synchronized: yes\n  NTP service: active\n"
TIMEDATECTL_BAD = "System clock synchronized: no\nNTP service: inactive\n"


# --------------------------------------------------------------------------- #
# gpu_health
# --------------------------------------------------------------------------- #


def test_gpu_health_nominal() -> None:
    check = gpu_health(GPU_OK)
    assert check.ok
    assert "8 GPUs nominal" in check.detail


def test_gpu_health_ecc_error_fails() -> None:
    bad = GPU_OK.replace("3, NVIDIA B300, 45, 612.50, 0", "3, NVIDIA B300, 45, 612.50, 2")
    check = gpu_health(bad)
    assert not check.ok
    assert "gpu3" in check.detail and "ECC" in check.detail


def test_gpu_health_over_temperature_fails() -> None:
    bad = GPU_OK.replace("5, NVIDIA B300, 45", "5, NVIDIA B300, 97")
    check = gpu_health(bad)
    assert not check.ok
    assert "97C" in check.detail


def test_gpu_health_na_ecc_is_note_not_failure() -> None:
    check = gpu_health("0, NVIDIA B300, 40, 300.00, [N/A]")
    assert check.ok
    assert "ECC counter N/A" in check.detail


def test_gpu_health_empty_output_fails() -> None:
    assert not gpu_health("").ok


# --------------------------------------------------------------------------- #
# nvlink + clock + disk + storage
# --------------------------------------------------------------------------- #


def test_nvlink_health_pass_and_fail() -> None:
    assert nvlink_health(topo_text()).ok
    check = nvlink_health(topo_text(break_pair=(1, 6)))
    assert not check.ok and "1-6:SYS" in check.detail
    assert not nvlink_health("garbage").ok


def test_clock_sync_chronyc() -> None:
    assert clock_sync(CHRONY_OK).ok
    drift = clock_sync(CHRONY_DRIFT)
    assert not drift.ok and "3.2" in drift.detail


def test_clock_sync_timedatectl() -> None:
    assert clock_sync(TIMEDATECTL_OK).ok
    assert not clock_sync(TIMEDATECTL_BAD).ok


def test_clock_sync_garbage_and_empty() -> None:
    assert not clock_sync("no clock info here").ok
    assert not clock_sync("").ok


def test_disk_space() -> None:
    assert disk_space("/", min_free_bytes=10, disk_usage=lambda _p: FakeUsage(100)).ok
    assert not disk_space("/", min_free_bytes=200, disk_usage=lambda _p: FakeUsage(100)).ok


async def test_storage_reachability_reachable_even_when_absent() -> None:
    class Storage:
        async def object_exists(self, bucket, key):
            return False

    check = await storage_reachability(Storage(), object())
    assert check.ok and "absent" in check.detail and "reachable" in check.detail


async def test_storage_reachability_transport_failure() -> None:
    class Storage:
        async def object_exists(self, bucket, key):
            raise ConnectionError("dns broke")

    check = await storage_reachability(Storage(), object())
    assert not check.ok and "dns broke" in check.detail


# --------------------------------------------------------------------------- #
# Assembly + CLI
# --------------------------------------------------------------------------- #


async def test_run_healthchecks_assembles_all_checks(tmp_path: Path) -> None:
    class Storage:
        async def object_exists(self, bucket, key):
            return True

    report = await run_healthchecks(
        gpu_csv=GPU_OK,
        topo_text=topo_text(),
        clock_text=CHRONY_OK,
        storage=Storage(),
        bucket=object(),
        cache_dir=tmp_path,
        min_free_bytes=1,
        clock=lambda: 123.0,
    )
    assert report.ok
    assert report.ts == 123.0
    assert [c.name for c in report.checks] == ["gpu", "nvlink", "clock", "storage", "disk"]
    payload = report.to_json()
    assert payload["ok"] is True and len(payload["checks"]) == 5


async def test_run_healthchecks_probe_failures_render_as_checks(tmp_path: Path) -> None:
    def broken(cmd):
        raise FileNotFoundError(cmd[0])

    report = await run_healthchecks(cache_dir=tmp_path, min_free_bytes=1, runner=broken)
    names = {c.name: c.ok for c in report.checks}
    assert names["gpu"] is False and names["nvlink"] is False and names["clock"] is False
    assert isinstance(report.checks[0], HealthCheck)


def test_main_emits_one_json_report(tmp_path: Path, capsys) -> None:
    (tmp_path / "gpu.csv").write_text(GPU_OK, encoding="utf-8")
    (tmp_path / "topo.txt").write_text(topo_text(), encoding="utf-8")
    (tmp_path / "clock.txt").write_text(CHRONY_OK, encoding="utf-8")
    rc = main(
        [
            "--iterations", "1",
            "--interval", "0",
            "--cache-dir", str(tmp_path),
            "--min-free-gb", "0.000001",
            "--gpu-csv", str(tmp_path / "gpu.csv"),
            "--topo", str(tmp_path / "topo.txt"),
            "--clock-status", str(tmp_path / "clock.txt"),
        ]
    )
    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["ok"] is True


def test_main_exit_code_reflects_health(tmp_path: Path, capsys) -> None:
    (tmp_path / "gpu.csv").write_text("", encoding="utf-8")  # no GPUs -> unhealthy
    (tmp_path / "topo.txt").write_text(topo_text(), encoding="utf-8")
    (tmp_path / "clock.txt").write_text(CHRONY_OK, encoding="utf-8")
    rc = main(
        [
            "--iterations", "1",
            "--cache-dir", str(tmp_path),
            "--min-free-gb", "0.000001",
            "--gpu-csv", str(tmp_path / "gpu.csv"),
            "--topo", str(tmp_path / "topo.txt"),
            "--clock-status", str(tmp_path / "clock.txt"),
        ]
    )
    assert rc == 1
    assert json.loads(capsys.readouterr().out.strip())["ok"] is False
