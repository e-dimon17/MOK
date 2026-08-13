"""Inductor-cache determinism: MOK_COMPILE=1, two fresh processes, equal roots.

The blessed container ships a pre-baked inductor FX-graph cache (max-autotune
is banned — protocol decision #4). This test proves the property that bake
relies on: two COLD process trees compiling through the SAME inductor cache
directory produce bitwise-identical window results — the first run populates
the cache, the second hits it, and neither path may change the math.

Mechanics: this module is a torchrun DRIVER, not a distributed test — it
launches `torchrun --standalone --nproc-per-node=8 tests/gpu/_synthetic.py`
twice (a 5-step toy window that prints STATE_ROOT=<hex> on rank 0) and
compares the printed roots. Under an outer `torchrun ... -m pytest` run, only
RANK 0 drives; other ranks skip (no collectives here, so ranks stay in
lockstep). If nested torchrun cannot launch on this host, the test SKIPS with
the probe's stderr.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import _synthetic as synth
import pytest
import torch

_GPU_DIR = Path(__file__).resolve().parent
_PROBE_TIMEOUT_S = 120
_WINDOW_TIMEOUT_S = 3600
_ROOT_RE = re.compile(r"^STATE_ROOT=([0-9a-f]{64})$", re.MULTILINE)

# Env that must NOT leak from an outer torchrun into a nested launch.
_ELASTIC_ENV_PREFIXES = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "ROLE_NAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC",
    "OMP_NUM_THREADS",
)


def _clean_env(extra: dict[str, str]) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(k == p or k.startswith(p + "_") or k.startswith(p) for p in _ELASTIC_ENV_PREFIXES)
    }
    env.update(extra)
    return env


def run_torchrun(
    script_args: list[str], *, nproc: int, extra_env: dict[str, str], timeout_s: float
) -> subprocess.CompletedProcess[str]:
    """Nested-torchrun helper: `--standalone` picks a free rendezvous port so a
    launch inside an outer torchrun job cannot collide with it."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={nproc}",
        *script_args,
    ]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=_clean_env(extra_env),
        cwd=str(_GPU_DIR.parents[1]),  # repo root
        check=False,
    )


def _nested_torchrun_supported(tmp_path: Path) -> str | None:
    """Return None if a 1-process nested torchrun works, else the failure text."""
    probe = tmp_path / "probe.py"
    probe.write_text("import os\nprint('PROBE_OK', os.environ.get('RANK'))\n", encoding="utf-8")
    try:
        proc = run_torchrun([str(probe)], nproc=1, extra_env={}, timeout_s=_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return "probe timed out"
    if proc.returncode != 0 or "PROBE_OK" not in proc.stdout:
        return f"probe rc={proc.returncode}: {proc.stderr[-2000:]}"
    return None


def _launch_window(data_dir: Path, cache_dir: Path, nproc: int) -> str:
    extra_env = {
        "MOK_COMPILE": "1",  # torch.compile ON — the property under test
        "TORCHINDUCTOR_CACHE_DIR": str(cache_dir),  # SHARED between the two runs
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_MAX_AUTOTUNE": "0",  # autotuning is timing-dependent: banned
    }
    proc = run_torchrun(
        [str(_GPU_DIR / "_synthetic.py"), "--data-dir", str(data_dir), "--inner-steps", "5"],
        nproc=nproc,
        extra_env=extra_env,
        timeout_s=_WINDOW_TIMEOUT_S,
    )
    assert proc.returncode == 0, (
        f"compiled toy window failed (rc={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-6000:]}"
    )
    match = _ROOT_RE.search(proc.stdout)
    assert match, f"no STATE_ROOT line in output:\n{proc.stdout[-4000:]}"
    return match.group(1)


def test_compile_cache_two_cold_runs_equal_roots(tmp_path: Path, toy_cfg) -> None:
    if os.environ.get("RANK", "0") != "0":
        pytest.skip("nested-torchrun driver runs on RANK 0 only")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable — needs the Tier-A node")
    nproc = toy_cfg.model.ep_size
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"needs {nproc} GPUs, found {torch.cuda.device_count()}")
    if torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("MoK requires SM103 (B300)")
    unsupported = _nested_torchrun_supported(tmp_path)
    if unsupported is not None:
        pytest.skip(f"nested torchrun unsupported on this host: {unsupported}")

    data_dir = tmp_path / "shards"
    synth.write_shard_files(data_dir)  # only this driver process reads/writes it
    cache_dir = tmp_path / "inductor-cache"
    cache_dir.mkdir()

    root_cold = _launch_window(data_dir, cache_dir, nproc)  # populates the cache
    assert any(cache_dir.iterdir()), "first compiled run left the inductor cache empty"
    root_warm = _launch_window(data_dir, cache_dir, nproc)  # served from the cache

    assert root_cold == root_warm, (
        "inductor-cache determinism broken: cold-compile root "
        f"{root_cold} != cache-hit root {root_warm}. The container's baked cache "
        "cannot be trusted — do NOT launch until this reproduces bitwise."
    )
