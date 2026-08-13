"""Environment gate: determinism pins, fingerprint, container digest.

These run first (filename order) so a misconfigured node fails loudly before
any expensive window test. None of them require the `mok` wheel.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from mok_core.determinism import (
    CONTAINER_DIGEST_ENV,
    DeterminismError,
    assert_container_digest,
    enforce_determinism,
    environment_fingerprint,
)
from mok_core.determinism.env import _REQUIRED_ENV

_FRESH_INTERPRETER_SNIPPET = """
import torch
assert not torch.cuda.is_initialized()
from mok_core.determinism import enforce_determinism
enforce_determinism()          # must succeed BEFORE any CUDA context exists
assert torch.are_deterministic_algorithms_enabled()
assert not torch.backends.cudnn.benchmark
assert not torch.backends.cuda.matmul.allow_tf32
import os
print("OK", os.environ["CUBLAS_WORKSPACE_CONFIG"])
"""


def test_enforce_determinism_succeeds_pre_cuda() -> None:
    """In a fresh interpreter (guaranteed CUDA-uninitialized), enforce_determinism
    passes and pins the full env set — the exact startup order of every node."""
    proc = subprocess.run(
        [sys.executable, "-c", _FRESH_INTERPRETER_SNIPPET],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, f"enforce_determinism failed pre-CUDA:\n{proc.stderr}"
    assert "OK :4096:8" in proc.stdout


def test_enforce_determinism_in_process_pins() -> None:
    """Idempotent re-entry in this (possibly CUDA-initialized) process: env pins
    land and deterministic algorithms are on. The pre-CUDA ordering property is
    proven by the fresh-interpreter test above."""
    enforce_determinism(allow_uninitialized_cuda_check=not torch.cuda.is_initialized())
    for key, value in _REQUIRED_ENV.items():
        assert os.environ.get(key) == value, f"{key} not pinned"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32


def test_enforce_determinism_rejects_conflicting_pin() -> None:
    previous = os.environ.get("NCCL_ALGO")
    os.environ["NCCL_ALGO"] = "Tree"  # anything but the pinned "Ring"
    try:
        with pytest.raises(DeterminismError, match="NCCL_ALGO"):
            enforce_determinism(allow_uninitialized_cuda_check=False)
    finally:
        if previous is None:
            del os.environ["NCCL_ALGO"]
        else:
            os.environ["NCCL_ALGO"] = previous


def test_fingerprint_fields_present() -> None:
    fp = environment_fingerprint()
    assert fp.torch_version == torch.__version__
    assert fp.cuda_version  # "cpu" on CPU hosts, the CUDA version on the node
    assert fp.cudnn_version
    assert fp.python_version.count(".") == 2
    assert fp.platform
    assert fp.mok_version  # "absent" counts as present-and-truthful
    assert set(fp.env_pins) == set(_REQUIRED_ENV)
    payload = fp.to_json()
    assert json.dumps(payload, sort_keys=True)  # telemetry-serializable
    assert payload["driver_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES", "all")


def test_container_digest_env_check() -> None:
    """Inside the blessed container MOK_CONTAINER_DIGEST must match itself and
    reject a foreign digest; outside the container this is an expected failure
    (the manifest's digest gate cannot pass on an unblessed host)."""
    digest = os.environ.get(CONTAINER_DIGEST_ENV, "")
    if not digest:
        pytest.xfail(f"{CONTAINER_DIGEST_ENV} unset — not running inside the blessed container")
    assert_container_digest(digest)  # the manifest-pinned digest gate passes
    with pytest.raises(DeterminismError, match="container digest mismatch"):
        assert_container_digest("sha256:" + "00" * 32)
