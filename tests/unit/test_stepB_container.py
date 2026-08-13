"""Inspection tests for B/container/* — the blessed-image contract lines.

These pin the reproducibility levers by text: pinned base image, pinned torch
cu130 wheel, pinned mixture-of-kittens commit built without isolation, the
baked inductor cache env (max-autotune OFF), role dispatch, digest self-check.
``bash -n`` lints the entrypoint syntax; compose.yml must parse as YAML with
the three role services and full-node GPU reservations.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CONTAINER_DIR = Path(__file__).resolve().parents[2] / "B" / "container"
DOCKERFILE = (CONTAINER_DIR / "Dockerfile").read_text(encoding="utf-8")
ENTRYPOINT = (CONTAINER_DIR / "entrypoint.sh").read_text(encoding="utf-8")
COMPOSE = (CONTAINER_DIR / "compose.yml").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Dockerfile
# --------------------------------------------------------------------------- #


def test_dockerfile_base_image_and_multi_stage() -> None:
    assert "nvidia/cuda:13.0.0-devel-ubuntu22.04" in DOCKERFILE
    assert DOCKERFILE.count("FROM ${CUDA_IMAGE}") == 2  # builder + runtime stages
    assert "AS builder" in DOCKERFILE and "AS runtime" in DOCKERFILE


def test_dockerfile_pins_torch_and_python() -> None:
    assert "torch==2.10.0+cu130" in DOCKERFILE
    assert "download.pytorch.org/whl/cu130" in DOCKERFILE
    assert "python3.12" in DOCKERFILE


def test_dockerfile_pins_mok_commit_no_build_isolation() -> None:
    assert "git clone https://github.com/cursor/mixture-of-kittens" in DOCKERFILE
    assert "MOK_COMMIT=8f90b74" in DOCKERFILE
    assert "git checkout ${MOK_COMMIT}" in DOCKERFILE
    assert "--no-build-isolation" in DOCKERFILE


def test_dockerfile_installs_the_one_wheel() -> None:
    assert "COPY dist/mok_subnet-*.whl" in DOCKERFILE
    assert "pip install --no-cache-dir /tmp/wheels/mok_subnet-*.whl" in DOCKERFILE


def test_dockerfile_bakes_inductor_cache_and_determinism_env() -> None:
    assert "TORCHINDUCTOR_CACHE_DIR=/opt/mok/inductor-cache" in DOCKERFILE
    assert "TORCHINDUCTOR_MAX_AUTOTUNE=0" in DOCKERFILE
    assert "TORCHINDUCTOR_FX_GRAPH_CACHE=1" in DOCKERFILE
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in DOCKERFILE
    assert "NCCL_ALGO=Ring" in DOCKERFILE
    assert "NCCL_PROTO=Simple" in DOCKERFILE


def test_dockerfile_label_and_entrypoint() -> None:
    assert 'LABEL org.mok.spec_version="1"' in DOCKERFILE
    assert 'ENTRYPOINT ["/opt/mok/entrypoint.sh"]' in DOCKERFILE


# --------------------------------------------------------------------------- #
# entrypoint.sh
# --------------------------------------------------------------------------- #


def test_entrypoint_strict_mode_and_digest_check() -> None:
    assert "set -euo pipefail" in ENTRYPOINT
    assert "require_env MOK_CONTAINER_DIGEST" in ENTRYPOINT
    assert "/opt/mok/IMAGE_DIGEST" in ENTRYPOINT  # baked-digest self-check


def test_entrypoint_role_dispatch() -> None:
    for role in ("miner)", "validator)", "auditor)", "attest)", "calibrate)", "healthcheck)"):
        assert role in ENTRYPOINT
    assert 'torchrun --standalone --nproc-per-node="${NPROC}"' in ENTRYPOINT
    assert "-m C.miner.main" in ENTRYPOINT
    assert "python -m C.validator.main" in ENTRYPOINT
    assert "python -m C.auditor.main" in ENTRYPOINT
    assert "-m B.attestation.reference_step" in ENTRYPOINT
    assert "mok-calibrate" in ENTRYPOINT
    assert "python -m B.ops.healthcheck" in ENTRYPOINT


def test_entrypoint_execs_rather_than_forks() -> None:
    # every role launch must exec so signals reach the workload (PID 1)
    assert ENTRYPOINT.count("exec ") >= 6


def test_entrypoint_validates_miner_env() -> None:
    for var in ("R2_WRITE_ACCESS_KEY_ID", "R2_READ_ACCESS_KEY_ID", "BT_WALLET_HOTKEY", "BT_NETUID"):
        assert var in ENTRYPOINT


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_entrypoint_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(CONTAINER_DIR / "entrypoint.sh")], check=True)


# --------------------------------------------------------------------------- #
# compose.yml
# --------------------------------------------------------------------------- #


def test_compose_parses_with_role_services() -> None:
    doc = yaml.safe_load(COMPOSE)
    assert set(doc["services"]) == {"miner", "validator", "auditor", "healthcheck"}
    for role in ("miner", "validator", "auditor"):
        svc = doc["services"][role]
        assert svc["command"] == [role]
        assert svc["restart"] == "unless-stopped"
        assert svc["env_file"] == ".env"
        devices = svc["deploy"]["resources"]["reservations"]["devices"]
        assert devices[0]["driver"] == "nvidia"
        assert devices[0]["count"] == 8
        assert devices[0]["capabilities"] == ["gpu"]


def test_compose_shares_cache_and_wallet_volumes() -> None:
    doc = yaml.safe_load(COMPOSE)
    assert set(doc["volumes"]) == {"mok-shard-cache", "mok-wallets"}
    miner = doc["services"]["miner"]
    assert any("mok-shard-cache" in v for v in miner["volumes"])
    assert miner["shm_size"] == "32gb"
