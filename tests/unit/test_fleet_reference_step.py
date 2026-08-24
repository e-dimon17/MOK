"""Tests for fleet/attestation/reference_step.py — the CPU attestation gate.

The tiny-overlay challenge runs the FULL respond path (philox shards ->
DatasetShardIndex -> minimal manifest -> WindowBatchPlan -> unmodified
InnerLoop -> state root) on CPU with the reference backend, and pins:

  1. determinism — two runs of the same challenge produce the same root;
  2. derive_expected == run_reference (verifier and responder share the path);
  3. a different seed produces a different root (the challenge binds compute).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import fleet.attestation.reference_step as reference_step
from fleet.attestation.challenge import derive_expected, make_challenge
from fleet.attestation.reference_step import (
    ATTEST_DATA_WORLD_SIZE,
    ATTEST_NUM_SHARDS,
    AttestationResponse,
    attestation_manifest,
    attestation_run_seed,
    challenge_run_config,
    main,
    run_reference,
    write_attestation_shards,
)
from mok_core.data.assignment import sequences_per_window
from mok_core.determinism import philox

BLOCK_HASH = bytes([1]) * 32

TINY_OVERLAY: dict[str, Any] = {
    "model": {
        "num_layers": 2,
        "num_dense_layers": 0,
        "hidden_size": 256,
        "num_q_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 128,
        "vocab_size": 512,
        "seq_len": 256,
        "num_experts": 8,
        "top_k": 2,
        "intermediate_size": 256,
        "ep_size": 4,
    },
    "window": {
        "inner_steps": 2,
        "tokens_per_rank_microbatch": 512,
        "grad_accum": 1,
        "accum_ramp_start": 1,
    },
    "inner": {"lr": {"kind": "const", "const_lr": 0.02}},
}


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def tiny_challenge():
    ch = make_challenge(BLOCK_HASH, issued_block=100)
    return ch.model_copy(update={"model_overlay": TINY_OVERLAY, "inner_steps": 2})


@pytest.fixture(scope="module")
def response(tiny_challenge) -> AttestationResponse:
    return run_reference(tiny_challenge, backend="reference", device="cpu")


# --------------------------------------------------------------------------- #
# Derivation goldens
# --------------------------------------------------------------------------- #


def test_run_seed_golden_vector() -> None:
    """# consensus constant — change requires SPEC_VERSION bump"""
    ch = make_challenge(BLOCK_HASH, 100)
    assert (
        attestation_run_seed(ch).hex()
        == "c780e611bdb56f64ec62b9e3a9f2d1cc25b83a35ec2fc06e116fa1140a8121a2"
    )


def test_run_seed_matches_documented_rule(tiny_challenge) -> None:
    h = hashlib.blake2b(digest_size=32)
    h.update(b"attest.data.v1")
    h.update(bytes.fromhex(tiny_challenge.challenge_id))
    h.update(tiny_challenge.seed.to_bytes(8, "little"))
    assert attestation_run_seed(tiny_challenge) == h.digest()


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #


def test_shards_deterministic_and_indexed(tiny_challenge, tmp_path: Path) -> None:
    cfg = challenge_run_config(tiny_challenge)
    index_a, ref_a = write_attestation_shards(tiny_challenge, cfg, tmp_path / "a")
    index_b, ref_b = write_attestation_shards(tiny_challenge, cfg, tmp_path / "b")
    assert index_a.shard_hashes == index_b.shard_hashes  # rank/platform independent
    assert ref_a == ref_b
    assert index_a.num_shards == ATTEST_NUM_SHARDS
    for i in range(ATTEST_NUM_SHARDS):
        a = (tmp_path / "a" / f"shard-{i:04d}.bin").read_bytes()
        assert (tmp_path / "b" / f"shard-{i:04d}.bin").read_bytes() == a
        assert hashlib.blake2b(a, digest_size=32).digest() == index_a.leaf(i)


def test_shard_tokens_come_from_philox_and_are_vocab_bounded(tiny_challenge, tmp_path: Path) -> None:
    cfg = challenge_run_config(tiny_challenge)
    index, ref = write_attestation_shards(tiny_challenge, cfg, tmp_path)
    seqs_per_shard = ref.shard_bytes // (2 * cfg.model.seq_len)
    expected = philox(tiny_challenge.seed).integers(
        0, cfg.model.vocab_size, size=(ATTEST_NUM_SHARDS * seqs_per_shard, cfg.model.seq_len),
        dtype=np.uint16,
    )
    got = np.frombuffer((tmp_path / "shard-0000.bin").read_bytes(), dtype="<u2")
    assert np.array_equal(got, expected[:seqs_per_shard].reshape(-1))
    assert int(got.max()) < cfg.model.vocab_size


def test_dataset_sized_for_the_full_tier_a_geometry(tiny_challenge, tmp_path: Path) -> None:
    cfg = challenge_run_config(tiny_challenge)
    _, ref = write_attestation_shards(tiny_challenge, cfg, tmp_path)
    need = sequences_per_window(
        tokens_per_rank_microbatch=cfg.window.tokens_per_rank_microbatch,
        grad_accum=cfg.window.grad_accum,
        inner_steps=cfg.window.inner_steps,
        ranks=ATTEST_DATA_WORLD_SIZE,
        seq_len=cfg.model.seq_len,
    )
    assert ref.tokens_total >= need * cfg.model.seq_len


def test_attestation_manifest_is_pure(tiny_challenge, tmp_path: Path) -> None:
    cfg = challenge_run_config(tiny_challenge)
    _, ref = write_attestation_shards(tiny_challenge, cfg, tmp_path)
    m1 = attestation_manifest(tiny_challenge, cfg, ref)
    m2 = attestation_manifest(tiny_challenge, cfg, ref)
    assert m1.manifest_hash() == m2.manifest_hash()
    assert m1.prf.run_seed_hex == attestation_run_seed(tiny_challenge).hex()
    assert m1.datasets[0].name == "attest"


# --------------------------------------------------------------------------- #
# The gate: determinism + verifier/responder equality
# --------------------------------------------------------------------------- #


def test_run_reference_is_deterministic(tiny_challenge, response) -> None:
    again = run_reference(tiny_challenge, backend="reference", device="cpu")
    assert again.state_root == response.state_root
    assert response.challenge_id == tiny_challenge.challenge_id
    assert response.wall_time_s > 0.0


def test_derive_expected_equals_responder_root(tiny_challenge, response) -> None:
    assert derive_expected(tiny_challenge, device="cpu") == response.state_root


def test_different_seed_different_root(tiny_challenge, response) -> None:
    other = tiny_challenge.model_copy(update={"seed": tiny_challenge.seed ^ 1})
    assert run_reference(other, backend="reference", device="cpu").state_root != response.state_root


def test_fingerprint_carries_environment(response) -> None:
    fp = response.fingerprint
    assert fp["torch_version"] == torch.__version__
    assert "env_pins" in fp and "python_version" in fp


# --------------------------------------------------------------------------- #
# GPU gate (torchrun --nproc-per-node=8; excluded by default addopts)
# --------------------------------------------------------------------------- #


@pytest.mark.gpu
def test_full_toy4l_attestation_on_hardware() -> None:
    """The real challenge on the real kernel: 8-rank mok backend, twice, same
    root, inside the deadline. This is the miner-side attestation dry run."""
    import torch.distributed as dist

    assert dist.is_initialized() and dist.get_world_size() == 8
    challenge = make_challenge(BLOCK_HASH, issued_block=100)
    device = f"cuda:{dist.get_rank()}"
    first = run_reference(challenge, backend="mok", device=device)
    second = run_reference(challenge, backend="mok", device=device)
    assert first.state_root == second.state_root
    assert first.wall_time_s <= challenge.deadline_s


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_main_emits_response_json(tiny_challenge, response, tmp_path: Path, monkeypatch) -> None:
    challenge_file = tmp_path / "challenge.json"
    challenge_file.write_text(json.dumps(tiny_challenge.model_dump()), encoding="utf-8")
    out_file = tmp_path / "response.json"

    calls: dict[str, Any] = {}

    def fake_run_reference(challenge, *, backend, device, comm=None):
        calls["challenge"] = challenge
        calls["backend"] = backend
        return response

    monkeypatch.setattr(reference_step, "run_reference", fake_run_reference)
    monkeypatch.setattr(reference_step, "enforce_determinism", lambda: None)
    assert main(["--challenge", str(challenge_file), "--backend", "reference",
                 "--device", "cpu", "--out", str(out_file)]) == 0
    emitted = json.loads(out_file.read_text(encoding="utf-8"))
    assert emitted["state_root"] == response.state_root
    assert calls["challenge"] == tiny_challenge
    assert calls["backend"] == "reference"
