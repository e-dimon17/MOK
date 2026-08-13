"""Tests for B/attestation/challenge.py — derivation goldens + config merge."""

from __future__ import annotations

import hashlib

import pytest

from B.attestation.challenge import (
    ATTEST_DOMAIN,
    BASE_CONFIG_PATH,
    TOY4L_CONFIG_PATH,
    AttestationChallenge,
    challenge_run_config,
    make_challenge,
    toy4l_overlay,
)
from mok_core.config import load_yaml

BLOCK_HASH = bytes([1]) * 32


def test_make_challenge_golden_vector() -> None:
    """# consensus constant — change requires SPEC_VERSION bump"""
    ch = make_challenge(BLOCK_HASH, issued_block=100)
    assert ch.challenge_id == "227a506a6bf08680"
    assert ch.seed == 4264113716381589857
    assert ch.issued_block == 100
    assert ch.deadline_s == 420.0
    assert ch.inner_steps == 20


def test_derivation_matches_documented_rule() -> None:
    digest = hashlib.blake2b(BLOCK_HASH + ATTEST_DOMAIN, digest_size=32).digest()
    ch = make_challenge(BLOCK_HASH, issued_block=0)
    assert ch.challenge_id == digest[:8].hex()
    assert ch.seed == int.from_bytes(digest[8:16], "little") & ((1 << 63) - 1)


def test_different_block_hashes_give_different_challenges() -> None:
    a = make_challenge(BLOCK_HASH, 1)
    b = make_challenge(bytes([2]) * 32, 1)
    assert a.challenge_id != b.challenge_id
    assert a.seed != b.seed


def test_overlay_is_toy4l_verbatim() -> None:
    ch = make_challenge(BLOCK_HASH, 1)
    assert ch.model_overlay == load_yaml(TOY4L_CONFIG_PATH)
    assert ch.model_overlay["model"]["num_layers"] == 4
    assert ch.model_overlay["model"]["ep_size"] == 8


def test_make_challenge_rejects_bad_hash_length() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        make_challenge(b"\x01" * 31, 0)


def test_challenge_validators() -> None:
    good = make_challenge(BLOCK_HASH, 1)
    with pytest.raises(ValueError, match="challenge_id"):
        AttestationChallenge(**{**good.model_dump(), "challenge_id": "XYZ"})
    with pytest.raises(ValueError, match="seed"):
        AttestationChallenge(**{**good.model_dump(), "seed": -1})
    with pytest.raises(ValueError, match="inner_steps"):
        AttestationChallenge(**{**good.model_dump(), "inner_steps": 0})
    with pytest.raises(ValueError, match="deadline_s"):
        AttestationChallenge(**{**good.model_dump(), "deadline_s": 0.0})


def test_challenge_run_config_merges_base_plus_overlay() -> None:
    ch = make_challenge(BLOCK_HASH, 1)
    cfg = challenge_run_config(ch)
    # toy4L overrides
    assert cfg.model.num_layers == 4
    assert cfg.model.hidden_size == 1024
    assert cfg.model.num_experts == 16
    assert cfg.window.tokens_per_rank_microbatch == 8192
    # challenge.inner_steps pins window.inner_steps over BOTH base and overlay
    assert cfg.window.inner_steps == ch.inner_steps == 20
    # base.yaml values survive where the overlay is silent
    base = load_yaml(BASE_CONFIG_PATH)
    assert cfg.model.rope_theta == base["model"]["rope_theta"]
    assert cfg.compression.topk == base["compression"]["topk"]


def test_challenge_run_config_honors_inner_steps_override() -> None:
    ch = make_challenge(BLOCK_HASH, 1, inner_steps=3)
    assert challenge_run_config(ch).window.inner_steps == 3


def test_toy4l_overlay_reads_the_file_fresh() -> None:
    assert toy4l_overlay() == load_yaml(TOY4L_CONFIG_PATH)
