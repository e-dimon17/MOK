"""Tests for C/core/phase.py — cumulative phase resolution, closed-form LR, accum ramp."""

from __future__ import annotations

import pytest

from C.core.phase import PhaseConfig, accum_at, lr_at, resolve_phase
from mok_core.config import (
    DatasetManifestRef,
    LRSpec,
    PhaseEntry,
    PhaseOverrides,
    PRFSpec,
    RunConfig,
    RunManifest,
    WindowConfig,
)

CFG = RunConfig()
TPS = CFG.tokens_per_inner_step  # 8192 * 8 * 8 = 524288
ANNEAL_LR = LRSpec(kind="wsd_linear_decay", peak_lr=3e-4, warmup_steps=2000, decay_total_tokens=400 * 10**9)


def _ds(name: str, seq_len: int = 4096) -> DatasetManifestRef:
    return DatasetManifestRef(
        name=name,
        merkle_root="44" * 32,
        num_shards=10,
        shard_bytes=512 * 2**20,
        seq_len=seq_len,
        tokens_total=10**12,
        tokenizer_hash="55" * 32,
    )


def _manifest(phase_table=None, datasets=None) -> RunManifest:
    return RunManifest(
        spec_version=1,
        run_id="stage2-test",
        netuid=77,
        network="test",
        config_hash="00" * 32,
        container_digest="sha256:" + "11" * 32,
        mok_commit="8f90b74",
        tk_commit="0badc0de",
        attention_backend="cudnn_det",
        start_block=1000,
        blocks_per_window=225,
        prf=PRFSpec(run_seed_hex="22" * 32),
        datasets=datasets if datasets is not None else (_ds("bulk"), _ds("anneal"), _ds("longdoc", 16384)),
        init_checkpoint_hash="33" * 32,
        phase_table=phase_table
        if phase_table is not None
        else (PhaseEntry(start_window=0, name="bulk"),),
    )


def _staged_manifest() -> RunManifest:
    """bulk@0 -> anneal@100 (D) -> capacity 0.5@150 -> context16k@200 (E) -> capacity 0.25@300."""
    m = _manifest(
        phase_table=(
            PhaseEntry(start_window=0, name="bulk"),
            PhaseEntry(start_window=100, name="anneal", overrides=PhaseOverrides(data="anneal", lr=ANNEAL_LR)),
        )
    )
    m = m.with_amendment(kind="capacity", effective_window=150, committed_block=5000, capacity_multiplier=0.5)
    m = m.with_amendment(
        kind="phase",
        effective_window=200,
        committed_block=6000,
        phase=PhaseEntry(
            start_window=200,
            name="context16k",
            overrides=PhaseOverrides(
                data="longdoc",
                seq_len=16384,
                tokens_per_rank_microbatch=16384,
                rope_theta=5e5,
                grad_accum=4,
                requires_restart=True,
            ),
        ),
    )
    return m.with_amendment(
        kind="capacity", effective_window=300, committed_block=8000, capacity_multiplier=0.25
    )


# --------------------------------------------------------------------------- #
# resolve_phase
# --------------------------------------------------------------------------- #


def test_resolve_defaults_at_window_zero():
    phase = resolve_phase(_staged_manifest(), CFG, 0)
    assert phase == PhaseConfig(
        name="bulk",
        data="bulk",  # defaults to the manifest's first dataset
        lr=CFG.inner.lr,
        seq_len=CFG.model.seq_len,
        tokens_per_rank_microbatch=CFG.window.tokens_per_rank_microbatch,
        rope_theta=CFG.model.rope_theta,
        grad_accum=CFG.window.grad_accum,
        capacity_multiplier=CFG.mok.schedule_capacity_multiplier,
        inner_steps=CFG.window.inner_steps,
        requires_restart=False,
    )


def test_resolve_anneal_phase_overrides():
    phase = resolve_phase(_staged_manifest(), CFG, 120)
    assert phase.name == "anneal"
    assert phase.data == "anneal"
    assert phase.lr == ANNEAL_LR
    assert phase.seq_len == CFG.model.seq_len
    assert phase.capacity_multiplier == 1.0


def test_capacity_amendment_layers_on_top_of_anneal():
    phase = resolve_phase(_staged_manifest(), CFG, 160)
    assert phase.name == "capacity_0.5"          # phase_at's entry
    assert phase.data == "anneal"                # cumulative: anneal's data survives
    assert phase.lr == ANNEAL_LR                 # cumulative: anneal's LR survives
    assert phase.capacity_multiplier == 0.5
    assert phase.requires_restart is False


def test_context16k_phase_keeps_layered_capacity():
    phase = resolve_phase(_staged_manifest(), CFG, 250)
    assert phase.name == "context16k"
    assert phase.data == "longdoc"
    assert phase.seq_len == 16384
    assert phase.tokens_per_rank_microbatch == 16384
    assert phase.rope_theta == 5e5
    assert phase.grad_accum == 4
    assert phase.capacity_multiplier == 0.5      # 150-amendment still layered
    assert phase.requires_restart is True


def test_later_capacity_amendment_keeps_context_overrides():
    phase = resolve_phase(_staged_manifest(), CFG, 310)
    assert phase.name == "capacity_0.25"
    assert phase.seq_len == 16384                # E overrides survive the capacity layer
    assert phase.data == "longdoc"
    assert phase.capacity_multiplier == 0.25
    # requires_restart marks the transition INTO an entry, not the whole lineage.
    assert phase.requires_restart is False


def test_resolve_boundary_is_inclusive():
    m = _staged_manifest()
    assert resolve_phase(m, CFG, 99).name == "bulk"
    assert resolve_phase(m, CFG, 100).name == "anneal"
    assert resolve_phase(m, CFG, 200).seq_len == 16384


def test_resolve_rejects_bad_inputs():
    with pytest.raises(ValueError):
        resolve_phase(_staged_manifest(), CFG, -1)
    with pytest.raises(ValueError):
        resolve_phase(_manifest(datasets=()), CFG, 0)


# --------------------------------------------------------------------------- #
# lr_at — golden values (consensus constants)
# --------------------------------------------------------------------------- #

FLAT = LRSpec(kind="wsd_flat", peak_lr=3e-4, warmup_steps=2000)
DECAY = LRSpec(kind="wsd_linear_decay", peak_lr=3e-4, warmup_steps=2000, decay_total_tokens=524_288_000)


def test_lr_warmup_golden_values():
    # consensus constant — change requires SPEC_VERSION bump
    assert lr_at(FLAT, 0, TPS) == 1.5e-07
    assert lr_at(FLAT, 999, TPS) == 0.00015
    assert lr_at(FLAT, 1999, TPS) == 0.0003     # warmup boundary: peak reached at warmup_steps - 1


def test_lr_flat_after_warmup():
    # consensus constant — change requires SPEC_VERSION bump
    assert lr_at(FLAT, 2000, TPS) == 0.0003
    assert lr_at(FLAT, 10**6, TPS) == 0.0003


def test_lr_linear_decay_golden_values():
    # decay_total_tokens = 1000 inner steps at TPS = 524288 tokens/step
    # consensus constant — change requires SPEC_VERSION bump
    assert lr_at(DECAY, 1999, TPS) == 0.0003    # still warming up / at peak
    assert lr_at(DECAY, 2000, TPS) == 0.0003    # decay starts here
    assert lr_at(DECAY, 2500, TPS) == 0.00015   # halfway through the token budget
    assert lr_at(DECAY, 3000, TPS) == 0.0       # decay endpoint: exactly zero
    assert lr_at(DECAY, 3500, TPS) == 0.0       # clamped past the endpoint


def test_lr_const_ignores_warmup():
    spec = LRSpec(kind="const", peak_lr=3e-4, warmup_steps=2000, const_lr=1e-4)
    assert lr_at(spec, 0, TPS) == 1e-4
    assert lr_at(spec, 10**6, TPS) == 1e-4


def test_lr_zero_warmup_starts_at_peak():
    spec = LRSpec(kind="wsd_flat", peak_lr=3e-4, warmup_steps=0)
    assert lr_at(spec, 0, TPS) == 3e-4


def test_lr_validation():
    with pytest.raises(ValueError):
        lr_at(FLAT, -1, TPS)
    with pytest.raises(ValueError):
        lr_at(DECAY, 5000, 0)


# --------------------------------------------------------------------------- #
# accum_at
# --------------------------------------------------------------------------- #


def test_accum_ramp_endpoints_and_monotonicity():
    wc = WindowConfig(accum_ramp_tokens=1000, accum_ramp_start=2, grad_accum=8)
    assert accum_at(wc, 0) == 2
    assert accum_at(wc, 999) == 7
    assert accum_at(wc, 1000) == 8
    assert accum_at(wc, 10**15) == 8
    prev = 0
    for tokens in range(0, 2001, 7):
        cur = accum_at(wc, tokens)
        assert isinstance(cur, int)
        assert 2 <= cur <= 8
        assert cur >= prev, "accum ramp must be monotonic"
        prev = cur


def test_accum_default_config_golden_points():
    wc = CFG.window  # ramp 2 -> 8 over 50B tokens
    assert accum_at(wc, 0) == 2
    assert accum_at(wc, 25 * 10**9) == 5
    assert accum_at(wc, 50 * 10**9) == 8


def test_accum_degenerate_ramp():
    wc = WindowConfig(accum_ramp_tokens=0, accum_ramp_start=2, grad_accum=8)
    assert accum_at(wc, 0) == 8
    flat = WindowConfig(accum_ramp_tokens=1000, accum_ramp_start=8, grad_accum=8)
    assert accum_at(flat, 500) == 8
    with pytest.raises(ValueError):
        accum_at(wc, -1)
