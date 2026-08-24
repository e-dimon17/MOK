"""subnet/core/scoring.py — score formulas, EMA, OpenSkill, weights ladder, eval pools."""

from __future__ import annotations

import json

import pytest

from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest
from mok_core.config.schemas import LRSpec, RunConfig, ScoringConfig
from subnet.core.phase import PhaseConfig, resolve_phase
from subnet.core.scoring import (
    RANDOM_POOL_UID,
    BinaryEMA,
    EvalPools,
    OpenSkillBook,
    binary_indicator,
    compute_weights,
    final_score,
    gradient_score,
    sync_score,
)

RUN_SEED = bytes(32)
BLOCK_HASH = b"\x01" * 32


# --------------------------------------------------------------------------- #
# Score formulas
# --------------------------------------------------------------------------- #


class TestGradientScore:
    def test_relative_improvement(self):
        assert gradient_score(2.0, 1.5) == pytest.approx(0.25)

    def test_regression_is_negative(self):
        assert gradient_score(2.0, 2.5) == pytest.approx(-0.25)

    def test_degenerate_inputs_zero(self):
        assert gradient_score(0.0, 1.0) == 0.0
        assert gradient_score(-1.0, 0.5) == 0.0
        assert gradient_score(float("nan"), 1.0) == 0.0
        assert gradient_score(2.0, float("inf")) == 0.0


class TestBinaryIndicator:
    def test_own_better(self):
        assert binary_indicator(0.3, 0.1) == 1

    def test_own_worse_or_equal(self):
        assert binary_indicator(0.1, 0.3) == -1
        assert binary_indicator(0.2, 0.2) == -1  # equality counts against the miner


class TestSyncScore:
    def test_goldens(self):
        assert sync_score(0.0) == pytest.approx(1.0)
        assert sync_score(1.5) == pytest.approx(0.5**2.5)  # 0.1767766952966369
        assert sync_score(3.0) == 0.0
        assert sync_score(10.0) == 0.0  # capped at max_behind

    def test_custom_max_behind(self):
        assert sync_score(1.0, max_behind=2.0) == pytest.approx(0.5**2.5)

    def test_invalid_max_behind(self):
        with pytest.raises(ValueError):
            sync_score(1.0, max_behind=0.0)


# --------------------------------------------------------------------------- #
# BinaryEMA
# --------------------------------------------------------------------------- #


class TestBinaryEMA:
    def test_update_formula(self):
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=2)
        assert ema.update(1, 1, window=0) == pytest.approx(0.5)
        assert ema.update(1, 1, window=1) == pytest.approx(0.75)
        assert ema.update(1, -1, window=2) == pytest.approx(-0.125)
        assert ema.value(1) == pytest.approx(-0.125)

    def test_unseen_uid_value_zero(self):
        assert BinaryEMA(0.05, 0.1, 10).value(99) == 0.0

    def test_warmup_always_passes(self):
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=3)
        ema.update(1, -1, window=10)  # ema = -0.5, clearly below threshold
        assert ema.passes(1, 10)
        assert ema.passes(1, 12)      # window - first(10) = 2 < 3 -> warmup
        assert not ema.passes(1, 13)  # warmup over, max(0, -0.5) < 0.1
        assert ema.passes(42, 999)    # never-seen uid: inside warmup by definition

    def test_threshold_gate_after_warmup(self):
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=1)
        ema.update(1, 1, window=0)    # ema = 0.5
        assert ema.passes(1, 5)
        ema2 = BinaryEMA(alpha=0.05, threshold=0.1, warmup_windows=1)
        ema2.update(2, 1, window=0)   # ema = 0.05 < 0.1
        assert not ema2.passes(2, 5)

    def test_invalid_indicator_rejected(self):
        with pytest.raises(ValueError):
            BinaryEMA(0.5, 0.1, 1).update(1, 0, window=0)

    def test_state_roundtrip_via_json(self):
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=2)
        ema.update(1, 1, window=3)
        ema.update(2, -1, window=4)
        state = json.loads(json.dumps(ema.state_dict()))
        restored = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=2)
        restored.load_state_dict(state)
        assert restored.value(1) == ema.value(1)
        assert restored.value(2) == ema.value(2)
        assert restored.passes(1, 4) == ema.passes(1, 4)
        assert restored.passes(2, 100) == ema.passes(2, 100)

    def test_reset(self):
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=1)
        ema.update(1, -1, window=0)
        ema.reset(1)
        assert ema.value(1) == 0.0
        assert ema.passes(1, 100)  # back inside warmup


# --------------------------------------------------------------------------- #
# OpenSkillBook
# --------------------------------------------------------------------------- #


class TestOpenSkillBook:
    def test_unseen_uid_default_ordinal_zero(self):
        assert OpenSkillBook(beta=7.0, tau=0.1).ordinal(5) == 0.0

    def test_determinism_two_books_same_sequence(self):
        windows = [
            {1: 0.30, 2: 0.20, 3: 0.10},
            {1: 0.25, 2: 0.28, 3: 0.05},
            {2: 0.30, 3: 0.29, 4: 0.10},
        ]
        book_a = OpenSkillBook(beta=7.0, tau=0.1)
        book_b = OpenSkillBook(beta=7.0, tau=0.1)
        for scores in windows:
            book_a.rate_window(dict(scores))
            # different insertion order must not matter
            book_b.rate_window(dict(sorted(scores.items(), reverse=True)))
        for uid in (1, 2, 3, 4):
            assert book_a.ordinal(uid) == book_b.ordinal(uid)
            assert book_a.mu_sigma(uid) == book_b.mu_sigma(uid)

    def test_better_scores_earn_higher_ordinal(self):
        book = OpenSkillBook(beta=7.0, tau=0.1)
        for _ in range(5):
            book.rate_window({1: 0.30, 2: 0.10})
        assert book.ordinal(1) > book.ordinal(2)

    def test_single_participant_noop_but_registered(self):
        book = OpenSkillBook(beta=7.0, tau=0.1)
        book.rate_window({7: 0.5})
        assert book.ordinal(7) == 0.0  # default rating: mu - 3*sigma = 0

    def test_reset_returns_to_default(self):
        book = OpenSkillBook(beta=7.0, tau=0.1)
        for _ in range(3):
            book.rate_window({1: 0.0, 2: 0.5})
        assert book.ordinal(1) != 0.0
        book.reset(1)
        assert book.ordinal(1) == 0.0

    def test_state_roundtrip_via_json(self):
        book = OpenSkillBook(beta=7.0, tau=0.1)
        book.rate_window({1: 0.3, 2: 0.1})
        book.rate_window({1: 0.2, 2: 0.4})
        state = json.loads(json.dumps(book.state_dict()))
        restored = OpenSkillBook(beta=7.0, tau=0.1)
        restored.load_state_dict(state)
        for uid in (1, 2):
            assert restored.ordinal(uid) == book.ordinal(uid)
        # restored book continues identically
        book.rate_window({1: 0.5, 2: 0.1})
        restored.rate_window({1: 0.5, 2: 0.1})
        assert restored.ordinal(1) == book.ordinal(1)


class TestFinalScore:
    def test_composition(self):
        book = OpenSkillBook(beta=7.0, tau=0.1)
        for _ in range(3):
            book.rate_window({1: 0.3, 2: 0.1})
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=1)
        ema.update(1, 1, window=0)
        expected = max(0.0, book.ordinal(1)) * 0.5 * sync_score(0.0)
        assert final_score(1, book, ema, sync_score(0.0)) == pytest.approx(expected)

    def test_negative_ordinal_clamped(self):
        book = OpenSkillBook(beta=7.0, tau=0.1)
        book.rate_window({1: 0.0, 2: 0.9})   # uid 1 loses -> negative ordinal
        assert book.ordinal(1) < 0.0
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=1)
        ema.update(1, 1, window=0)
        assert final_score(1, book, ema, 1.0) == 0.0

    def test_negative_ema_clamped(self):
        book = OpenSkillBook(beta=7.0, tau=0.1)
        book.rate_window({1: 0.9, 2: 0.0})
        ema = BinaryEMA(alpha=0.5, threshold=0.1, warmup_windows=1)
        ema.update(1, -1, window=0)
        assert final_score(1, book, ema, 1.0) == 0.0


# --------------------------------------------------------------------------- #
# Weights ladder
# --------------------------------------------------------------------------- #


def _cfg() -> ScoringConfig:
    return ScoringConfig()


class TestComputeWeights:
    def test_empty_and_all_nonpositive(self):
        assert compute_weights({}, _cfg()) == {}
        assert compute_weights({1: 0.0, 2: -0.5}, _cfg()) == {}

    def test_single_peer_gets_everything(self):
        assert compute_weights({5: 0.2}, _cfg()) == {5: pytest.approx(1.0)}

    def test_sums_to_one(self):
        scores = {uid: 1.0 / (uid + 1) for uid in range(30)}
        weights = compute_weights(scores, _cfg(), gather_count=20, reserve_count=10)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert len(weights) == 30

    def test_ramp_shape_and_monotonicity(self):
        scores = {uid: 100.0 - uid for uid in range(30)}
        weights = compute_weights(scores, _cfg(), gather_count=20, reserve_count=10)
        ranked = sorted(scores, key=lambda u: -scores[u])
        ws = [weights[u] for u in ranked]
        # non-increasing along the ladder
        assert all(a >= b for a, b in zip(ws, ws[1:], strict=False))
        # gather ramp: top weight / bottom gather weight == top_ratio
        assert ws[0] / ws[19] == pytest.approx(_cfg().top_ratio)
        # reserve stays below the gather floor and decays geometrically
        assert max(ws[20:]) <= ws[19]
        assert ws[21] / ws[20] == pytest.approx(_cfg().reserve_decay)

    def test_gather_share_split(self):
        scores = {uid: 100.0 - uid for uid in range(30)}
        weights = compute_weights(scores, _cfg(), gather_count=20, reserve_count=10)
        ranked = sorted(scores, key=lambda u: -scores[u])
        gather_sum = sum(weights[u] for u in ranked[:20])
        reserve_sum = sum(weights[u] for u in ranked[20:])
        # reserve was capped below the gather floor, then everything renormalized:
        # gather keeps >= its nominal share and the two still sum to 1.
        assert gather_sum >= _cfg().gather_share
        assert gather_sum + reserve_sum == pytest.approx(1.0)

    def test_tie_determinism_by_uid(self):
        scores = {7: 0.5, 3: 0.5, 11: 0.5, 1: 0.9}
        a = compute_weights(dict(scores), _cfg(), gather_count=2, reserve_count=2)
        b = compute_weights(dict(reversed(list(scores.items()))), _cfg(), gather_count=2, reserve_count=2)
        assert a == b
        # lower uid wins the tie for the last gather slot
        assert a[3] > a[7] > a[11]

    def test_fewer_than_gather_count(self):
        weights = compute_weights({1: 0.5, 2: 0.4}, _cfg(), gather_count=20, reserve_count=10)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights[1] > weights[2]

    def test_nonfinite_scores_excluded(self):
        weights = compute_weights({1: float("nan"), 2: 0.4}, _cfg())
        assert weights == {2: pytest.approx(1.0)}


# --------------------------------------------------------------------------- #
# EvalPools
# --------------------------------------------------------------------------- #


def _manifest() -> RunManifest:
    return RunManifest(
        spec_version=1,
        run_id="test",
        netuid=0,
        network="test",
        config_hash="00" * 32,
        container_digest="sha256:" + "11" * 32,
        mok_commit="8f90b74",
        tk_commit="deadbee",
        attention_backend="cudnn_det",
        start_block=0,
        blocks_per_window=225,
        prf=PRFSpec(run_seed_hex="00" * 32),
        datasets=(
            DatasetManifestRef(
                name="bulk",
                merkle_root="22" * 32,
                num_shards=4,
                shard_bytes=4096 * 2 * 16,  # 16 sequences of 4096 uint16 tokens per shard
                seq_len=4096,
                tokens_total=4 * 16 * 4096,
                tokenizer_hash="33" * 32,
            ),
        ),
        init_checkpoint_hash="44" * 32,
    )


_POOL = [(i // 10, i % 10) for i in range(100)]


def _factory(manifest: RunManifest, run_seed: bytes, uid: int, window: int, phase: PhaseConfig):
    return _POOL


class TestEvalPools:
    def setup_method(self):
        self.manifest = _manifest()
        self.phase = resolve_phase(self.manifest, RunConfig(), 3)
        self.pools = EvalPools(plan_factory=_factory)

    def test_own_pool_golden(self):
        got = self.pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 4, BLOCK_HASH)
        # consensus constant — change requires SPEC_VERSION bump
        assert got == [(4, 7), (7, 5), (0, 0), (1, 9)]

    def test_random_pool_golden(self):
        got = self.pools.random_pool(self.manifest, RUN_SEED, 3, self.phase, 6, BLOCK_HASH)
        # consensus constant — change requires SPEC_VERSION bump
        assert got == [(3, 12), (0, 10), (1, 7), (1, 0), (3, 0), (0, 9)]

    def test_own_pool_deterministic_and_seed_sensitive(self):
        a = self.pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 8, BLOCK_HASH)
        b = self.pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 8, BLOCK_HASH)
        assert a == b
        other_uid = self.pools.own_pool(self.manifest, RUN_SEED, 8, 3, self.phase, 8, BLOCK_HASH)
        other_block = self.pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 8, b"\x02" * 32)
        assert a != other_uid
        assert a != other_block

    def test_own_pool_subset_and_unique(self):
        got = self.pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 32, BLOCK_HASH)
        assert len(got) == 32
        assert len(set(got)) == 32
        assert all(item in _POOL for item in got)

    def test_own_pool_caps_at_pool_size(self):
        got = self.pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 500, BLOCK_HASH)
        assert sorted(got) == sorted(_POOL)

    def test_own_pool_empty_requests(self):
        assert self.pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 0, BLOCK_HASH) == []
        empty = EvalPools(plan_factory=lambda *a: [])
        assert empty.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 4, BLOCK_HASH) == []

    def test_random_pool_bounds_and_uniqueness(self):
        got = self.pools.random_pool(self.manifest, RUN_SEED, 3, self.phase, 40, BLOCK_HASH)
        assert len(got) == 40
        assert len(set(got)) == 40
        for shard, seq in got:
            assert 0 <= shard < 4
            assert 0 <= seq < 16

    def test_random_pool_sentinel_is_uid_independent(self):
        # the random pool never depends on any miner uid — only block hash + tree
        assert RANDOM_POOL_UID == 2**32 - 1
        a = self.pools.random_pool(self.manifest, RUN_SEED, 3, self.phase, 6, BLOCK_HASH)
        b = self.pools.random_pool(self.manifest, RUN_SEED, 99, self.phase, 6, BLOCK_HASH)
        assert a == b  # window is not part of the seed; block_hash is

    def test_plan_with_sequences_attribute(self):
        class Plan:
            sequences = [(0, 0), (0, 1), (1, 0)]

        pools = EvalPools(plan_factory=lambda *a: Plan())
        got = pools.own_pool(self.manifest, RUN_SEED, 7, 3, self.phase, 2, BLOCK_HASH)
        assert len(got) == 2
        assert all(item in Plan.sequences for item in got)

    def test_default_factory_builds_real_window_batch_plan(self):
        # a toy phase small enough for the 4-shard fixture dataset
        phase = PhaseConfig(
            name="bulk",
            data="bulk",
            lr=LRSpec(),
            seq_len=4096,
            tokens_per_rank_microbatch=4096,
            rope_theta=10_000.0,
            grad_accum=1,
            capacity_multiplier=1.0,
            inner_steps=2,
            requires_restart=False,
        )
        pools = EvalPools(world_size=8)  # real WindowBatchPlan.build path
        got = pools.own_pool(self.manifest, RUN_SEED, 7, 3, phase, 5, BLOCK_HASH)
        assert len(got) == 5
        assert len(set(got)) == 5
        # the pool is the miner's all-rank window draw: 1*1*2*8 = 16 sequences
        from mok_core.data.window_dataset import WindowBatchPlan

        plan = WindowBatchPlan.build(
            self.manifest,
            run_seed=RUN_SEED,
            uid=7,
            window=3,
            rank=0,
            world_size=8,
            tokens_per_rank_microbatch=4096,
            grad_accum=1,
            inner_steps=2,
            seq_len=4096,
            dataset="bulk",
        )
        pool = {(int(s), int(q)) for s, q in plan.global_pairs}
        assert set(got) <= pool
        # identical draw on a second validator
        assert got == EvalPools(world_size=8).own_pool(
            self.manifest, RUN_SEED, 7, 3, phase, 5, BLOCK_HASH
        )
