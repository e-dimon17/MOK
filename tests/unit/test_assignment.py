"""Assignment PRF: golden vectors, distinctness, salt behavior, shape math."""

from __future__ import annotations

import numpy as np
import pytest

from mok_core.config.manifest import DatasetManifestRef
from mok_core.data.assignment import (
    effective_run_seed,
    sample_order,
    sequences_per_window,
    shard_ids,
    tokens_per_shard,
)

RUN_SEED = bytes(range(32))


def _dataset(num_shards: int = 1024, seq_len: int = 4096, seqs_per_shard: int = 64) -> DatasetManifestRef:
    return DatasetManifestRef(
        name="bulk",
        merkle_root="00" * 32,
        num_shards=num_shards,
        shard_bytes=2 * seq_len * seqs_per_shard,
        seq_len=seq_len,
        tokens_total=num_shards * seq_len * seqs_per_shard,
        tokenizer_hash="ab" * 32,
    )


def test_tokens_per_shard() -> None:
    assert tokens_per_shard(_dataset()) == 4096 * 64
    bad = _dataset().model_copy(update={"shard_bytes": 1000})
    with pytest.raises(ValueError, match="multiple"):
        tokens_per_shard(bad)


def test_shard_ids_golden() -> None:
    ids = shard_ids(_dataset(), RUN_SEED, 7, 42, 262144 * 5 + 1)
    # consensus constant — change requires SPEC_VERSION bump
    assert ids == [915, 629, 732, 976, 217, 633]


def test_shard_ids_salted_golden() -> None:
    ids = shard_ids(_dataset(), RUN_SEED, 7, 42, 262144 * 5 + 1, reseed_salt=b"rollback-1")
    # consensus constant — change requires SPEC_VERSION bump
    assert ids == [68, 464, 881, 269, 227, 155]


def test_shard_ids_distinct_and_in_range() -> None:
    ds = _dataset(num_shards=32)
    ids = shard_ids(ds, RUN_SEED, 11, 3, tokens_per_shard(ds) * 30)
    assert len(ids) == 30
    assert len(set(ids)) == 30
    assert all(0 <= i < 32 for i in ids)


def test_shard_ids_ceil_cover() -> None:
    ds = _dataset()
    per = tokens_per_shard(ds)
    assert len(shard_ids(ds, RUN_SEED, 0, 0, per)) == 1
    assert len(shard_ids(ds, RUN_SEED, 0, 0, per + 1)) == 2


def test_shard_ids_prefix_stability() -> None:
    """Drawing more tokens extends the shard list without reshuffling the prefix."""
    ds = _dataset()
    per = tokens_per_shard(ds)
    small = shard_ids(ds, RUN_SEED, 7, 42, per * 3)
    big = shard_ids(ds, RUN_SEED, 7, 42, per * 6)
    assert big[:3] == small


def test_shard_ids_errors() -> None:
    ds = _dataset(num_shards=4)
    with pytest.raises(ValueError, match="positive"):
        shard_ids(ds, RUN_SEED, 0, 0, 0)
    with pytest.raises(ValueError, match="only 4"):
        shard_ids(ds, RUN_SEED, 0, 0, tokens_per_shard(ds) * 5)


def test_shard_ids_vary_by_uid_and_window() -> None:
    ds = _dataset()
    base = shard_ids(ds, RUN_SEED, 7, 42, 262144 * 5)
    assert shard_ids(ds, RUN_SEED, 8, 42, 262144 * 5) != base
    assert shard_ids(ds, RUN_SEED, 7, 43, 262144 * 5) != base


def test_sample_order_golden() -> None:
    order = sample_order(RUN_SEED, 7, 42, 8, 1000)
    # consensus constant — change requires SPEC_VERSION bump
    assert order.tolist() == [923, 383, 744, 385, 670, 797, 778, 68]
    assert order.dtype == np.int64


def test_sample_order_without_replacement() -> None:
    order = sample_order(RUN_SEED, 1, 2, 500, 500)
    assert sorted(order.tolist()) == list(range(500))


def test_sample_order_independent_of_shard_draw() -> None:
    """order domain != assign domain: same (seed, uid, window) gives unrelated streams."""
    ds = _dataset(num_shards=1000)
    ids = shard_ids(ds, RUN_SEED, 7, 42, tokens_per_shard(ds) * 8)
    order = sample_order(RUN_SEED, 7, 42, 8, 1000)
    assert ids != order.tolist()


def test_sample_order_salt_and_errors() -> None:
    base = sample_order(RUN_SEED, 7, 42, 8, 1000)
    salted = sample_order(RUN_SEED, 7, 42, 8, 1000, reseed_salt=b"x")
    assert base.tolist() != salted.tolist()
    with pytest.raises(ValueError, match="positive"):
        sample_order(RUN_SEED, 0, 0, 0, 10)
    with pytest.raises(ValueError, match="available"):
        sample_order(RUN_SEED, 0, 0, 11, 10)


def test_effective_run_seed() -> None:
    assert effective_run_seed(RUN_SEED, b"") == RUN_SEED
    salted = effective_run_seed(RUN_SEED, b"salt")
    assert salted != RUN_SEED
    assert len(salted) == 32
    # consensus constant — change requires SPEC_VERSION bump
    assert salted.hex() == "42cd7d5e56afcb4fbf92f063b8e2cd090c62d0c5ab9df7d81745af2619a5d913"


def test_sequences_per_window_production_shape() -> None:
    # RunConfig defaults: 8192-token microbatch, accum 8, 500 steps, 8 ranks, seq 4096
    n = sequences_per_window(
        tokens_per_rank_microbatch=8192, grad_accum=8, inner_steps=500, ranks=8, seq_len=4096
    )
    assert n == 2 * 8 * 500 * 8 == 64_000


def test_sequences_per_window_errors() -> None:
    with pytest.raises(ValueError, match="multiple"):
        sequences_per_window(tokens_per_rank_microbatch=100, grad_accum=1, inner_steps=1, ranks=1, seq_len=16)
    with pytest.raises(ValueError, match="positive"):
        sequences_per_window(tokens_per_rank_microbatch=64, grad_accum=0, inner_steps=1, ranks=1, seq_len=16)
