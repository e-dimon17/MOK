"""WindowBatchPlan: bit-identical rebuilds, rank striping, receipts, token IO."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest
from mok_core.data.shards import ShardReader
from mok_core.data.window_dataset import WindowBatchPlan

RUN_SEED = bytes(range(32))
SEQ_LEN = 16
SEQS_PER_SHARD = 32
NUM_SHARDS = 64


def _manifest(reseed_salt_hex: str = "") -> RunManifest:
    dataset = DatasetManifestRef(
        name="bulk",
        merkle_root="00" * 32,
        num_shards=NUM_SHARDS,
        shard_bytes=2 * SEQ_LEN * SEQS_PER_SHARD,
        seq_len=SEQ_LEN,
        tokens_total=NUM_SHARDS * SEQ_LEN * SEQS_PER_SHARD,
        tokenizer_hash="ab" * 32,
    )
    return RunManifest(
        spec_version=1,
        run_id="testrun",
        netuid=11,
        network="test",
        config_hash="11" * 32,
        container_digest="sha256:" + "22" * 32,
        mok_commit="deadbeef",
        tk_commit="cafebabe",
        attention_backend="cudnn_det",
        start_block=100,
        blocks_per_window=225,
        prf=PRFSpec(run_seed_hex=RUN_SEED.hex(), reseed_salt_hex=reseed_salt_hex),
        datasets=(dataset,),
        init_checkpoint_hash="33" * 32,
    )


def _build(rank: int = 0, world_size: int = 2, **overrides: object) -> WindowBatchPlan:
    kwargs: dict = {
        "run_seed": RUN_SEED,
        "uid": 3,
        "window": 5,
        "rank": rank,
        "world_size": world_size,
        "tokens_per_rank_microbatch": 64,
        "grad_accum": 2,
        "inner_steps": 3,
        "seq_len": SEQ_LEN,
        "dataset": "bulk",
    }
    kwargs.update(overrides)
    manifest = kwargs.pop("manifest", _manifest())
    return WindowBatchPlan.build(manifest, **kwargs)


def test_two_builds_byte_identical() -> None:
    a, b = _build(), _build()
    assert a.shard_ids == b.shard_ids
    assert a.global_pairs.tobytes() == b.global_pairs.tobytes()
    assert a.schedule.tobytes() == b.schedule.tobytes()
    assert a.sample_digest() == b.sample_digest()


def test_goldens() -> None:
    plan = _build()
    # consensus constants — change requires SPEC_VERSION bump
    assert plan.shard_ids == (61, 0)
    assert plan.sample_digest() == "449915d475cfca69af4b767ad3017d87ede92a304341f19725dc8cb9f34d14d2"
    assert plan.schedule[0, 0].tolist() == [[61, 20], [61, 0], [0, 12], [0, 16]]


def test_shapes_and_counts() -> None:
    plan = _build()
    assert plan.total_sequences == 48  # 4 seq/mb * 2 accum * 3 steps * 2 ranks
    assert plan.sequences_per_rank == 24
    assert plan.seqs_per_microbatch == 4
    assert plan.schedule.shape == (3, 2, 4, 2)
    assert plan.global_pairs.shape == (48, 2)
    assert plan.global_pairs.dtype == np.int64


def test_rank_striping_partitions_global_order() -> None:
    r0, r1 = _build(rank=0), _build(rank=1)
    interleaved = np.empty_like(r0.global_pairs)
    interleaved[0::2] = r0.schedule.reshape(-1, 2)
    interleaved[1::2] = r1.schedule.reshape(-1, 2)
    np.testing.assert_array_equal(interleaved, r0.global_pairs)
    pairs0 = {tuple(p) for p in r0.schedule.reshape(-1, 2).tolist()}
    pairs1 = {tuple(p) for p in r1.schedule.reshape(-1, 2).tolist()}
    assert not pairs0 & pairs1
    assert len(pairs0 | pairs1) == 48  # without replacement across ranks


def test_sample_digest_is_rank_invariant() -> None:
    assert _build(rank=0).sample_digest() == _build(rank=1).sample_digest()


def test_digest_varies_by_uid_window_salt() -> None:
    base = _build()
    assert _build(uid=4).sample_digest() != base.sample_digest()
    assert _build(window=6).sample_digest() != base.sample_digest()
    salted = _build(manifest=_manifest(reseed_salt_hex="ab" * 8))
    assert salted.sample_digest() != base.sample_digest()


def test_all_seq_indices_within_shard() -> None:
    plan = _build()
    assert set(np.unique(plan.global_pairs[:, 0])) <= set(plan.shard_ids)
    assert plan.global_pairs[:, 1].min() >= 0
    assert plan.global_pairs[:, 1].max() < SEQS_PER_SHARD


def test_schedule_is_readonly() -> None:
    plan = _build()
    with pytest.raises(ValueError, match="read-only"):
        plan.schedule[0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        plan.global_pairs[0, 0] = 1


def test_build_validation() -> None:
    with pytest.raises(ValueError, match="multiple"):
        _build(tokens_per_rank_microbatch=65)
    with pytest.raises(ValueError, match="seq_len"):
        _build(seq_len=32, tokens_per_rank_microbatch=64)
    with pytest.raises(ValueError, match="rank"):
        _build(rank=2, world_size=2)
    with pytest.raises(KeyError):
        _build(dataset="anneal")


# --------------------------------------------------------------------------- #
# token IO
# --------------------------------------------------------------------------- #


def _shard_tokens(shard_idx: int) -> np.ndarray:
    base = np.arange(SEQS_PER_SHARD * SEQ_LEN, dtype=np.int64)
    return ((base + shard_idx * 1000) % 65536).astype("<u2").reshape(SEQS_PER_SHARD, SEQ_LEN)


def test_microbatch_tokens(tmp_path: Path) -> None:
    plan = _build()
    readers: dict[int, ShardReader] = {}
    for shard_idx in plan.shard_ids:
        path = tmp_path / f"shard-{shard_idx}.bin"
        path.write_bytes(_shard_tokens(shard_idx).tobytes())
        readers[shard_idx] = ShardReader(path, SEQ_LEN)
    try:
        seen: list[tuple[int, int]] = []
        for step in range(plan.inner_steps):
            for accum in range(plan.grad_accum):
                out = plan.microbatch_tokens(step, accum, readers.__getitem__)
                assert isinstance(out, torch.Tensor)
                assert out.dtype == torch.int64
                assert out.shape == (64,)
                expected = np.concatenate(
                    [_shard_tokens(s)[q] for s, q in plan.microbatch_pairs(step, accum)]
                ).astype(np.int64)
                np.testing.assert_array_equal(out.numpy(), expected)
                seen += [tuple(p) for p in plan.microbatch_pairs(step, accum).tolist()]
        assert len(seen) == plan.sequences_per_rank
        assert len(set(seen)) == len(seen)
    finally:
        for reader in readers.values():
            reader.close()


def test_microbatch_bounds() -> None:
    plan = _build()
    with pytest.raises(IndexError):
        plan.microbatch_pairs(3, 0)
    with pytest.raises(IndexError):
        plan.microbatch_pairs(0, 2)
