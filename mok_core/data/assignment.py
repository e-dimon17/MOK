"""Data-assignment PRF — the consensus function mapping (uid, window) to data.

Any validator reproduces any miner's exact shard set and sequence order from
the on-chain manifest alone; the audit path replays the identical draw. Built
on mok_core.determinism.seeding (blake2b domains + Philox permutations), so
draws are platform-stable. All outputs are golden-vector pinned: any change
here is a SPEC_VERSION bump.

Window PRF and without-replacement draw, built on the blake2b domain PRF
and counter-based Philox for platform-stable streams.
"""

from __future__ import annotations

import hashlib

import numpy as np

from mok_core.config.manifest import DatasetManifestRef
from mok_core.determinism.seeding import philox, window_seed

_MASK63 = (1 << 63) - 1
_TOKEN_BYTES = 2  # shards are raw little-endian uint16 tokens


def effective_run_seed(run_seed: bytes, reseed_salt: bytes) -> bytes:
    """Fold a rollback reseed salt into the run seed (identity when salt is empty).

    Consensus rule: blake2b-256(run_seed ‖ "reseed.v1" ‖ salt).
    """
    if not reseed_salt:
        return run_seed
    h = hashlib.blake2b(digest_size=32)
    h.update(run_seed)
    h.update(b"reseed.v1")
    h.update(reseed_salt)
    return h.digest()


def _order_seed(run_seed: bytes, uid: int, window: int) -> int:
    """Sequence-order seed: same construction as seeding.window_seed but in the
    'order.v1' domain, so the shard draw and the order draw are independent."""
    h = hashlib.blake2b(digest_size=32)
    h.update(run_seed)
    h.update(b"order.v1")
    h.update(int(uid).to_bytes(8, "little", signed=False))
    h.update(int(window).to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little") & _MASK63


def tokens_per_shard(dataset: DatasetManifestRef) -> int:
    """Tokens held by one full shard of `dataset` (uint16 tokens, all shards full)."""
    if dataset.shard_bytes <= 0 or dataset.shard_bytes % (_TOKEN_BYTES * dataset.seq_len) != 0:
        raise ValueError(
            f"shard_bytes ({dataset.shard_bytes}) must be a positive multiple of "
            f"{_TOKEN_BYTES} * seq_len ({dataset.seq_len})"
        )
    return dataset.shard_bytes // _TOKEN_BYTES


def shard_ids(
    dataset: DatasetManifestRef,
    run_seed: bytes,
    uid: int,
    window: int,
    tokens_needed: int,
    *,
    reseed_salt: bytes = b"",
) -> list[int]:
    """Distinct shard indices covering `tokens_needed`, in PRF draw order.

    Philox permutation of [0, num_shards) keyed by window_seed(run_seed', uid,
    window), truncated to the ceil-cover count.
    """
    if tokens_needed <= 0:
        raise ValueError(f"tokens_needed must be positive, got {tokens_needed}")
    per_shard = tokens_per_shard(dataset)
    count = -(-tokens_needed // per_shard)  # ceil
    if count > dataset.num_shards:
        raise ValueError(
            f"window needs {count} shards ({tokens_needed} tokens) but dataset "
            f"{dataset.name!r} has only {dataset.num_shards}"
        )
    seed = window_seed(effective_run_seed(run_seed, reseed_salt), uid, window)
    perm = philox(seed).permutation(dataset.num_shards)
    return [int(i) for i in perm[:count]]


def sample_order(
    run_seed: bytes,
    uid: int,
    window: int,
    n: int,
    num_sequences_available: int,
    *,
    reseed_salt: bytes = b"",
) -> np.ndarray:
    """`n` distinct sequence indices in [0, num_sequences_available) — the
    within-window sample order. Philox permutation draw (without replacement)
    in the 'order.v1' domain; int64 array."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n > num_sequences_available:
        raise ValueError(f"window needs {n} sequences but only {num_sequences_available} are available")
    seed = _order_seed(effective_run_seed(run_seed, reseed_salt), uid, window)
    perm = philox(seed).permutation(num_sequences_available)
    return perm[:n].astype(np.int64)


def sequences_per_window(
    *,
    tokens_per_rank_microbatch: int,
    grad_accum: int,
    inner_steps: int,
    ranks: int,
    seq_len: int,
) -> int:
    """Packed sequences one window consumes across all ranks of one miner."""
    if min(tokens_per_rank_microbatch, grad_accum, inner_steps, ranks, seq_len) <= 0:
        raise ValueError("all window-shape parameters must be positive")
    if tokens_per_rank_microbatch % seq_len != 0:
        raise ValueError(
            f"tokens_per_rank_microbatch ({tokens_per_rank_microbatch}) must be an "
            f"exact multiple of seq_len ({seq_len})"
        )
    return (tokens_per_rank_microbatch // seq_len) * grad_accum * inner_steps * ranks
