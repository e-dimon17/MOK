"""WindowBatchPlan — the fully precomputed per-rank data schedule of one window.

Built once per (uid, window) from the manifest and the PRF; after `build` no
randomness remains: microbatch m of inner step s reads exactly the pinned
(shard_idx, seq_idx) pairs. Two builds with identical inputs are byte-identical
(tested), which is what makes a window replayable bit-for-bit.

Rank striping takes the global order's arr[rank::world_size] slice; the
sample-digest receipt hashes the sorted sample ids (here (shard_idx,
seq_idx) pairs).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from mok_core.config.manifest import RunManifest

from .assignment import sample_order, sequences_per_window, shard_ids
from .shards import ShardReader

_DIGEST_SIZE = 32


@dataclass(frozen=True, eq=False)
class WindowBatchPlan:
    """Immutable schedule: `schedule[s, m, j] = (shard_idx, seq_idx)` for this rank."""

    dataset_name: str
    uid: int
    window: int
    rank: int
    world_size: int
    seq_len: int
    tokens_per_rank_microbatch: int
    grad_accum: int
    inner_steps: int
    shard_ids: tuple[int, ...]  # PRF draw order; indices into the dataset tree
    global_pairs: np.ndarray  # [total_sequences, 2] int64 — all ranks, PRF order
    schedule: np.ndarray  # [inner_steps, grad_accum, seqs_per_microbatch, 2] int64

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def build(
        cls,
        manifest: RunManifest,
        *,
        run_seed: bytes,
        uid: int,
        window: int,
        rank: int,
        world_size: int,
        tokens_per_rank_microbatch: int,
        grad_accum: int,
        inner_steps: int,
        seq_len: int,
        dataset: str,
    ) -> WindowBatchPlan:
        """Resolve the PRF into a concrete schedule. The manifest supplies the
        dataset tree and the active reseed salt; phase params arrive resolved
        (C/core/phase.py owns the merge)."""
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")
        ds = manifest.dataset(dataset)
        if seq_len != ds.seq_len:
            raise ValueError(f"phase seq_len {seq_len} != dataset {dataset!r} seq_len {ds.seq_len}")
        total = sequences_per_window(
            tokens_per_rank_microbatch=tokens_per_rank_microbatch,
            grad_accum=grad_accum,
            inner_steps=inner_steps,
            ranks=world_size,
            seq_len=seq_len,
        )
        salt = bytes.fromhex(manifest.prf.reseed_salt_hex)
        shards = shard_ids(ds, run_seed, uid, window, total * seq_len, reseed_salt=salt)
        seqs_per_shard = ds.shard_bytes // (2 * seq_len)
        pool = len(shards) * seqs_per_shard
        order = sample_order(run_seed, uid, window, total, pool, reseed_salt=salt)

        shard_arr = np.asarray(shards, dtype=np.int64)
        global_pairs = np.stack([shard_arr[order // seqs_per_shard], order % seqs_per_shard], axis=1)
        local = global_pairs[rank::world_size]  # deterministic rank striping
        seqs_per_microbatch = tokens_per_rank_microbatch // seq_len
        schedule = local.reshape(inner_steps, grad_accum, seqs_per_microbatch, 2)
        global_pairs.setflags(write=False)
        schedule.setflags(write=False)
        return cls(
            dataset_name=dataset,
            uid=uid,
            window=window,
            rank=rank,
            world_size=world_size,
            seq_len=seq_len,
            tokens_per_rank_microbatch=tokens_per_rank_microbatch,
            grad_accum=grad_accum,
            inner_steps=inner_steps,
            shard_ids=tuple(shards),
            global_pairs=global_pairs,
            schedule=schedule,
        )

    # ------------------------------------------------------------------ #
    # consumption
    # ------------------------------------------------------------------ #

    @property
    def total_sequences(self) -> int:
        """Sequences the whole window consumes across all ranks of this miner."""
        return int(self.global_pairs.shape[0])

    @property
    def sequences_per_rank(self) -> int:
        return self.total_sequences // self.world_size

    @property
    def seqs_per_microbatch(self) -> int:
        return self.tokens_per_rank_microbatch // self.seq_len

    def microbatch_pairs(self, step: int, accum_idx: int) -> np.ndarray:
        """The (shard_idx, seq_idx) rows feeding microbatch `accum_idx` of `step`."""
        if not 0 <= step < self.inner_steps:
            raise IndexError(f"step {step} out of range [0, {self.inner_steps})")
        if not 0 <= accum_idx < self.grad_accum:
            raise IndexError(f"accum_idx {accum_idx} out of range [0, {self.grad_accum})")
        return self.schedule[step, accum_idx]

    def microbatch_tokens(
        self,
        step: int,
        accum_idx: int,
        shard_lookup: Callable[[int], ShardReader],
    ) -> torch.LongTensor:
        """Concatenated whole sequences for one microbatch: [tokens_per_rank_microbatch]."""
        parts = [
            shard_lookup(int(shard_idx)).sequence(int(seq_idx))
            for shard_idx, seq_idx in self.microbatch_pairs(step, accum_idx)
        ]
        tokens = np.concatenate(parts).astype(np.int64, copy=False)
        out = torch.from_numpy(tokens)
        assert out.shape == (self.tokens_per_rank_microbatch,)
        return out  # type: ignore[return-value]  # LongTensor by construction

    # ------------------------------------------------------------------ #
    # receipt
    # ------------------------------------------------------------------ #

    def sample_digest(self) -> str:
        """Miner's data receipt: blake2b-256 hex over the sorted global
        (shard_idx, seq_idx) list — identical on every rank of this (uid, window).

        Wire format (consensus constant): "samples.v1" ‖ n(le64) ‖ for each pair
        in lexicographic order: shard_idx(le64) ‖ seq_idx(le64).
        """
        pairs = self.global_pairs
        ordered = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        h.update(b"samples.v1")
        h.update(len(ordered).to_bytes(8, "little"))
        h.update(np.ascontiguousarray(ordered, dtype="<u8").tobytes())
        return h.hexdigest()
