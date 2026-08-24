"""Window state hashing — the `state_root` consensus value and audit diagnostics.

`state_root` is the hex blake2b-256 over the ordered (name, tensor) master
weights that miners commit on-chain every window; a replay audit passes iff
the recomputed root matches bitwise. All byte rules live in
`mok_core.determinism.hashing` (single source of truth) — this module only
delegates and adds the rank-parallel combiner used on multi-GPU nodes, where
each rank hashes the tensors it owns and rank 0 combines the digests.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping

import torch

from mok_core.determinism import (
    DivergenceRecord,
    first_divergence,
    hash_named_tensors,
    per_tensor_digests,
)

_DIGEST_SIZE = 32

# A rank's contribution to the combined root: (tensor name, tensor digest) pairs.
RankDigests = list[tuple[str, bytes]]
# Injected collective: takes this rank's pairs, returns every rank's pairs on
# rank 0 (in rank order) and None on all other ranks.
GatherFn = Callable[[RankDigests], list[RankDigests] | None]


def state_root(named_params: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hex blake2b-256 state root — delegates to mok_core.determinism.hash_named_tensors."""
    return hash_named_tensors(named_params)


def collect_digests(named: Iterable[tuple[str, torch.Tensor]]) -> dict[str, bytes]:
    """Per-tensor digests for divergence reports and A2 spot-checks."""
    return per_tensor_digests(named)


def divergence_report(
    expected_digests: Mapping[str, bytes],
    actual_digests: Mapping[str, bytes],
    limit: int = 16,
) -> list[DivergenceRecord]:
    """First `limit` per-tensor mismatches in sorted-name order (audit mismatch evidence)."""
    return first_divergence(expected_digests, actual_digests, limit=limit)


def _combine_digest_pairs(pairs: Iterable[tuple[str, bytes]]) -> str:
    """Combine (name, digest) pairs into the root.

    Framing MUST stay byte-identical to mok_core.determinism.hashing.hash_named_tensors
    (len(name) le32 ‖ name utf-8 ‖ digest, sorted by name); the equality is pinned by
    test_window_state.py, so any drift fails CI before it can fork consensus.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for name, digest in sorted(pairs, key=lambda kv: kv[0]):
        h.update(len(name).to_bytes(4, "little"))
        h.update(name.encode("utf-8"))
        h.update(digest)
    return h.hexdigest()


def rank_parallel_state_root(
    named: Iterable[tuple[str, torch.Tensor]],
    gather: GatherFn,
) -> str | None:
    """state_root computed cooperatively: each rank hashes only the tensors it owns.

    `named` is this rank's locally-owned (name, tensor) pairs — every master
    tensor must be owned by exactly one rank (replicated params are hashed by
    a single designated owner). `gather` is the injected collective; pure
    logic here, so a fake gather makes this CPU-testable.

    Returns the combined root on rank 0 (where `gather` yields all ranks'
    lists) and None on other ranks. Raises ValueError if two ranks claim the
    same tensor name.
    """
    local: RankDigests = [(name, digest) for name, digest in per_tensor_digests(named).items()]
    gathered = gather(local)
    if gathered is None:
        return None
    merged: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for rank_pairs in gathered:
        for name, digest in rank_pairs:
            if name in seen:
                raise ValueError(f"tensor {name!r} hashed by more than one rank — ownership must be unique")
            seen.add(name)
            merged.append((name, digest))
    return _combine_digest_pairs(merged)
