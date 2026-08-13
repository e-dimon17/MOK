"""Tests for C/core/window_state.py — delegation to mok_core hashing + rank-parallel root."""

from __future__ import annotations

import pytest
import torch

from C.core.window_state import (
    collect_digests,
    divergence_report,
    rank_parallel_state_root,
    state_root,
)
from mok_core.determinism import hash_named_tensors, per_tensor_digests


def _named() -> list[tuple[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(3)
    return [
        ("model.wte", torch.randn(4, 8, generator=g)),
        ("model.router", torch.randn(8, generator=g).to(torch.bfloat16)),
        ("model.bias", torch.arange(6, dtype=torch.int64)),
    ]


def test_state_root_delegates_to_mok_core_hashing():
    named = _named()
    assert state_root(named) == hash_named_tensors(named)


def test_collect_digests_delegates():
    named = _named()
    assert collect_digests(named) == per_tensor_digests(named)


def test_divergence_report_flags_only_changed_tensors():
    named = _named()
    expected = collect_digests(named)
    tampered = [(n, t + 1 if n == "model.bias" else t) for n, t in named]
    actual = collect_digests(tampered)
    records = divergence_report(expected, actual)
    assert [r.name for r in records] == ["model.bias"]
    assert records[0].expected != records[0].actual
    assert records[0].expected and records[0].actual


def test_divergence_report_respects_limit_and_missing():
    named = _named()
    expected = collect_digests(named)
    actual = dict(collect_digests([(n, t + 1) for n, t in named]))
    del actual["model.bias"]  # missing tensor also counts as divergence
    assert len(divergence_report(expected, actual, limit=2)) == 2
    full = divergence_report(expected, actual, limit=16)
    assert len(full) == 3
    missing = next(r for r in full if r.name == "model.bias")
    assert missing.actual == ""


# --------------------------------------------------------------------------- #
# rank-parallel root
# --------------------------------------------------------------------------- #


def _simulate_ranks(named_by_rank):
    """Run rank_parallel_state_root on every simulated rank with a fake gather."""

    def make_gather(rank: int):
        def gather(local):
            if rank != 0:
                return None
            out = []
            for r, named in enumerate(named_by_rank):
                out.append(local if r == 0 else list(per_tensor_digests(named).items()))
            return out

        return gather

    return [rank_parallel_state_root(named_by_rank[r], make_gather(r)) for r in range(len(named_by_rank))]


def test_rank_parallel_equals_direct_state_root():
    named = _named()
    named_by_rank = [named[0:1], named[1:2], named[2:3]]
    roots = _simulate_ranks(named_by_rank)
    assert roots[0] == state_root(named)
    assert roots[1] is None and roots[2] is None


def test_rank_parallel_order_independent():
    named = _named()
    shuffled = [named[2:3], named[0:1], named[1:2]]  # ownership moved between ranks
    assert _simulate_ranks(shuffled)[0] == state_root(named)


def test_rank_parallel_rejects_duplicate_ownership():
    named = _named()
    dup = [named[0:1], named[0:1]]  # both ranks claim model.wte
    with pytest.raises(ValueError, match="more than one rank"):
        _simulate_ranks(dup)


def test_rank_parallel_empty_rank_ok():
    named = _named()
    roots = _simulate_ranks([named, [], []])
    assert roots[0] == state_root(named)
