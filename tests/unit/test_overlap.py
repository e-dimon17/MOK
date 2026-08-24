"""subnet/core/overlap.py — copy detection, offender attribution, severity ladder."""

from __future__ import annotations

import pytest
import torch

from subnet.core.overlap import (
    OverlapPair,
    determine_offender,
    index_overlap_report,
    severity,
)


def _chunks(rows: list[list[int]]) -> torch.Tensor:
    """(C, k) LongTensor of per-chunk top-k indices."""
    return torch.tensor(rows, dtype=torch.int64)


def _uniform(base: int, chunks: int = 8, k: int = 4) -> torch.Tensor:
    return _chunks([[base + i for i in range(k)]] * chunks)


class TestIndexOverlapReport:
    def test_copied_pair_flagged_at_full_overlap(self):
        peers = {1: {"w": _uniform(0)}, 2: {"w": _uniform(0).clone()}}
        report = index_overlap_report(peers, threshold=0.9)
        assert report.pairs_checked == 1
        assert report.pairs == [OverlapPair(uid_a=1, uid_b=2, overlap=pytest.approx(1.0))]
        assert report.mean_overlap == pytest.approx(1.0)

    def test_independent_sets_not_flagged(self):
        peers = {1: {"w": _uniform(0)}, 2: {"w": _uniform(100)}}
        report = index_overlap_report(peers, threshold=0.4)
        assert report.pairs == []
        assert report.pairs_checked == 1
        assert report.mean_overlap == pytest.approx(0.0)

    def test_partial_overlap_value(self):
        # per chunk: {0,1,2,3} vs {0,1,8,9} -> 2/4 = 0.5
        a = _chunks([[0, 1, 2, 3]] * 4)
        b = _chunks([[0, 1, 8, 9]] * 4)
        report = index_overlap_report({1: {"w": a}, 2: {"w": b}}, threshold=0.4)
        assert report.pairs == [OverlapPair(uid_a=1, uid_b=2, overlap=pytest.approx(0.5))]

    def test_threshold_is_inclusive(self):
        a = _chunks([[0, 1, 2, 3]] * 4)
        b = _chunks([[0, 1, 8, 9]] * 4)
        assert index_overlap_report({1: {"w": a}, 2: {"w": b}}, threshold=0.5).pairs
        assert not index_overlap_report({1: {"w": a}, 2: {"w": b}}, threshold=0.51).pairs

    def test_size_weighted_mean_across_params(self):
        # p_small: identical (overlap 1.0, weight 4); p_big: disjoint (0.0, weight 16)
        peers = {
            1: {"p_small": _chunks([[0, 1, 2, 3]]), "p_big": _uniform(0, chunks=4)},
            2: {"p_small": _chunks([[0, 1, 2, 3]]), "p_big": _uniform(100, chunks=4)},
        }
        report = index_overlap_report(peers, threshold=0.1)
        assert report.pairs == [OverlapPair(uid_a=1, uid_b=2, overlap=pytest.approx(4 / 20))]

    def test_all_pairs_and_only_copiers_flagged(self):
        peers = {
            3: {"w": _uniform(0)},
            1: {"w": _uniform(0).clone()},   # copies uid 3
            2: {"w": _uniform(100)},         # independent
        }
        report = index_overlap_report(peers, threshold=0.9)
        assert report.pairs_checked == 3
        assert report.pairs == [OverlapPair(uid_a=1, uid_b=3, overlap=pytest.approx(1.0))]

    def test_mismatched_shapes_and_missing_params_skipped(self):
        peers = {
            1: {"w": _uniform(0), "only_1": _uniform(0)},
            2: {"w": _uniform(0, chunks=2), "only_2": _uniform(0)},
        }
        report = index_overlap_report(peers, threshold=0.4)
        assert report.pairs_checked == 0  # nothing comparable
        assert report.pairs == []

    def test_multidim_chunk_layout(self):
        # (2, 3, k) layout — leading dims are all chunk dims
        a = torch.arange(24, dtype=torch.int64).reshape(2, 3, 4)
        report = index_overlap_report({1: {"w": a}, 2: {"w": a.clone()}}, threshold=0.9)
        assert report.pairs[0].overlap == pytest.approx(1.0)

    def test_fewer_than_two_peers(self):
        assert index_overlap_report({}, threshold=0.4).pairs == []
        assert index_overlap_report({1: {"w": _uniform(0)}}, threshold=0.4).pairs == []


class TestDetermineOffender:
    def test_later_uploader_blamed(self):
        pair = OverlapPair(uid_a=1, uid_b=2, overlap=0.8)
        assert determine_offender(pair, {1: 100.0, 2: 200.0}) == 2
        assert determine_offender(pair, {1: 300.0, 2: 200.0}) == 1

    def test_tie_blames_higher_uid(self):
        pair = OverlapPair(uid_a=1, uid_b=2, overlap=0.8)
        assert determine_offender(pair, {1: 100.0, 2: 100.0}) == 2


class TestSeverity:
    @pytest.mark.parametrize(
        ("overlap", "level", "multiplier", "naughty"),
        [
            (0.0, "none", 1.0, False),
            (0.39, "none", 1.0, False),
            (0.40, "high", 0.5, False),
            (0.49, "high", 0.5, False),
            (0.50, "max", 0.0, False),
            (0.59, "max", 0.0, False),
            (0.60, "mega", 0.0, True),
            (1.00, "mega", 0.0, True),
        ],
    )
    def test_ladder(self, overlap: float, level: str, multiplier: float, naughty: bool):
        sev = severity(overlap)
        assert (sev.level, sev.multiplier, sev.naughty) == (level, multiplier, naughty)

    def test_bool_means_any_sanction(self):
        assert not severity(0.1)
        assert severity(0.4)

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_out_of_range_rejected(self, bad: float):
        with pytest.raises(ValueError):
            severity(bad)


def test_pair_overlap_batching_is_exact(monkeypatch):
    """Row-batched kernel must equal the single-shot result bit for bit."""
    import subnet.core.overlap as ov

    g = torch.Generator().manual_seed(7)
    a = torch.stack([torch.randperm(4096, generator=g)[:64] for _ in range(1000)])
    b = torch.stack([torch.randperm(4096, generator=g)[:64] for _ in range(1000)])
    full = ov._pair_overlap(a, b)
    monkeypatch.setattr(ov, "_PAIR_BATCH_CHUNKS", 7)   # force many uneven batches
    assert ov._pair_overlap(a, b) == pytest.approx(full, abs=0.0)
