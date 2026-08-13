"""Window/block arithmetic and upload-gate rules (mok_core/chain/windows.py)."""

from __future__ import annotations

import pytest

from mok_core.chain.windows import (
    blocks_into_window,
    boundary_block,
    gate_deadline_s,
    is_in_gate,
    window_of_block,
)

START = 1000
BPW = 225


class TestWindowOfBlock:
    def test_first_window(self) -> None:
        assert window_of_block(START, START, BPW) == 0
        assert window_of_block(START + BPW - 1, START, BPW) == 0

    def test_window_increments_at_boundary(self) -> None:
        assert window_of_block(START + BPW, START, BPW) == 1
        assert window_of_block(START + 2 * BPW, START, BPW) == 2
        assert window_of_block(START + 7 * BPW + 13, START, BPW) == 7

    def test_block_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            window_of_block(START - 1, START, BPW)

    def test_bad_params_raise(self) -> None:
        with pytest.raises(ValueError, match="blocks_per_window"):
            window_of_block(START, START, 0)
        with pytest.raises(ValueError, match="blocks_per_window"):
            window_of_block(START, START, -5)
        with pytest.raises(ValueError, match="start_block"):
            window_of_block(0, -1, BPW)


class TestBoundaryBlock:
    def test_boundaries(self) -> None:
        assert boundary_block(0, START, BPW) == START
        assert boundary_block(1, START, BPW) == START + BPW
        assert boundary_block(10, START, BPW) == START + 10 * BPW

    def test_negative_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window"):
            boundary_block(-1, START, BPW)

    def test_round_trip_with_window_of_block(self) -> None:
        for w in (0, 1, 5, 999):
            b = boundary_block(w, START, BPW)
            assert window_of_block(b, START, BPW) == w
            assert window_of_block(b + BPW - 1, START, BPW) == w
            assert window_of_block(b + BPW, START, BPW) == w + 1


class TestBlocksIntoWindow:
    def test_offsets(self) -> None:
        assert blocks_into_window(START, START, BPW) == 0
        assert blocks_into_window(START + 1, START, BPW) == 1
        assert blocks_into_window(START + BPW - 1, START, BPW) == BPW - 1
        assert blocks_into_window(START + BPW, START, BPW) == 0
        assert blocks_into_window(START + 3 * BPW + 17, START, BPW) == 17

    def test_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="precedes"):
            blocks_into_window(START - 1, START, BPW)


class TestGate:
    def test_deadline(self) -> None:
        assert gate_deadline_s(1000.0, 90.0) == 1090.0

    def test_deadline_requires_positive_grace(self) -> None:
        with pytest.raises(ValueError, match="grace_s"):
            gate_deadline_s(1000.0, 0.0)
        with pytest.raises(ValueError, match="grace_s"):
            gate_deadline_s(1000.0, -1.0)

    def test_gate_is_half_open(self) -> None:
        boundary = 1_722_945_600.0
        grace = 90.0
        assert is_in_gate(boundary, boundary, grace)                    # inclusive start
        assert is_in_gate(boundary + 89.999, boundary, grace)
        assert not is_in_gate(boundary + 90.0, boundary, grace)         # exclusive deadline
        assert not is_in_gate(boundary - 0.001, boundary, grace)        # early upload rejected
        assert not is_in_gate(boundary + 1e6, boundary, grace)
