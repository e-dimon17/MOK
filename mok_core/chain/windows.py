"""Pure block <-> window arithmetic and the two-phase-commit upload gate.

Window boundaries are defined entirely by the on-chain manifest
(`start_block`, `blocks_per_window`), so every node computes identical
boundaries from its own subtensor view — no clocks involved. The upload
gate is the only wall-clock rule: an object is in-gate iff its storage
timestamp lies in [boundary_ts, boundary_ts + grace_s).
"""

from __future__ import annotations


def _check_params(start_block: int, blocks_per_window: int) -> None:
    if start_block < 0:
        raise ValueError(f"start_block must be >= 0, got {start_block}")
    if blocks_per_window <= 0:
        raise ValueError(f"blocks_per_window must be > 0, got {blocks_per_window}")


def window_of_block(block: int, start_block: int, blocks_per_window: int) -> int:
    """Window index (>= 0) containing `block`. Raises if `block` predates the run."""
    _check_params(start_block, blocks_per_window)
    if block < start_block:
        raise ValueError(f"block {block} precedes run start_block {start_block}")
    return (block - start_block) // blocks_per_window


def boundary_block(window: int, start_block: int, blocks_per_window: int) -> int:
    """First block of `window` — the block whose arrival ends window `window - 1`."""
    _check_params(start_block, blocks_per_window)
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    return start_block + window * blocks_per_window


def blocks_into_window(block: int, start_block: int, blocks_per_window: int) -> int:
    """Offset of `block` inside its window, in [0, blocks_per_window)."""
    _check_params(start_block, blocks_per_window)
    if block < start_block:
        raise ValueError(f"block {block} precedes run start_block {start_block}")
    return (block - start_block) % blocks_per_window


def gate_deadline_s(boundary_block_ts: float, grace_s: float) -> float:
    """Exclusive upload deadline: boundary timestamp plus the grace period."""
    if grace_s <= 0:
        raise ValueError(f"grace_s must be > 0, got {grace_s}")
    return boundary_block_ts + grace_s


def is_in_gate(object_ts: float, boundary_ts: float, grace_s: float) -> bool:
    """True iff `object_ts` lies in the half-open gate [boundary_ts, boundary_ts + grace_s)."""
    return boundary_ts <= object_ts < gate_deadline_s(boundary_ts, grace_s)
