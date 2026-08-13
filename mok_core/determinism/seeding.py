"""Seed derivation — consensus constants.

Every stochastic choice in the protocol derives from blake2b over
(run_seed ‖ domain ‖ little-endian ints). Golden-vector tests pin outputs;
changing any of this is a SPEC_VERSION bump.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

_MASK63 = (1 << 63) - 1


def _digest(run_seed: bytes, domain: bytes, *parts: int) -> bytes:
    h = hashlib.blake2b(digest_size=32)
    h.update(run_seed)
    h.update(domain)
    for p in parts:
        h.update(int(p).to_bytes(8, "little", signed=False))
    return h.digest()


def window_seed(run_seed: bytes, uid: int, window: int) -> int:
    """Seed for miner `uid`'s data assignment in `window` (dataset PRF domain)."""
    return int.from_bytes(_digest(run_seed, b"assign.v1", uid, window)[:8], "little") & _MASK63


def window_rank_seed(run_seed: bytes, uid: int, window: int, rank: int) -> int:
    """Seed for anything rank-local inside a window (there is no dropout; this
    exists for future-proofing and for the attestation toy run)."""
    return (
        int.from_bytes(_digest(run_seed, b"rank.v1", uid, window, rank)[:8], "little") & _MASK63
    )


def eval_seed(run_seed: bytes, block_hash: bytes, uid: int) -> int:
    """Validator eval-slice seed: unpredictable to miners during their commit
    window (block hash is post-commit), identical across validators."""
    h = hashlib.blake2b(digest_size=32)
    h.update(run_seed)
    h.update(b"eval.v1")
    h.update(block_hash)
    h.update(int(uid).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little") & _MASK63


def audit_seed(run_seed: bytes, block_hash: bytes, window: int) -> int:
    h = hashlib.blake2b(digest_size=32)
    h.update(run_seed)
    h.update(b"audit.v1")
    h.update(block_hash)
    h.update(int(window).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little") & _MASK63


def philox(seed: int) -> np.random.Generator:
    """Counter-based generator — platform-stable sample orders."""
    return np.random.Generator(np.random.Philox(key=seed))


def seed_everything(seed: int) -> None:
    """Process-wide seeding (torch, numpy legacy, python). Data order never
    depends on this — it flows from the PRF — but model init and any library
    internals must still be pinned."""
    import torch  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed & 0xFFFF_FFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
