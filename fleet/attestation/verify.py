"""Attestation verdicts — root equality plus the hardware wall-clock gate.

WHY A DEADLINE IS A HARDWARE CHECK: the challenge window is shaped for the MoK
kernel (T = 8192 tokens/rank ≥ 512 and %256, hidden %256, EP over NVLink), and
its 20 steps complete in well under a minute on a real 8×SM103 NVLink node.
Anything else — consumer GPUs, PCIe-only peering, CPU emulation, or splitting
the EP group across a network — is orders of magnitude slower, so a correct
root arriving inside ``deadline_s`` (default 420 s, covering container start,
model init, the run, hashing and upload) is evidence of the real machine.
Timestamps are the verifier's own observations (challenge publication time and
response arrival time), never self-reported by the miner.

Random re-attestation during the run (``schedule_reattestation``) is a
consensus draw — every validator derives the same sample from chain data:

    seed = le64(blake2b-256(run_seed ‖ b"reattest.v1" ‖ block_hash)[:8]) & (2**63-1)
    keep uid  ⇔  philox(seed).random()[i] < rate   (uids in ascending order)

Golden-pinned in ``tests/unit/test_fleet_verify.py`` (SPEC_VERSION-bound).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from mok_core.config.schemas import FrozenModel
from mok_core.determinism import philox

from .challenge import (
    DEFAULT_DEADLINE_S,
    DEFAULT_INNER_STEPS,
    AttestationChallenge,
    make_challenge,
)
from .reference_step import AttestationResponse

__all__ = [
    "REATTEST_DOMAIN",
    "AttestationVerdict",
    "Verifier",
    "judge",
    "schedule_reattestation",
]

REATTEST_DOMAIN = b"reattest.v1"  # consensus constant — change requires SPEC_VERSION bump

_MASK63 = (1 << 63) - 1


class AttestationVerdict(FrozenModel):
    ok: bool
    reason: str = ""


def judge(
    challenge: AttestationChallenge,
    response: AttestationResponse,
    expected_root: str,
    *,
    received_ts: float,
    issued_ts: float,
) -> AttestationVerdict:
    """Judge one response: id match, deadline, root equality. Never raises."""
    problems: list[str] = []
    if response.challenge_id != challenge.challenge_id:
        problems.append(
            f"challenge_id mismatch: response {response.challenge_id!r} "
            f"!= challenge {challenge.challenge_id!r}"
        )
    elapsed = received_ts - issued_ts
    if elapsed < 0:
        problems.append(f"timestamps out of order: received {received_ts} < issued {issued_ts}")
    elif elapsed > challenge.deadline_s:
        problems.append(
            f"deadline exceeded: {elapsed:.1f}s > {challenge.deadline_s:.1f}s "
            "(only real SM103 NVLink hardware finishes in time — see module docstring)"
        )
    if response.state_root.lower() != expected_root.lower():
        problems.append(
            f"state_root mismatch: got {response.state_root} expected {expected_root}"
        )
    if problems:
        return AttestationVerdict(ok=False, reason="; ".join(problems))
    return AttestationVerdict(ok=True)


def schedule_reattestation(
    run_seed: bytes, block_hash: bytes, active_uids: Iterable[int], rate: float
) -> list[int]:
    """The deterministic re-attestation sample for one block (module docstring
    derivation). Ascending uids; every validator computes the same list."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    h = hashlib.blake2b(digest_size=32)
    h.update(run_seed)
    h.update(REATTEST_DOMAIN)
    h.update(block_hash)
    seed = int.from_bytes(h.digest()[:8], "little") & _MASK63
    uids = sorted(set(active_uids))
    if not uids:
        return []
    draws = philox(seed).random(len(uids))
    return [uid for uid, draw in zip(uids, draws, strict=True) if draw < rate]


class Verifier:
    """Owner/validator-side attestation driver: issue → precompute → judge."""

    def __init__(
        self,
        *,
        deadline_s: float = DEFAULT_DEADLINE_S,
        inner_steps: int = DEFAULT_INNER_STEPS,
    ) -> None:
        self.deadline_s = float(deadline_s)
        self.inner_steps = int(inner_steps)

    def issue(self, chain: Any) -> AttestationChallenge:
        """A fresh challenge from the chain head (current block's hash)."""
        block = chain.current_block()
        return make_challenge(
            chain.block_hash(block),
            block,
            deadline_s=self.deadline_s,
            inner_steps=self.inner_steps,
        )

    def judge(
        self,
        challenge: AttestationChallenge,
        response: AttestationResponse,
        expected_root: str,
        *,
        received_ts: float,
        issued_ts: float,
    ) -> AttestationVerdict:
        return judge(
            challenge, response, expected_root, received_ts=received_ts, issued_ts=issued_ts
        )

    def schedule_reattestation(
        self, run_seed: bytes, block_hash: bytes, active_uids: Iterable[int], rate: float
    ) -> list[int]:
        return schedule_reattestation(run_seed, block_hash, active_uids, rate)
