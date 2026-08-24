"""Attestation challenges — block-hash-seeded toy training runs.

A joining miner proves it runs real Tier-A hardware (8×B300, NVLink, the
blessed container) by re-running a small but kernel-shaped training window —
the 4-layer toy model from ``subnet/configs/toy4L.yaml`` — from a chain-published
seed and returning the resulting ``state_root`` within a wall-clock deadline.
The owner/auditor precomputes the expected root once per challenge on a
Tier-A node; verification is a string compare plus a clock check
(``fleet.attestation.verify``).

Consensus derivation (SPEC_VERSION-bound, golden-pinned in
``tests/unit/test_fleet_challenge.py``), in the style of
``mok_core.determinism.seeding`` domain digests::

    digest       = blake2b-256(block_hash ‖ b"attest.v1")
    challenge_id = digest[:8].hex()                     # 16 lowercase hex chars
    seed         = le64(digest[8:16]) & (2**63 - 1)

``block_hash`` is the raw 32-byte hash of the issuing block — unpredictable
before that block exists, identical for every observer afterwards, so any
validator can re-derive (and re-judge) any challenge from chain data alone.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import torch
from pydantic import model_validator

import subnet
from mok_core.config import RunConfig, deep_merge, load_yaml
from mok_core.config.loader import _interpolate  # env-default interpolation, pinned by loader tests
from mok_core.config.schemas import FrozenModel

__all__ = [
    "ATTEST_DOMAIN",
    "BASE_CONFIG_PATH",
    "DEFAULT_DEADLINE_S",
    "DEFAULT_INNER_STEPS",
    "TOY4L_CONFIG_PATH",
    "AttestationChallenge",
    "challenge_run_config",
    "derive_expected",
    "make_challenge",
    "toy4l_overlay",
]

ATTEST_DOMAIN = b"attest.v1"  # consensus constant — change requires SPEC_VERSION bump

#: Wall-clock budget for the full respond path (init + 20 steps + hash + upload).
#: Sized so that only a real 8×SM103 NVLink node makes it (see verify.py).
DEFAULT_DEADLINE_S = 420.0

#: The reference run is a 20-inner-step toy window.
DEFAULT_INNER_STEPS = 20

_MASK63 = (1 << 63) - 1
_HEX16 = re.compile(r"^[0-9a-f]{16}$")

_SUBNET_CONFIGS = Path(subnet.__file__).resolve().parent / "configs"
BASE_CONFIG_PATH = _SUBNET_CONFIGS / "base.yaml"
TOY4L_CONFIG_PATH = _SUBNET_CONFIGS / "toy4L.yaml"


def toy4l_overlay() -> dict[str, Any]:
    """The ``subnet/configs/toy4L.yaml`` overlay dict, verbatim (the challenge model shape)."""
    return load_yaml(TOY4L_CONFIG_PATH)


class AttestationChallenge(FrozenModel):
    """One hardware-attestation task. Fully self-describing: a responder needs
    nothing but this object (and the blessed container) to produce its root."""

    challenge_id: str          # 16 lowercase hex chars — digest[:8] of the derivation
    seed: int                  # 63-bit init/data seed — digest[8:16] of the derivation
    model_overlay: dict[str, Any]  # config overlay over subnet/configs/base.yaml (toy4L shape)
    inner_steps: int = DEFAULT_INNER_STEPS
    deadline_s: float
    issued_block: int

    @model_validator(mode="after")
    def _check(self) -> AttestationChallenge:
        if not _HEX16.match(self.challenge_id):
            raise ValueError(f"challenge_id must be 16 lowercase hex chars, got {self.challenge_id!r}")
        if not 0 <= self.seed <= _MASK63:
            raise ValueError(f"seed must be a 63-bit unsigned int, got {self.seed}")
        if self.inner_steps <= 0:
            raise ValueError(f"inner_steps must be positive, got {self.inner_steps}")
        if self.deadline_s <= 0:
            raise ValueError(f"deadline_s must be positive, got {self.deadline_s}")
        if self.issued_block < 0:
            raise ValueError(f"issued_block must be >= 0, got {self.issued_block}")
        return self


def make_challenge(
    block_hash: bytes,
    issued_block: int,
    *,
    deadline_s: float = DEFAULT_DEADLINE_S,
    inner_steps: int = DEFAULT_INNER_STEPS,
) -> AttestationChallenge:
    """Derive the challenge for ``block_hash`` (module-docstring derivation).

    The model overlay is the toy4L file content verbatim, so the challenge
    object pins the exact shape even if the local configs drift later.
    """
    if len(block_hash) != 32:
        raise ValueError(f"block_hash must be 32 bytes, got {len(block_hash)}")
    digest = hashlib.blake2b(block_hash + ATTEST_DOMAIN, digest_size=32).digest()
    return AttestationChallenge(
        challenge_id=digest[:8].hex(),
        seed=int.from_bytes(digest[8:16], "little") & _MASK63,
        model_overlay=toy4l_overlay(),
        inner_steps=inner_steps,
        deadline_s=deadline_s,
        issued_block=issued_block,
    )


def challenge_run_config(challenge: AttestationChallenge) -> RunConfig:
    """The challenge's full RunConfig: base.yaml + the challenge overlay
    (production loader semantics: ``deep_merge`` + env interpolation), with
    ``window.inner_steps`` pinned to ``challenge.inner_steps`` on top."""
    merged = deep_merge(load_yaml(BASE_CONFIG_PATH), challenge.model_overlay)
    merged = deep_merge(merged, {"window": {"inner_steps": challenge.inner_steps}})
    return RunConfig.model_validate(_interpolate(merged))


def derive_expected(
    challenge: AttestationChallenge,
    *,
    device: str | torch.device,
    backend: str = "reference",
    comm: Any | None = None,
) -> str:
    """The expected ``state_root`` of ``challenge`` — the verifier's precompute.

    Runs THE SAME deterministic path as attestation responders
    (``fleet.attestation.reference_step.run_reference``), so expected and actual
    can only differ if hardware/software genuinely diverges. On a Tier-A node
    launch this under torchrun -n 8 with ``backend='mok'``; on CPU it runs
    single-process with ``backend='reference'`` (tests do exactly that).
    The root is broadcast to every rank, so any rank may read it.
    """
    from .reference_step import run_reference  # noqa: PLC0415 — avoids an import cycle

    return run_reference(challenge, backend=backend, device=device, comm=comm).state_root
