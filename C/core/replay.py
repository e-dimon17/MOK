"""The bitwise audit executor — replaying a miner's window and signing the verdict.

The subnet's differentiator (playbook step C, audits A1): because a window is
a pure function of (θ_start, uid, window, manifest, run counters), an auditor
that loads θ_start(window) and calls `window_runner.run_training_phase` with
the MINER's uid — the exact function the miner ran — must reproduce the
miner's committed `H(θ_end)` bit for bit. A mismatch is proof of fake or
divergent work; per `slashing.SlashLedger.audit_verdicts`, a slash needs the
same mismatch reproduced by a quorum of distinct auditors.

θ_start materialization is CALLER-PROVIDED: the auditor app wires
`checkpoint.Checkpointer.load_latest` + `checkpoint.catch_up` (or a direct
state-dict load) to bring its replica to θ_start(window) BEFORE calling
`replay`; this module only verifies that precondition against the miner's
on-chain `WindowCommit.state_root` and raises `PreconditionError` when the
replica is anywhere else. After the replay the auditor's replica is unchanged
(`run_training_phase` restores θ_start via `pseudo_grad` and asserts the
round-trip on per-tensor digests), so one materialized θ_start serves every
sampled miner of that window.

`audit_sampler` is the consensus assignment function: Bernoulli(ρ) per active
miner from `philox(audit_seed(run_seed, block_hash, window))`, sampled miners
partitioned round-robin over sorted auditor uids — every honest node computes
the identical assignment (golden-pinned in tests). Reports are signed over the
canonical JSON of their unsigned fields (`sign_report`/`verify_report`) so
`exchange.put_audit_report` can publish them without re-serialization drift.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from typing import Any

import torch

from mok_core.chain.schemas import WindowCommit
from mok_core.config import RunConfig
from mok_core.config.canonical import canonical_hash
from mok_core.config.manifest import RunManifest
from mok_core.data import ShardReader, WindowBatchPlan
from mok_core.determinism.seeding import audit_seed, philox
from mok_core.model import MoKTransformer
from mok_core.telemetry import get_logger

from .phase import resolve_phase
from .window_runner import (
    RunnerComm,
    RunState,
    build_window_plan,
    run_state_at,
    run_training_phase,
    shared_master_root,
)
from .window_state import divergence_report

__all__ = [
    "AuditReport",
    "PreconditionError",
    "ReplayTask",
    "WindowReplayer",
    "audit_sampler",
    "report_message",
    "sign_report",
    "verify_report",
]

log = get_logger("core.replay")

ShardLookup = Callable[[int], ShardReader]
# Given the consensus plan of (miner_uid, window), yield a shard-index -> reader
# lookup; the auditor app wires this to its ShardCache (prefetch inside __enter__).
ShardLookupFactory = Callable[[WindowBatchPlan], AbstractContextManager[ShardLookup]]

SignFn = Callable[[bytes], bytes]
VerifyFn = Callable[[bytes, bytes], bool]


class PreconditionError(RuntimeError):
    """The auditor's replica is not at the committed θ_start — replay refused."""


# --------------------------------------------------------------------------- #
# Task + report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ReplayTask:
    """One sampled (miner, window) audit, keyed to the miner's on-chain commit."""

    miner_uid: int
    window: int
    commit: WindowCommit


@dataclass(frozen=True)
class AuditReport:
    """The signed verdict of one bitwise replay (published via exchange)."""

    miner_uid: int
    window: int
    theta_start_root: str
    committed_theta_end: str
    replayed_theta_end: str
    match: bool
    divergences: list[dict[str, str]]
    wall_time_s: float
    auditor_uid: int
    signature: str = ""

    def to_json(self) -> dict[str, Any]:
        """The canonical wire dict (exchange.put_audit_report's required shape)."""
        return {
            "miner_uid": int(self.miner_uid),
            "window": int(self.window),
            "theta_start_root": self.theta_start_root,
            "committed_theta_end": self.committed_theta_end,
            "replayed_theta_end": self.replayed_theta_end,
            "match": bool(self.match),
            "divergences": [dict(d) for d in self.divergences],
            "wall_time_s": float(self.wall_time_s),
            "auditor_uid": int(self.auditor_uid),
            "signature": self.signature,
        }


def _unsigned_wire(report: AuditReport | Mapping[str, Any]) -> dict[str, Any]:
    wire = report.to_json() if isinstance(report, AuditReport) else dict(report)
    wire.pop("signature", None)
    return wire


def report_message(report: AuditReport | Mapping[str, Any]) -> bytes:
    """The exact bytes an auditor signs: raw 32-byte blake2b-256 canonical hash
    of the report's unsigned fields (certificate_message convention)."""
    return bytes.fromhex(canonical_hash(_unsigned_wire(report)))


def sign_report(report: AuditReport, sign: SignFn) -> AuditReport:
    """Return the report with `signature` set (hex over `report_message`)."""
    return replace(report, signature=sign(report_message(report)).hex())


def verify_report(report: AuditReport | Mapping[str, Any], verify: VerifyFn) -> bool:
    """True iff the report's signature verifies over its unsigned fields.
    Never raises — malformed signatures are simply invalid."""
    sig_hex = report.signature if isinstance(report, AuditReport) else report.get("signature", "")
    try:
        sig = bytes.fromhex(sig_hex)
    except (TypeError, ValueError):
        return False
    if not sig:
        return False
    try:
        return bool(verify(report_message(report), sig))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# The replayer
# --------------------------------------------------------------------------- #


class WindowReplayer:
    """Runs bitwise replays on a replica that the CALLER placed at θ_start.

    `model` must be topology-identical to the audited miner's (same backend
    semantics, same rank/world layout — auditors run the same 8-rank nodes;
    CPU tests use the reference backend at world_size 1). `shard_lookup_factory`
    supplies verified shard readers for the miner's consensus plan.
    """

    def __init__(
        self,
        model: MoKTransformer,
        cfg: RunConfig,
        manifest: RunManifest,
        *,
        comm: RunnerComm,
        shard_lookup_factory: ShardLookupFactory,
        auditor_uid: int = 0,
        rank: int = 0,
        world_size: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")
        self.model = model
        self.cfg = cfg
        self.manifest = manifest
        self.comm = comm
        self.shard_lookup_factory = shard_lookup_factory
        self.auditor_uid = auditor_uid
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)
        self.run_seed = bytes.fromhex(manifest.prf.run_seed_hex)

    # ------------------------------------------------------------------ #

    def _root(self) -> str | None:
        return shared_master_root(
            self.model, rank=self.rank, world_size=self.world_size, comm=self.comm
        )

    def replay(
        self,
        task: ReplayTask,
        *,
        global_state: RunState | None = None,
        expected_digests: Mapping[str, bytes] | None = None,
    ) -> AuditReport:
        """Execute one bitwise replay; the replica ends unchanged at θ_start.

        `global_state` defaults to the consensus `run_state_at` derivation;
        `expected_digests` (the miner's published per-tensor θ_end digests,
        when available) upgrades a mismatch verdict from a bare root
        comparison to a tensor-naming `divergence_report`.
        """
        t0 = time.monotonic()
        if task.commit.window != task.window:
            raise PreconditionError(
                f"commit is for window {task.commit.window}, task says {task.window}"
            )
        start_root = self.comm.broadcast_object(self._root(), 0)
        if start_root != task.commit.state_root:
            raise PreconditionError(
                f"replica state_root {start_root} != committed θ_start root "
                f"{task.commit.state_root} for window {task.window} — run catch_up first"
            )

        phase = resolve_phase(self.manifest, self.cfg, task.window)
        state = (
            run_state_at(self.cfg, self.manifest, task.window, world_size=self.world_size)
            if global_state is None
            else global_state
        )
        plan = build_window_plan(
            self.manifest,
            phase,
            run_seed=self.run_seed,
            uid=task.miner_uid,
            window=task.window,
            rank=self.rank,
            world_size=self.world_size,
        )
        with self.shard_lookup_factory(plan) as shard_lookup:
            # THE SAME function the miner ran (window_runner phases 2-5); no
            # compression state — replay verdicts bind θ_end, not payload bytes.
            artifacts = run_training_phase(
                self.model,
                self.cfg,
                self.manifest,
                phase,
                uid=task.miner_uid,
                window=task.window,
                rank=self.rank,
                world_size=self.world_size,
                comm=self.comm,
                shard_lookup=shard_lookup,
                global_state=state,
                device=self.device,
                plan=plan,
                run_seed=self.run_seed,
            )
        # run_training_phase asserted the bitwise θ_start restore on per-tensor
        # digests; the replica is exactly where the caller left it.

        replayed = self.comm.broadcast_object(artifacts.theta_end_root, 0)
        match = replayed == task.commit.theta_end_hash
        divergences: list[dict[str, str]] = []
        if not match:
            if expected_digests is not None:
                divergences = [
                    r.to_json()
                    for r in divergence_report(dict(expected_digests), artifacts.theta_end_digests)
                ]
            else:
                divergences = [
                    {
                        "name": "<state_root>",
                        "expected": task.commit.theta_end_hash,
                        "actual": str(replayed),
                    }
                ]
            log.warning(
                "replay mismatch",
                miner_uid=task.miner_uid,
                window=task.window,
                divergent=len(divergences),
            )
        return AuditReport(
            miner_uid=task.miner_uid,
            window=task.window,
            theta_start_root=str(start_root),
            committed_theta_end=task.commit.theta_end_hash,
            replayed_theta_end=str(replayed),
            match=match,
            divergences=divergences,
            wall_time_s=time.monotonic() - t0,
            auditor_uid=self.auditor_uid,
        )


# --------------------------------------------------------------------------- #
# Consensus audit assignment
# --------------------------------------------------------------------------- #


def audit_sampler(
    run_seed: bytes,
    block_hash: bytes,
    window: int,
    active_uids: Iterable[int],
    rho: float,
    auditor_uids: Iterable[int],
) -> dict[int, list[tuple[int, int]]]:
    """Deterministic audit assignment: auditor uid -> [(miner_uid, window), ...].

    Consensus function (golden-pinned, SPEC_VERSION-bound): one Philox stream
    seeded by `audit_seed(run_seed, block_hash, window)` draws one uniform per
    active miner IN SORTED-UID ORDER; miners with draw < ρ are sampled, then
    dealt round-robin over sorted auditor uids. `block_hash` is the boundary
    block of the audited window's successor — unpredictable while mining.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    auditors = sorted(set(auditor_uids))
    if not auditors:
        raise ValueError("audit_sampler needs at least one auditor uid")
    miners = sorted(set(active_uids))
    draws = philox(audit_seed(run_seed, block_hash, window)).random(len(miners))
    assignments: dict[int, list[tuple[int, int]]] = {a: [] for a in auditors}
    sampled = (uid for uid, d in zip(miners, draws, strict=True) if float(d) < rho)
    for i, uid in enumerate(sampled):
        assignments[auditors[i % len(auditors)]].append((uid, window))
    return assignments
