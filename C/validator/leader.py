"""Leader-validator duties: certificate, aggregator mirror, debug slices,
checkpoints, and the spike→rollback pipeline.

The leader is the highest-stake validator (ties break to the lowest uid —
`resolve_leader_uid`); every validator computes the same answer, so exactly
one node performs these duties per window while the rest verify. Deputy
fallback: if the leader misses its certificate, miners time out into catch-up
and the next window's leader resolution simply picks the live highest-stake
validator again — liveness degrades one window, never forks.

Rollback trust model: votes travel as on-chain `VoteCommit`s (authenticity =
the extrinsic's own signature); the vote payload hash binds the canonical
`RollbackVote` body. The spike probe set is CANONICAL and window-independent:
`EvalPools.random_pool` seeded with the fixed `PROBE_BLOCK_HASH` sentinel, so
the trailing-median baseline compares like against like.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from C.core.certificate import WindowCertificate, build_certificate
from C.core.checkpoint import Checkpointer, CheckpointMeta
from C.core.exchange import get_certificate, put_aggregator_object, put_certificate, put_debug_slices
from C.core.rollback import RollbackDecision, RollbackStateMachine, SpikeDetector
from C.core.window_runner import RunState
from C.miner.bootstrap import NodeContext, resolve_leader_uid
from mok_core.chain.schemas import VoteCommit
from mok_core.config import canonical_hash
from mok_core.telemetry import get_logger

__all__ = ["PROBE_BLOCK_HASH", "CommitView", "LeaderDuties"]

log = get_logger("app.validator.leader")

#: Fixed sentinel block hash for the canonical probe pool (window-independent).
PROBE_BLOCK_HASH = bytes(32)


@dataclass(frozen=True)
class CommitView:
    """`certificate.CommitLike` view of one miner's window commit."""

    uid: int
    payload_hash: str
    in_gate: bool
    valid: bool


class LeaderDuties:
    """The per-window leader work, bound to one validator's NodeContext."""

    def __init__(
        self,
        ctx: NodeContext,
        checkpointer: Checkpointer,
        spike: SpikeDetector,
        rollback: RollbackStateMachine,
    ) -> None:
        self.ctx = ctx
        self.checkpointer = checkpointer
        self.spike = spike
        self.rollback = rollback

    def is_leader(self) -> bool:
        return resolve_leader_uid(self.ctx.chain, fallback=self.ctx.uid) == self.ctx.uid

    # ------------------------------------------------------------------ #
    # Gate-close duties (before anyone can gather)
    # ------------------------------------------------------------------ #

    async def publish_certificate(
        self,
        window: int,
        commit_views: Mapping[int, CommitView],
        payload_bytes: Mapping[int, bytes],
        scores: Mapping[int, float],
        theta_start_root: str,
    ) -> WindowCertificate:
        """Build, sign and publish the certificate + the aggregator mirror.

        IMMUTABLE once published: if this window already has a certificate in
        the leader bucket (e.g. this leader crashed after publishing and is
        reprocessing the window), ADOPT it. Rebuilding from live chain slots
        would fork consensus — window commits are transient (a miner's next
        commit overwrites its slot), so a replayed window reads empty views and
        a rebuilt certificate would clobber the one peers already applied."""
        existing = await self._existing_certificate(window)
        if existing is not None:
            log.info(
                "certificate already published — adopting",
                window=window,
                included=list(existing.included_uids),
            )
            return existing
        cfg = self.ctx.cfg
        cert = build_certificate(
            window,
            commit_views,
            dict(scores),
            gather_count=cfg.window.gather_peer_count,
            reserve_count=cfg.window.reserve_peer_count,
            theta_start_root=theta_start_root,
            leader_uid=self.ctx.uid,
            sign=self.ctx.signer.sign,
        )
        await put_certificate(self.ctx.storage, cert)
        mirror = {uid: payload_bytes[uid] for uid in cert.included_uids if uid in payload_bytes}
        await put_aggregator_object(self.ctx.storage, window, mirror)
        log.info("certificate published", window=window, included=list(cert.included_uids))
        return cert

    async def _existing_certificate(self, window: int) -> WindowCertificate | None:
        from mok_core.storage import ObjectMissingError, StorageError  # noqa: PLC0415

        try:
            return await get_certificate(
                self.ctx.storage, self.ctx.own_bucket, window, max_bytes=1 << 20
            )
        except ObjectMissingError:
            return None
        except (StorageError, ValueError) as e:  # unreadable = treat as absent, rebuild
            log.warning("existing certificate unreadable — rebuilding", window=window, error=str(e))
            return None

    # ------------------------------------------------------------------ #
    # Post-outer-step duties
    # ------------------------------------------------------------------ #

    async def publish_debug_slices(
        self, window: int, master: Mapping[str, torch.Tensor]
    ) -> None:
        await put_debug_slices(self.ctx.storage, window, self.ctx.uid, dict(master))

    async def maybe_checkpoint(
        self,
        window: int,
        master: Mapping[str, torch.Tensor],
        outer_state: Mapping[str, torch.Tensor],
        state_root: str,
        run_state: RunState,
    ) -> bool:
        """DCP save + upload on the cadence windows; meta binds θ_start(window+1)."""
        if window % self.ctx.cfg.window.checkpoint_every_windows != 0:
            return False
        meta = CheckpointMeta(
            window=window,
            global_step=run_state.global_step,
            tokens_consumed=run_state.tokens_consumed,
            state_root=state_root,
            manifest_hash=self.ctx.manifest.manifest_hash(),
            spec_version=self.ctx.manifest.spec_version,
        )
        await self.checkpointer.save(window, master, dict(outer_state), meta)
        log.info("leader checkpoint saved", window=window, root=state_root)
        return True

    # ------------------------------------------------------------------ #
    # Spike detection → rollback voting
    # ------------------------------------------------------------------ #

    def observe_probe_loss(self, window: int, probe_loss: float) -> RollbackDecision | None:
        """Feed the canonical probe loss; drive the rollback state machine.

        On a fresh alert: fixes the rewind target at the newest local
        checkpoint window and commits a rollback `VoteCommit` (our yes-vote).
        Every window: ingest peers' on-chain votes stake-weighted, tick the
        timeout, and surface an activated `RollbackDecision` (the caller
        rewinds/relaunches; the manifest amendment is the owner's duty).
        """
        ctx = self.ctx
        alerted = self.spike.observe(window, probe_loss)
        if alerted:
            checkpoints = self.checkpointer.local_windows()
            targets = [w for w in checkpoints if w < window]
            if not targets:
                log.warning("loss spike but no checkpoint to rewind to", window=window)
            elif self.rollback.on_alert(window, targets[-1]):
                target = targets[-1]
                vote_body = {
                    "voter_uid": ctx.uid,
                    "stake": float(ctx.chain.stakes().get(ctx.uid, 0.0)),
                    "target_window": target,
                    "window_cast": window,
                }
                ctx.chain.commit_vote(
                    VoteCommit(kind="rollback", target=target, payload_hash=canonical_hash(vote_body))
                )
                log.warning("loss spike — rollback vote cast", window=window, target=target)

        if self.rollback.target_window is not None:
            stakes = ctx.chain.stakes()
            total = sum(stakes.values())
            votes = ctx.chain.get_votes(kind="rollback", target=self.rollback.target_window)
            from C.core.rollback import RollbackVote  # noqa: PLC0415 — local to the voting path

            for uid, vc in sorted(votes.items()):
                self.rollback.add_vote(
                    RollbackVote(
                        voter_uid=uid,
                        stake=float(stakes.get(uid, 0.0)),
                        target_window=vc.target,
                        window_cast=window,
                        sig="",  # authenticity is the on-chain commitment itself
                    ),
                    total_stake=total,
                )
        self.rollback.tick(window)
        decision = self.rollback.maybe_activate(window)
        if decision is not None:
            log.warning(
                "rollback activated",
                target=decision.target_window,
                void=(decision.void.first_window, decision.void.last_window),
            )
        return decision
