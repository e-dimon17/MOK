"""The validator application: lockstep replica, trust pipeline, chain weights.

A validator never trains. It maintains the reference-backend replica
(`ep_size` forced to 1 — all experts local) in bitwise lockstep by replaying
every window's certified outer step via `checkpoint.catch_up`, and around that
core it runs the whole trust pipeline for each window `w` once `w`'s upload
gate has closed (i.e. the validator trails the chain head by one window):

  (a) read WindowCommits + gate timestamps (`exchange.gate_check`) → SlashLedger
      events (missing/received/invalid);
  (b) leader only: publish the window certificate + aggregator mirror FIRST so
      miners can gather (`C.validator.leader`);
  (c) evaluate payloads at θ_start(w) — loss deltas on deterministic pools →
      gradient scores, BinaryEMA, OpenSkill (`C.validator.evaluator`);
  (d) top-k index-overlap copy detection → ledger sanctions;
  (e) apply w's outer step bitwise (catch_up) — the lockstep core;
  (f) sync scores from miner debug slices vs own post-step params;
  (g) leader only: debug slices, checkpoint cadence, canonical probe loss →
      SpikeDetector → rollback voting;
  (h) every `scoring.windows_per_weights` windows: `compute_weights` →
      `chain.set_weights`;
  (i) ingest audit reports for window `w - audit.report_deadline_windows`
      (2-of-3 quorum → score zero + naughty span);
  (j) persist `ValidatorState` (every stateful component) to JSON.

State persistence: BinaryEMA/OpenSkillBook/SlashLedger expose state_dict();
SpikeDetector's baseline is rebuilt by replaying the recorded probe losses;
the RollbackStateMachine's public window fields plus its vote map are stored
directly (it is a pure state machine — reconstruction is attribute-exact).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from C.core.checkpoint import CatchUpError, Checkpointer, sparse_pairs_from_compressed
from C.core.exchange import ExchangeError, gate_check, get_debug_slices
from C.core.overlap import determine_offender, index_overlap_report, severity
from C.core.payload import PayloadError, WindowPayload, deserialize, validate_structure
from C.core.phase import resolve_phase
from C.core.rollback import RollbackState, RollbackStateMachine, SpikeDetector
from C.core.scoring import BinaryEMA, OpenSkillBook, final_score, sync_score
from C.core.slashing import SlashLedger
from C.core.window_runner import DENSE_SUFFIX, run_state_at
from C.miner.bootstrap import (
    INIT_SEED,
    NodeContext,
    build_compressor,
    build_outer_step,
    catch_up_replica,
    load_master_state,
)
from mok_core.chain.schemas import WindowCommit
from mok_core.chain.windows import boundary_block
from mok_core.config.schemas import BucketCreds
from mok_core.determinism import hash_bytes, hash_named_tensors
from mok_core.model import build_reference_model, evaluate_sequences
from mok_core.storage import StorageError, keys
from mok_core.telemetry import get_logger

from .audit_ingest import ingest_window_audits
from .evaluator import EvalRecord, WindowEvaluator
from .leader import PROBE_BLOCK_HASH, CommitView, LeaderDuties
from .weights import submit_weights

__all__ = ["PAYLOAD_VERSION", "ValidatorApp", "ValidatorState"]

log = get_logger("app.validator")

#: The payload-key version slug this protocol generation reads/writes.
PAYLOAD_VERSION = 1

_STATE_SPEC = 1


class ValidatorState:
    """JSON persistence of every stateful trust component (atomic file writes)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, app: ValidatorApp) -> None:
        sm = app.rollback
        state = {
            "spec": _STATE_SPEC,
            "window": app.window,
            "windows_since_weights": app.windows_since_weights,
            "ema": app.ema.state_dict(),
            "book": app.book.state_dict(),
            "ledger": app.ledger.state_dict(),
            "final_scores": {str(uid): float(s) for uid, s in app.final_scores.items()},
            "probe_losses": list(app.probe_losses),
            "rollback": {
                "state": sm.state.value,
                "alert_window": sm.alert_window,
                "target_window": sm.target_window,
                "activation_window": sm.activation_window,
                "votes": {str(uid): float(stake) for uid, stake in sm._votes.items()},
            },
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def load(self, app: ValidatorApp) -> bool:
        if not self.path.is_file():
            return False
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("spec") != _STATE_SPEC:
            log.warning("validator state spec mismatch — starting fresh", found=state.get("spec"))
            return False
        app.window = int(state["window"])
        app.windows_since_weights = int(state["windows_since_weights"])
        app.ema.load_state_dict(state["ema"])
        app.book.load_state_dict(state["book"])
        app.ledger.load_state_dict(state["ledger"])
        app.final_scores = {int(u): float(s) for u, s in state["final_scores"].items()}
        app.probe_losses = [float(v) for v in state["probe_losses"]]
        for loss in app.probe_losses:
            app.spike.observe(0, loss)  # rebuild the trailing baseline
        rb = state["rollback"]
        sm = app.rollback
        sm.state = RollbackState(rb["state"])
        sm.alert_window = rb["alert_window"]
        sm.target_window = rb["target_window"]
        sm.activation_window = rb["activation_window"]
        sm._votes = {int(u): float(s) for u, s in rb["votes"].items()}  # noqa: SLF001
        return True


class ValidatorApp:
    """One validator process (single rank — the reference replica is EP-free)."""

    def __init__(
        self,
        ctx: NodeContext,
        *,
        max_windows: int | None = None,
        on_window: Callable[[int], None] | None = None,
        catchup_retries: int = 5,
        catchup_retry_s: float = 5.0,
        poll_s: float = 2.0,
    ) -> None:
        self.ctx = ctx
        self.max_windows = max_windows
        self.on_window = on_window
        self.catchup_retries = int(catchup_retries)
        self.catchup_retry_s = float(catchup_retry_s)
        self.poll_s = float(poll_s)

        cfg = ctx.cfg
        self.ema = BinaryEMA(
            cfg.scoring.binary_ema_alpha,
            cfg.scoring.binary_ema_threshold,
            cfg.scoring.binary_warmup_windows,
        )
        self.book = OpenSkillBook(cfg.scoring.openskill_beta, cfg.scoring.openskill_tau)
        self.ledger = SlashLedger(cfg.audit)
        self.spike = SpikeDetector(cfg.rollback.spike_threshold_nats, cfg.rollback.spike_baseline_windows)
        self.rollback = RollbackStateMachine(cfg.rollback, ctx.run_seed)
        self.checkpointer = Checkpointer(ctx.storage, ctx.state_dir / "checkpoints")
        self.leader = LeaderDuties(ctx, self.checkpointer, self.spike, self.rollback)
        self.evaluator = WindowEvaluator(ctx)
        self.state = ValidatorState(ctx.state_dir / "validator_state.json")

        self.model: Any = None
        self.outer_step: Any = None
        self.compressor: Any = None
        self.window = 0
        self.windows_since_weights = 0
        self.final_scores: dict[int, float] = {}
        self.probe_losses: list[float] = []
        self.last_eval: dict[int, EvalRecord] = {}
        self.last_sync: dict[int, float] = {}
        self.completed_windows = 0
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, self.stop)

    async def run(self) -> int:
        ctx = self.ctx
        self._install_signal_handlers()
        self.model = build_reference_model(ctx.cfg.model, INIT_SEED, device=ctx.device)
        self.outer_step = build_outer_step(self.model, ctx.cfg)
        self.compressor = build_compressor(self.model, ctx.cfg)
        restored = self.state.load(self)
        if restored:
            loaded = await self.checkpointer.load_latest()
            if loaded is not None:
                model_state, outer_state, meta = loaded
                load_master_state(self.model, model_state)
                self.outer_step.load_state_dict(outer_state)
                if meta.window + 1 > self.window:
                    self.window = meta.window + 1
                log.info("validator replica restored", window=meta.window, root=meta.state_root)
            else:
                log.warning("state file but no checkpoint — replica at θ_init, catching up from 0")
                self.window = 0
        log.info("validator running", start_window=self.window, restored=restored)

        while True:
            if self._stop.is_set():
                self.state.save(self)
                return 0
            head = await asyncio.to_thread(ctx.chain.current_window, ctx.manifest)
            target = head - 1  # the newest window whose gate can be closed
            gate_close = ctx.clock.boundary_ts(self.window + 1) + ctx.cfg.window.upload_grace_s
            if self.window > target or ctx.clock.now() < gate_close:
                self._log_waiting(head, gate_close)
                await asyncio.sleep(self.poll_s)
                continue
            if target - self.window >= 3:
                # far behind — catch up silently, then resume scoring at the head
                await self._catch_up_retrying(from_window=self.window - 1, to_window=target - 1)
                self.window = target
                continue
            await self.process_window(self.window)
            if self.on_window is not None:
                self.on_window(self.window)
            self.window += 1
            self.completed_windows += 1
            if self.max_windows is not None and self.completed_windows >= self.max_windows:
                self.state.save(self)
                return 0

    # ------------------------------------------------------------------ #
    # One window
    # ------------------------------------------------------------------ #

    _last_wait_log: float = 0.0

    def _log_waiting(self, head: int, gate_close: float) -> None:
        """One 'waiting' line per minute so an idle validator is visibly alive."""
        now = self.ctx.clock.now()
        if now - self._last_wait_log < 60.0:
            return
        self._last_wait_log = now
        log.info(
            "waiting for gate close",
            next_window=self.window,
            chain_head_window=head,
            gate_closes_in_s=round(max(0.0, gate_close - now), 0),
        )

    async def process_window(self, window: int) -> None:
        ctx = self.ctx
        t_window = time.monotonic()
        cfg = ctx.cfg
        if ctx.manifest.is_void(window):
            log.info("void window skipped", window=window)
            return
        phase = resolve_phase(ctx.manifest, cfg, window)
        theta_start_root = hash_named_tensors(self.model.iter_master_params())
        buckets = await asyncio.to_thread(ctx.peer_buckets)

        # (a) commits + gate timestamps → ledger events
        commits: dict[int, WindowCommit] = await asyncio.to_thread(
            ctx.chain.get_window_commits, window
        )
        views, payload_bytes, payloads, upload_ts = await self._gate_and_fetch(
            window, commits, buckets, theta_start_root
        )
        log.info(
            "window commits",
            window=window,
            committed_uids=sorted(commits),
            in_gate=sorted(u for u, v in views.items() if v.in_gate),
            valid=sorted(u for u, v in views.items() if v.valid),
            theta_start=theta_start_root,
        )

        # (b) leader duties before anyone can gather
        is_leader = self.leader.is_leader()
        if is_leader:
            await self.leader.publish_certificate(
                window, views, payload_bytes, self.final_scores, theta_start_root
            )
            log.info(
                "certificate published (leader)",
                window=window,
                included=sorted(u for u, v in views.items() if v.in_gate and v.valid),
            )

        # (c) evaluate at θ_start(window)
        block_hash = await asyncio.to_thread(
            ctx.chain.block_hash,
            boundary_block(window + 1, ctx.manifest.start_block, ctx.manifest.blocks_per_window),
        )
        self.last_eval = await self.evaluator.evaluate_window(
            self.model, window, phase, payloads, self.compressor, block_hash
        )
        scores: dict[int, float] = {}
        for uid, rec in self.last_eval.items():
            self.ema.update(uid, rec.indicator, window)
            scores[uid] = rec.score
            log.info(
                "evaluated miner",
                window=window,
                miner=uid,
                gradient_score=rec.score,
                indicator=rec.indicator,
                ema=self.ema.value(uid),
            )
        if scores:
            self.book.rate_window(scores)

        # (d) overlap copy detection
        self._overlap_check(window, payloads, upload_ts)

        # (e) THE lockstep core: apply window's certified outer step bitwise
        await self._catch_up_retrying(from_window=window - 1, to_window=window)
        master = dict(self.model.iter_master_params())
        state_root_after = hash_named_tensors(master.items())
        log.info(
            "outer step applied (lockstep)",
            window=window,
            state_root_before=theta_start_root,
            state_root_after=state_root_after,
            changed=state_root_after != theta_start_root,
        )

        # (f) sync scores from miner debug slices vs our post-step params
        self.last_sync = await self._sync_scores(window, sorted(payloads), buckets, master)

        # (g) leader post-step duties
        probe_loss = await self._probe_loss(window, phase)
        self.probe_losses = (self.probe_losses + [probe_loss])[-cfg.rollback.spike_baseline_windows - 4 :]
        if is_leader:
            await self.leader.publish_debug_slices(window, master)
            await self.leader.maybe_checkpoint(
                window,
                master,
                self.outer_step.state_dict(),
                state_root_after,
                run_state_at(cfg, ctx.manifest, window + 1, world_size=ctx.protocol_world_size),
            )
            decision = self.leader.observe_probe_loss(window, probe_loss)
            if decision is not None:
                ctx.metrics.emit(
                    "rollback_activated", window=window, target=decision.target_window
                )
                self.state.save(self)
                raise SystemExit(3)  # relaunch against the amended manifest

        # (h) final scores + weights cadence
        universe = sorted(set(scores) | set(self.final_scores))
        self.final_scores = {
            uid: self.ledger.apply(
                uid,
                final_score(uid, self.book, self.ema, self.last_sync.get(uid, sync_score(1.0))),
                window,
            )
            for uid in universe
        }
        self.windows_since_weights += 1
        if self.windows_since_weights >= cfg.scoring.windows_per_weights:
            positive = {u: round(v, 4) for u, v in self.final_scores.items() if v > 0}
            log.info("submitting weights", window=window, final_scores=positive or "none")
            await submit_weights(ctx.chain, self.final_scores, cfg)
            self.windows_since_weights = 0

        # (i) audit ingest for the report-deadline window
        audited = window - cfg.audit.report_deadline_windows
        if audited >= 0:
            await ingest_window_audits(
                ctx.storage, ctx.chain, audited, self.ledger, apply_window=window
            )

        # (j) persist + telemetry
        self.state.save(self)
        log.info(
            "window processed",
            window=window,
            leader=is_leader,
            committed=len(commits),
            evaluated=len(self.last_eval),
            probe_loss=probe_loss,
            state_root_after=state_root_after,
            window_s=round(time.monotonic() - t_window, 1),
        )
        ctx.metrics.emit(
            "validator_window",
            window=window,
            evaluated=len(self.last_eval),
            committed=len(commits),
            leader=is_leader,
            probe_loss=probe_loss,
            state_root_after=state_root_after,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _gate_and_fetch(
        self,
        window: int,
        commits: Mapping[int, WindowCommit],
        buckets: Mapping[int, BucketCreds],
        theta_start_root: str,
    ) -> tuple[dict[int, CommitView], dict[int, bytes], dict[int, WindowPayload], dict[int, float]]:
        """Gate-check every commit, fetch + validate in-gate payloads."""
        ctx = self.ctx
        cfg = ctx.cfg
        boundary = ctx.clock.boundary_ts(window + 1)
        comp_shapes = {
            n: tuple(s) for n, s in self.model.param_shapes().items() if not n.endswith(DENSE_SUFFIX)
        }
        dense_shapes = {
            n: tuple(s) for n, s in self.model.param_shapes().items() if n.endswith(DENSE_SUFFIX)
        }
        views: dict[int, CommitView] = {}
        raw: dict[int, bytes] = {}
        payloads: dict[int, WindowPayload] = {}
        upload_ts: dict[int, float] = {}
        for uid in sorted(commits):
            commit = commits[uid]
            bucket = buckets.get(uid)
            key = keys.payload_key(window, uid, str(PAYLOAD_VERSION))
            in_gate = False
            if bucket is not None:
                try:
                    in_gate = await gate_check(
                        ctx.storage, bucket, key, boundary, cfg.window.upload_grace_s
                    )
                except (TimeoutError, StorageError, ExchangeError) as e:
                    log.warning("gate check failed", miner=uid, window=window, error=str(e))
            if not in_gate:
                self.ledger.missing_gradient(uid, window)
                views[uid] = CommitView(uid=uid, payload_hash=commit.payload_hash, in_gate=False, valid=False)
                continue
            valid = False
            full_hash = commit.payload_hash
            try:
                upload_ts[uid] = await ctx.storage.object_timestamp(bucket, key)
                data = await ctx.storage.get_bytes(
                    bucket, key, max_bytes=cfg.storage.max_payload_bytes
                )
                # The on-chain WindowCommit binds H(payload)'s 128-bit prefix; the full
                # hash goes into the certificate so every peer fetches at full strength.
                full_hash = hash_bytes(data)
                if not commit.binds_payload_hash(full_hash):
                    raise PayloadError("payload bytes do not match the on-chain commit")
                payload = deserialize(data, max_bytes=cfg.storage.max_payload_bytes)
                if payload.uid != uid or payload.window != window:
                    raise PayloadError("payload identity contradicts its commit")
                validate_structure(
                    payload,
                    comp_shapes,
                    dense_shapes,
                    cfg.compression.topk,
                    target_chunk=cfg.compression.target_chunk,
                )
                raw[uid] = data
                payloads[uid] = payload
                valid = True
                self.ledger.gradient_received(uid, window)
                if payload.metadata.state_root != theta_start_root:
                    self.ledger.sync_behind(uid, window)
            except (TimeoutError, StorageError, ExchangeError, PayloadError) as e:
                log.warning("payload rejected", miner=uid, window=window, error=str(e))
                self.ledger.invalid_payload(uid, window)
            views[uid] = CommitView(
                uid=uid, payload_hash=full_hash if valid else commit.payload_hash, in_gate=True, valid=valid
            )
        return views, raw, payloads, upload_ts

    def _overlap_check(
        self,
        window: int,
        payloads: Mapping[int, WindowPayload],
        upload_ts: Mapping[int, float],
    ) -> None:
        if len(payloads) < 2:
            return
        peer_indices: dict[int, dict[str, torch.Tensor]] = {}
        for uid, payload in payloads.items():
            idxs: dict[str, torch.Tensor] = {}
            for name, ct in payload.compressed.items():
                flat, _vals = sparse_pairs_from_compressed(name, ct, self.compressor)
                idxs[name] = flat
            peer_indices[uid] = idxs
        report = index_overlap_report(peer_indices, self.ctx.cfg.scoring.overlap_threshold)
        for pair in report.pairs:
            sev = severity(pair.overlap)
            if not sev:
                continue
            offender = determine_offender(pair, dict(upload_ts))
            self.ledger.overlap(offender, window, sev.multiplier, sev.naughty)
            log.warning(
                "index overlap sanction",
                window=window,
                pair=(pair.uid_a, pair.uid_b),
                overlap=round(pair.overlap, 4),
                offender=offender,
            )

    async def _sync_scores(
        self,
        window: int,
        uids: list[int],
        buckets: Mapping[int, BucketCreds],
        master: Mapping[str, torch.Tensor],
    ) -> dict[int, float]:
        """Debug-slice comparison → steps-behind estimate → sync_score per miner."""
        out: dict[int, float] = {}
        max_behind = self.ctx.cfg.scoring.sync_max_steps_behind
        for uid in uids:
            bucket = buckets.get(uid)
            steps_behind = 1.0  # unknown: mild penalty
            if bucket is not None:
                try:
                    slices = await get_debug_slices(self.ctx.storage, bucket, window, uid)
                    divergent = 0
                    compared = 0
                    for name in sorted(set(slices) & set(master)):
                        head = master[name].detach().reshape(-1)[: len(slices[name])]
                        ours = [float(v) for v in head.to(device="cpu", dtype=torch.float32).tolist()]
                        compared += 1
                        if ours != slices[name]:
                            divergent += 1
                    if compared:
                        steps_behind = 0.0 if divergent == 0 else max_behind
                except (TimeoutError, StorageError, ExchangeError):
                    steps_behind = 1.0
            out[uid] = sync_score(steps_behind)
        return out

    async def _probe_loss(self, window: int, phase: Any) -> float:
        """Canonical probe CE: fixed-sentinel random pool, post-outer-step model."""
        pairs = self.evaluator.pools.random_pool(
            self.ctx.manifest,
            self.ctx.run_seed,
            window,
            phase,
            self.ctx.cfg.scoring.eval_sequences,
            PROBE_BLOCK_HASH,
        )
        batch = await self.evaluator._batch_for_pairs(phase, pairs)  # noqa: SLF001 — same package
        if batch is None:
            return float("nan")
        return evaluate_sequences(self.model, [batch], device=self.ctx.device)

    async def _catch_up_retrying(self, *, from_window: int, to_window: int) -> None:
        last: Exception | None = None
        for attempt in range(self.catchup_retries):
            try:
                await catch_up_replica(
                    self.ctx, self.model, self.outer_step, from_window=from_window, to_window=to_window
                )
                return
            except (CatchUpError, StorageError, ExchangeError, TimeoutError) as e:
                last = e
                log.warning(
                    "validator catch-up attempt failed",
                    attempt=attempt + 1,
                    of=self.catchup_retries,
                    error=str(e),
                )
                await asyncio.sleep(self.catchup_retry_s)
        assert last is not None
        raise last
