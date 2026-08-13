"""The auditor application: sampled bitwise window replays.

An auditor is a Tier-A replica that NEVER trains for rewards: it holds
θ_start(w) in lockstep exactly like a miner (checkpoint restore + bitwise
catch-up), and at the gate close of every window `w`:

  1. computes the consensus audit assignment (`C.core.replay.audit_sampler`)
     from (run_seed, block_hash of w+1's boundary, w, active miners, ρ, the
     sorted auditor set) — every honest node derives the identical partition;
  2. for each assigned (miner, w): replays the miner's window with
     `WindowReplayer` — the exact `run_training_phase` code the miner ran, on
     the exact θ_start(w) the replica already holds (the replayer refuses to
     run anywhere else and restores θ_start bitwise afterwards, so one
     materialized θ_start serves every task of the window);
  3. signs the verdict over its canonical unsigned fields and publishes it to
     its OWN bucket (`exchange.put_audit_report`) — validators poll auditor
     buckets and verify the hotkey signature (trust model documented in
     C/miner/bootstrap.py); a wall-time budget guards multi-task windows
     (tasks over budget are skipped with telemetry, never rushed);
  4. applies window w's certified outer step (catch_up) and advances.

Auditor discovery: this app maintains the `AUDITOR_COMMITMENT` ("auditor.v1")
chain commitment. Auditors make no WindowCommits, so the single commitment
slot keeps the tag; validators and fellow auditors read it via
`auditor_uids_from_chain`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from C.core.checkpoint import CatchUpError, Checkpointer
from C.core.exchange import ExchangeError, put_audit_report, put_telemetry
from C.core.phase import resolve_phase
from C.core.replay import (
    AuditReport,
    PreconditionError,
    ReplayTask,
    WindowReplayer,
    audit_sampler,
    sign_report,
)
from C.core.window_runner import build_window_plan, run_state_at
from C.miner.bootstrap import (
    AUDITOR_COMMITMENT,
    NodeContext,
    auditor_uids_from_chain,
    catch_up_replica,
    materialize_replica,
)
from mok_core.chain.schemas import WindowCommit
from mok_core.chain.windows import boundary_block
from mok_core.data import ShardReader, WindowBatchPlan
from mok_core.storage import StorageError
from mok_core.telemetry import get_logger

__all__ = ["AUDIT_WALL_BUDGET_FRACTION", "AuditorApp"]

log = get_logger("app.auditor")

#: Fraction of the window's wall time an auditor may spend replaying before
#: skipping remaining tasks (it must still apply the outer step in time).
AUDIT_WALL_BUDGET_FRACTION = 0.5


class AuditorApp:
    """One auditor process (single-node; multi-rank replay mirrors the miner)."""

    def __init__(
        self,
        ctx: NodeContext,
        *,
        max_windows: int | None = None,
        on_window: Callable[[int], None] | None = None,
        wall_budget_s: float | None = None,
        catchup_retries: int = 5,
        catchup_retry_s: float = 5.0,
        poll_s: float = 2.0,
    ) -> None:
        self.ctx = ctx
        self.max_windows = max_windows
        self.on_window = on_window
        window_wall_s = ctx.manifest.blocks_per_window * ctx.cfg.chain.block_time_s
        self.wall_budget_s = (
            float(wall_budget_s)
            if wall_budget_s is not None
            else window_wall_s * AUDIT_WALL_BUDGET_FRACTION
        )
        self.catchup_retries = int(catchup_retries)
        self.catchup_retry_s = float(catchup_retry_s)
        self.poll_s = float(poll_s)
        self.checkpointer = Checkpointer(ctx.storage, ctx.state_dir / "checkpoints")
        self.state_path = ctx.state_dir / "auditor_state.json"
        self.model: Any = None
        self.outer_step: Any = None
        self.replayer: WindowReplayer | None = None
        self.window = 0
        self.completed_windows = 0
        self.reports: list[AuditReport] = []
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

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"window": self.window}), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def _load_state(self) -> None:
        if self.state_path.is_file():
            self.window = int(json.loads(self.state_path.read_text(encoding="utf-8"))["window"])

    async def _ensure_auditor_commitment(self) -> None:
        ctx = self.ctx
        current = await asyncio.to_thread(ctx.chain.get_commitment, ctx.uid)
        if current != AUDITOR_COMMITMENT:
            await asyncio.to_thread(ctx.chain.commit, AUDITOR_COMMITMENT)
            log.info("auditor commitment published", tag=AUDITOR_COMMITMENT)

    # ------------------------------------------------------------------ #
    # Shard plumbing for the replayer
    # ------------------------------------------------------------------ #

    def _shard_lookup_factory(self, plan: WindowBatchPlan) -> Any:
        """ContextManager[shard_idx -> ShardReader] over the verified cache.

        Shards must already be resident (`_prefetch_plan` runs before replay —
        the factory itself is called from the synchronous replay path)."""
        cache = self.ctx.shard_caches[plan.dataset_name]

        @contextmanager
        def open_readers() -> Iterator[Callable[[int], ShardReader]]:
            readers = {
                i: ShardReader(cache.path_for(i), plan.seq_len) for i in set(plan.shard_ids)
            }
            try:
                yield readers.__getitem__
            finally:
                for reader in readers.values():
                    reader.close()

        return open_readers()

    async def _prefetch_plan(self, miner_uid: int, window: int) -> WindowBatchPlan:
        ctx = self.ctx
        phase = resolve_phase(ctx.manifest, ctx.cfg, window)
        plan = build_window_plan(
            ctx.manifest,
            phase,
            run_seed=ctx.run_seed,
            uid=miner_uid,
            window=window,
            rank=ctx.rank,
            world_size=ctx.world_size,
        )
        await ctx.shard_caches[phase.data].prefetch(
            set(plan.shard_ids), ctx.fetch_fns[phase.data]
        )
        return plan

    # ------------------------------------------------------------------ #
    # The run loop
    # ------------------------------------------------------------------ #

    async def run(self) -> int:
        ctx = self.ctx
        self._install_signal_handlers()
        self.model, self.outer_step, from_window = await materialize_replica(ctx, self.checkpointer)
        self.replayer = WindowReplayer(
            self.model,
            ctx.cfg,
            ctx.manifest,
            comm=ctx.comm,
            shard_lookup_factory=self._shard_lookup_factory,
            auditor_uid=ctx.uid,
            rank=ctx.rank,
            world_size=ctx.world_size,
            device=ctx.device,
        )
        self.window = from_window + 1
        self._load_state()
        self.window = max(self.window, from_window + 1)
        await self._ensure_auditor_commitment()
        log.info("auditor running", start_window=self.window)

        while True:
            if self._stop.is_set():
                self._save_state()
                return 0
            head = await asyncio.to_thread(ctx.chain.current_window, ctx.manifest)
            gate_close = ctx.clock.boundary_ts(self.window + 1) + ctx.cfg.window.upload_grace_s
            if self.window > head - 1 or ctx.clock.now() < gate_close:
                await asyncio.sleep(self.poll_s)
                continue
            await self.process_window(self.window)
            if self.on_window is not None:
                self.on_window(self.window)
            self.window += 1
            self.completed_windows += 1
            self._save_state()
            if self.max_windows is not None and self.completed_windows >= self.max_windows:
                return 0

    async def process_window(self, window: int) -> None:
        """Audit the sampled miners of `window`, then apply its outer step."""
        ctx = self.ctx
        if ctx.manifest.is_void(window):
            log.info("void window skipped", window=window)
            return
        commits: dict[int, WindowCommit] = await asyncio.to_thread(
            ctx.chain.get_window_commits, window
        )
        tasks = await self._my_tasks(window, commits)
        t0 = time.monotonic()
        done = 0
        skipped = 0
        for miner_uid, w in tasks:
            if time.monotonic() - t0 > self.wall_budget_s:
                skipped = len(tasks) - done
                log.warning(
                    "audit wall budget exhausted — skipping remaining tasks",
                    window=window,
                    skipped=skipped,
                )
                break
            await self._audit_one(miner_uid, w, commits[miner_uid])
            done += 1
        with contextlib.suppress(TimeoutError, StorageError, ExchangeError):
            await put_telemetry(
                ctx.storage,
                window,
                ctx.uid,
                {
                    "window": window,
                    "uid": ctx.uid,
                    "role": "auditor",
                    "assigned": len(tasks),
                    "replayed": done,
                    "skipped_over_budget": skipped,
                    "wall_time_s": time.monotonic() - t0,
                },
            )
        # Advance the replica: apply window's certified outer step bitwise.
        await self._catch_up_retrying(from_window=window - 1, to_window=window)

    async def _my_tasks(
        self, window: int, commits: dict[int, WindowCommit]
    ) -> list[tuple[int, int]]:
        """This auditor's slice of the consensus audit assignment for `window`."""
        ctx = self.ctx
        active = sorted(commits)
        if not active:
            return []
        block_hash = await asyncio.to_thread(
            ctx.chain.block_hash,
            boundary_block(window + 1, ctx.manifest.start_block, ctx.manifest.blocks_per_window),
        )
        auditors = sorted(set(auditor_uids_from_chain(ctx.chain)) | {ctx.uid})
        assignments = audit_sampler(
            ctx.run_seed, block_hash, window, active, ctx.cfg.audit.probability, auditors
        )
        mine = assignments.get(ctx.uid, [])
        log.info(
            "audit assignment",
            window=window,
            active=len(active),
            auditors=auditors,
            mine=[uid for uid, _ in mine],
        )
        return mine

    async def _audit_one(self, miner_uid: int, window: int, commit: WindowCommit) -> None:
        """Replay one miner-window, sign the verdict, publish it (best-effort I/O)."""
        ctx = self.ctx
        assert self.replayer is not None
        await self._prefetch_plan(miner_uid, window)
        state = run_state_at(ctx.cfg, ctx.manifest, window, world_size=ctx.protocol_world_size)
        task = ReplayTask(miner_uid=miner_uid, window=window, commit=commit)
        try:
            report = await asyncio.to_thread(self.replayer.replay, task, global_state=state)
        except PreconditionError as e:
            log.error("replay precondition failed — replica off θ_start", error=str(e))
            raise
        report = sign_report(report, ctx.signer.sign)
        self.reports.append(report)
        ctx.metrics.emit(
            "audit_replay",
            window=window,
            miner=miner_uid,
            match=report.match,
            wall_time_s=report.wall_time_s,
        )
        try:
            key = await put_audit_report(ctx.storage, report.to_json())
            log.info("audit report published", key=key, match=report.match)
        except (TimeoutError, StorageError, ExchangeError) as e:
            log.warning("audit report upload failed", miner=miner_uid, window=window, error=str(e))

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
                    "auditor catch-up attempt failed",
                    attempt=attempt + 1,
                    of=self.catchup_retries,
                    error=str(e),
                )
                await asyncio.sleep(self.catchup_retry_s)
        assert last is not None
        raise last
