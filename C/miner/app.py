"""The miner application: the productionized window loop.

Wraps `C.core.window_runner.WindowRunner` (the exact wiring proven by the
loopback fixture in tests/unit/test_window_runner.py) into a long-running
process:

  1. Build the model per the manifest config (mok backend on CUDA when the
     wheel is importable, reference otherwise — logged loudly) and restore the
     newest checkpoint, else the verified seed-42 init.
  2. Catch up bitwise to the current chain window (`checkpoint.catch_up`).
  3. Fresh uids run `cfg.window.warmup_null_windows` warmup windows: the full
     hot path executes (compile caches warm, θ restored to θ_start as always)
     but publication is suppressed by forcing the upload gate closed, so the
     network never sees a cold miner's payload.
  4. The window loop: per window, prefetch window+1's shards concurrently with
     `run_window` (blocking chain calls run through `asyncio.to_thread` so the
     comms tasks keep the loop), then dispatch on the outcome:
       - `restart_required` → final state persists, clean `SystemExit(3)`
         (the supervisor relaunches into the new workspace shape);
       - `desync`         → `catch_up_replica` re-applies the missed windows
         from the leader's certified aggregator mirror, then continue;
       - `late_upload`    → telemetry only; the window still advanced.
  5. SIGTERM/SIGINT: finish the in-flight window, checkpoint, exit 0.

Rank-0-owns-comms: every rank constructs the runner; only rank 0 gets the
storage-side collaborators (checkpointer, metrics) — the runner itself already
routes chain/storage I/O through rank 0 and broadcasts verdicts.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import signal
from collections.abc import Callable, Mapping
from typing import Any

from C.core.checkpoint import CatchUpError, CertificatePendingError, Checkpointer, CheckpointMeta
from C.core.compress import ErrorFeedback
from C.core.exchange import ExchangeError, put_debug_slices, put_telemetry
from C.core.phase import resolve_phase
from C.core.window_runner import (
    RunState,
    WindowClock,
    WindowOutcome,
    WindowRunner,
    run_state_at,
)
from mok_core.chain.schemas import WindowCommit, decode_commitment
from mok_core.config.schemas import BucketCreds
from mok_core.determinism import hash_named_tensors
from mok_core.storage import StorageError
from mok_core.telemetry import get_logger

from .bootstrap import (
    NodeContext,
    build_compressor,
    catch_up_replica,
    materialize_replica,
)

__all__ = ["MinerApp", "RESTART_EXIT_CODE"]

log = get_logger("app.miner")

#: Exit code meaning "relaunch me" (phase changed the workspace shape).
RESTART_EXIT_CODE = 3


class _GateClock:
    """Clock wrapper whose `now()` can be forced past every gate (warmup windows)."""

    def __init__(self, inner: WindowClock) -> None:
        self.inner = inner
        self.force_late = False

    def boundary_ts(self, window: int) -> float:
        return self.inner.boundary_ts(window)

    def now(self) -> float:
        return math.inf if self.force_late else self.inner.now()


class MinerApp:
    """One miner node (all ranks construct one; rank 0 owns comms).

    `max_windows` bounds the number of completed (or recovered) windows —
    None runs forever; tests use small values. `on_window` is an observation
    hook called after every `run_window` outcome (tests advance fake clocks
    with it). `self_leader` defaults to `ctx.local`: single-node loopback runs
    certify themselves exactly like the step-B calibration bootstrap.
    """

    def __init__(
        self,
        ctx: NodeContext,
        *,
        self_leader: bool | None = None,
        max_windows: int | None = None,
        on_window: Callable[[WindowOutcome], None] | None = None,
        cert_poll_s: float = 2.0,
        cert_timeout_s: float = 180.0,
        catchup_retries: int = 5,
        catchup_retry_s: float = 10.0,
        window_poll_s: float = 2.0,
    ) -> None:
        self.ctx = ctx
        self.self_leader = ctx.local if self_leader is None else bool(self_leader)
        self.max_windows = max_windows
        self.on_window = on_window
        self.cert_poll_s = float(cert_poll_s)
        self.cert_timeout_s = float(cert_timeout_s)
        self.catchup_retries = int(catchup_retries)
        self.catchup_retry_s = float(catchup_retry_s)
        self.window_poll_s = float(window_poll_s)

        self.gate_clock = _GateClock(ctx.clock)
        self.compressor: Any = None  # built in run() once the model exists
        self.error_feedback = ErrorFeedback(beta=ctx.cfg.compression.ef_beta)
        self.checkpointer = Checkpointer(
            ctx.storage if ctx.rank == 0 else None, ctx.state_dir / "checkpoints"
        )
        self.model: Any = None
        self.outer_step: Any = None
        self.run_state = RunState(0, 0, 0)
        self.window = 0
        self.warmup_left = 0
        self.completed_windows = 0
        self.last_outcome: WindowOutcome | None = None
        self._stop = asyncio.Event()
        self._buckets: dict[int, BucketCreds] = {}
        self._leader_creds: BucketCreds = ctx.own_bucket

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def stop(self) -> None:
        """Request a graceful stop after the in-flight window."""
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, self.stop)

    async def run(self) -> int:
        """The whole miner lifetime; returns the process exit code (0 = clean)."""
        ctx = self.ctx
        self._install_signal_handlers()
        self.model, self.outer_step, from_window = await materialize_replica(ctx, self.checkpointer)
        self.compressor = build_compressor(self.model, ctx.cfg)
        self.window = from_window + 1
        self.run_state = run_state_at(
            ctx.cfg, ctx.manifest, self.window, world_size=ctx.protocol_world_size
        )
        self.warmup_left = ctx.cfg.window.warmup_null_windows if await self._is_fresh(from_window) else 0
        log.info(
            "miner ready",
            uid=ctx.uid,
            start_window=self.window,
            replica_root=hash_named_tensors(self.model.iter_master_params()),
            device=str(ctx.device),
            world_size=ctx.world_size,
            warmup_windows=self.warmup_left,
        )
        if self.warmup_left:
            log.info("fresh uid — warmup null windows ahead (train, publish nothing)", warmup=self.warmup_left)

        while True:
            if self._stop.is_set():
                await self._final_checkpoint()
                return 0
            head = await asyncio.to_thread(ctx.chain.current_window, ctx.manifest)
            if self.window < head:
                log.info("behind chain head — catching up", my_window=self.window, head=head)
                await self._recover(to_window=head - 1)
                continue
            if self.window > head:
                log.info("waiting for next window boundary", next_window=self.window, head=head)
            while self.window > head and not self._stop.is_set():
                await asyncio.sleep(self.window_poll_s)
                head = await asyncio.to_thread(ctx.chain.current_window, ctx.manifest)
            if self._stop.is_set():
                continue

            outcome = await self._run_one_window()
            self.last_outcome = outcome
            if self.on_window is not None:
                self.on_window(outcome)

            if outcome.restart_required:
                log.warning("phase requires restart — exiting for relaunch", reason=outcome.reason)
                ctx.metrics.emit("restart_required", window=self.window, reason=outcome.reason)
                raise SystemExit(RESTART_EXIT_CODE)
            if outcome.desync:
                ctx.metrics.emit("desync", window=self.window, reason=outcome.reason)
                log.warning(
                    "window desynced — entering catch-up",
                    window=self.window,
                    reason=outcome.reason,
                    late_upload=outcome.late_upload,
                )
                head = await asyncio.to_thread(ctx.chain.current_window, ctx.manifest)
                await self._recover(to_window=max(self.window, head - 1))
                self.completed_windows += 1
                if self.warmup_left > 0:
                    self.warmup_left -= 1
                    log.info("warmup window served (via catch-up)", warmup_left=self.warmup_left)
            else:
                if outcome.late_upload:
                    log.warning("late upload — payload skipped this window", window=self.window)
                await self._post_window(outcome)
                self.run_state = outcome.state_after
                self.window += 1
                self.completed_windows += 1
                if self.warmup_left > 0:
                    self.warmup_left -= 1

            if self.max_windows is not None and self.completed_windows >= self.max_windows:
                await self._final_checkpoint()
                return 0

    # ------------------------------------------------------------------ #
    # One window
    # ------------------------------------------------------------------ #

    def _build_runner(self, phase_data: str) -> WindowRunner:
        ctx = self.ctx
        return WindowRunner(
            self.model,
            ctx.cfg,
            ctx.manifest,
            uid=ctx.uid,
            rank=ctx.rank,
            world_size=ctx.world_size,
            comm=ctx.comm,
            storage=ctx.storage,
            chain=ctx.chain,
            shard_cache=ctx.shard_caches[phase_data],
            fetch_fn=ctx.fetch_fns[phase_data],
            compressor=self.compressor,
            error_feedback=self.error_feedback,
            outer_step=self.outer_step,
            checkpointer=self.checkpointer if ctx.rank == 0 else None,
            metrics=ctx.metrics if ctx.rank == 0 else None,
            wait_for_gate=not ctx.local,
            clock=self.gate_clock,
            peer_buckets=lambda _w: self._buckets,
            leader_bucket=lambda _w: self._leader_creds,
            device=ctx.device,
            self_leader=self.self_leader,
            sign_fn=ctx.signer.sign,
            cert_poll_s=self.cert_poll_s,
            cert_timeout_s=self.cert_timeout_s,
        )

    async def _run_one_window(self) -> WindowOutcome:
        ctx = self.ctx
        window = self.window
        phase = resolve_phase(ctx.manifest, ctx.cfg, window)
        if ctx.rank == 0:
            self._buckets = await asyncio.to_thread(ctx.peer_buckets)
            self._leader_creds = (
                ctx.own_bucket if self.self_leader else await asyncio.to_thread(ctx.leader_bucket)
            )
        runner = self._build_runner(phase.data)
        self.gate_clock.force_late = self.warmup_left > 0

        prefetch = asyncio.create_task(self._prefetch_next(window + 1))
        try:
            outcome = await runner.run_window(window, self.run_state)
        finally:
            self.gate_clock.force_late = False
            prefetch.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await prefetch
        return outcome

    async def _prefetch_next(self, window: int) -> None:
        """Comms-side overlap: warm the shard cache for window+1 during training."""
        ctx = self.ctx
        if ctx.rank != 0 or ctx.manifest.is_void(window):
            return
        try:
            phase = resolve_phase(ctx.manifest, ctx.cfg, window)
            from C.core.window_runner import build_window_plan  # noqa: PLC0415 — cycle-free but heavy

            plan = build_window_plan(
                ctx.manifest,
                phase,
                run_seed=ctx.run_seed,
                uid=ctx.uid,
                window=window,
                rank=ctx.rank,
                world_size=ctx.world_size,
            )
            await ctx.shard_caches[phase.data].prefetch(
                set(plan.shard_ids), ctx.fetch_fns[phase.data]
            )
        except Exception as e:  # noqa: BLE001 — prefetch is opportunistic
            log.info("next-window prefetch skipped", window=window, error=str(e))

    async def _post_window(self, outcome: WindowOutcome) -> None:
        """Rank-0 after a completed window: debug slices + telemetry (best-effort)."""
        ctx = self.ctx
        if ctx.rank != 0:
            return
        try:
            if not outcome.late_upload:
                await put_debug_slices(
                    ctx.storage, outcome.window, ctx.uid, dict(self.model.iter_master_params())
                )
            await put_telemetry(
                ctx.storage,
                outcome.window,
                ctx.uid,
                {
                    "window": outcome.window,
                    "uid": ctx.uid,
                    "late_upload": bool(outcome.late_upload),
                    "final_loss": None
                    if outcome.train_result is None
                    else float(outcome.train_result.final_loss),
                    "tokens": 0 if outcome.train_result is None else int(outcome.train_result.tokens),
                    "state_root_after": outcome.state_root_after,
                },
            )
        except (TimeoutError, StorageError, ExchangeError) as e:
            log.warning("post-window publication failed", window=outcome.window, error=str(e))

    # ------------------------------------------------------------------ #
    # Recovery / persistence
    # ------------------------------------------------------------------ #

    def _gate_close_ts(self, window: int) -> float:
        """When the leader can first have certified `window` (its upload gate closes)."""
        ctx = self.ctx
        return ctx.clock.boundary_ts(window + 1) + ctx.cfg.window.upload_grace_s

    async def _recover(self, *, to_window: int) -> None:
        """Catch up (window-1, to_window] then rejoin.

        Optimistic then patient: the certificate for `to_window` may already
        exist (leaders publish asynchronously), so try at once; if it is still
        missing, the gate for that window may simply not have closed yet (a
        rejoining node commonly desyncs mid-window) — keep retrying until the
        gate closes plus a window-scale publication budget, not for seconds.
        """
        ctx = self.ctx
        from_window = self.window - 1
        window_s = ctx.clock.boundary_ts(1) - ctx.clock.boundary_ts(0)
        # Retry until at least: gate close of the target + half a window for the
        # leader to gather/evaluate/certify. Never less than the configured retries.
        patience_until = self._gate_close_ts(to_window) + 0.5 * window_s
        attempt = 0
        last: CatchUpError | None = None
        done = False
        while not done and not self._stop.is_set():
            attempt += 1
            try:
                await catch_up_replica(
                    ctx, self.model, self.outer_step, from_window=from_window, to_window=to_window
                )
                done = True
            except CertificatePendingError as e:
                # Leader down/lagging: the certificate is the only source of this
                # window's outer step — wait for it, however long it takes.
                last = e
                log.warning(
                    "leader has not certified the window yet — waiting",
                    attempt=attempt,
                    error=str(e),
                )
                await asyncio.sleep(max(self.catchup_retry_s, 5.0))
            except (CatchUpError, StorageError, ExchangeError, TimeoutError) as e:
                last = e if isinstance(e, CatchUpError) else CatchUpError(str(e))
                now = ctx.clock.now()
                log.warning(
                    "catch-up attempt failed",
                    attempt=attempt,
                    error=str(e),
                    gate_closes_in_s=round(self._gate_close_ts(to_window) - now, 1),
                )
                if attempt >= self.catchup_retries and now >= patience_until:
                    break
                await asyncio.sleep(self.catchup_retry_s)
        if not done:
            if self._stop.is_set():
                return
            assert last is not None
            raise last
        self.window = to_window + 1
        self.run_state = run_state_at(
            ctx.cfg, ctx.manifest, self.window, world_size=ctx.protocol_world_size
        )
        log.info(
            "caught up — rejoining",
            through_window=to_window,
            next_window=self.window,
            replica_root=hash_named_tensors(self.model.iter_master_params()),
        )
        ctx.metrics.emit("catch_up", window=self.window, to_window=to_window)

    async def _is_fresh(self, from_window: int) -> bool:
        """Fresh uid: no checkpoint restored AND no prior WindowCommit on-chain."""
        if from_window >= 0:
            return False
        wire = await asyncio.to_thread(self.ctx.chain.get_commitment, self.ctx.uid)
        if not wire:
            return True
        try:
            return not isinstance(decode_commitment(wire), WindowCommit)
        except ValueError:
            return True

    async def _final_checkpoint(self) -> None:
        """Persist θ_start(window) on a clean stop (rank 0, best-effort)."""
        ctx = self.ctx
        if ctx.rank != 0 or self.model is None or self.completed_windows == 0:
            return
        window = self.window - 1
        if window < 0 or window in self.checkpointer.local_windows():
            return
        master: Mapping[str, Any] = dict(self.model.iter_master_params())
        meta = CheckpointMeta(
            window=window,
            global_step=self.run_state.global_step,
            tokens_consumed=self.run_state.tokens_consumed,
            state_root=hash_named_tensors(master.items()),
            manifest_hash=ctx.manifest.manifest_hash(),
            spec_version=ctx.manifest.spec_version,
        )
        try:
            await self.checkpointer.save(window, master, self.outer_step.state_dict(), meta)
            log.info("final checkpoint saved", window=window)
        except Exception as e:  # noqa: BLE001 — shutdown must not crash on I/O
            log.warning("final checkpoint failed", window=window, error=str(e))
