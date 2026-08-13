"""Calibration dress rehearsal — full-protocol windows on one node (step B).

``run_calibration_windows`` drives the real ``C.core.window_runner.
WindowRunner`` through ``LocalLoopbackHarness`` (self as the only certified
peer): every window exercises training, compression, the two-phase commit,
the certificate, the certified gather, the deterministic outer step, the sync
check and the periodic checkpoint — exactly the live miner loop, minus other
peers. The report carries the loss curve, wall time and capacity utilization
per window, and a DETERMINISM CHECK: the follow-on window is trained twice
from the same θ (via ``run_training_phase``, which restores θ bitwise) and
must produce identical θ_end roots — the property the whole subnet audits.

On the fleet this runs inside the blessed container on a Tier-A node
(``entrypoint.sh calibrate``); on CPU it runs verbatim with the reference
backend and a tiny config (tests do).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import torch

from C.core.phase import resolve_phase
from C.core.window_runner import (
    RunState,
    SingleNodeComm,
    WindowOutcome,
    build_window_plan,
    run_state_at,
    run_training_phase,
)
from mok_core.config import RunConfig
from mok_core.config.manifest import RunManifest
from mok_core.config.schemas import FrozenModel
from mok_core.data import DatasetShardIndex, ShardReader
from mok_core.model import MoKTransformer

from .local_harness import LocalLoopbackHarness

__all__ = ["CalibrationError", "CalibrationReport", "run_calibration_windows"]


class CalibrationError(RuntimeError):
    pass


class CalibrationReport(FrozenModel):
    """What the dress rehearsal measured (all windows completed cleanly)."""

    windows: tuple[int, ...]
    loss_curve: tuple[float, ...]           # final_loss per window
    entry_losses: tuple[float, ...]
    window_wall_times: tuple[float, ...]    # seconds per full window (timer-injectable)
    capacity_utils: tuple[float, ...]       # max MoeHealth utilization per window
    state_roots: tuple[str, ...]            # post-outer-step root per window
    determinism_check: bool                 # same window twice -> identical θ_end roots

    @property
    def ok(self) -> bool:
        return self.determinism_check and len(self.windows) > 0


def run_calibration_windows(
    n_windows: int,
    cfg: RunConfig,
    manifest: RunManifest,
    *,
    model: MoKTransformer,
    index: DatasetShardIndex,
    shard_path: Callable[[int], Path],
    work_dir: str | Path,
    uid: int = 0,
    start_window: int = 0,
    device: str | torch.device = "cpu",
    timer: Callable[[], float] = time.perf_counter,
    determinism_probe: bool = True,
    metrics: object | None = None,
) -> CalibrationReport:
    """Run ``n_windows`` loopback windows starting at ``start_window``.

    The model advances in place (each window applies the real outer step), so
    the loss curve is a genuine short training trajectory. Any window that
    does not complete cleanly (desync/late/restart) raises
    ``CalibrationError`` — a dress rehearsal that cannot complete on loopback
    means the node is misconfigured. Synchronous by design (the CLI path);
    internally drives the async runner.
    """
    if n_windows <= 0:
        raise ValueError(f"n_windows must be positive, got {n_windows}")
    return asyncio.run(
        _run(
            n_windows,
            cfg,
            manifest,
            model=model,
            index=index,
            shard_path=shard_path,
            work_dir=Path(work_dir),
            uid=uid,
            start_window=start_window,
            device=device,
            timer=timer,
            determinism_probe=determinism_probe,
            metrics=metrics,
        )
    )


async def _run(
    n_windows: int,
    cfg: RunConfig,
    manifest: RunManifest,
    *,
    model: MoKTransformer,
    index: DatasetShardIndex,
    shard_path: Callable[[int], Path],
    work_dir: Path,
    uid: int,
    start_window: int,
    device: str | torch.device,
    timer: Callable[[], float],
    determinism_probe: bool,
    metrics: object | None,
) -> CalibrationReport:
    harness = LocalLoopbackHarness(
        model,
        cfg,
        manifest,
        index,
        shard_path=shard_path,
        work_dir=work_dir,
        uid=uid,
        device=device,
        metrics=metrics,
    )
    state = run_state_at(cfg, manifest, start_window, world_size=1)

    windows: list[int] = []
    finals: list[float] = []
    entries: list[float] = []
    walls: list[float] = []
    utils: list[float] = []
    roots: list[str] = []
    for window in range(start_window, start_window + n_windows):
        t0 = timer()
        outcome: WindowOutcome = await harness.run_window(window, state)
        walls.append(timer() - t0)
        if outcome.restart_required or outcome.desync or outcome.late_upload:
            raise CalibrationError(
                f"window {window} did not complete cleanly on loopback: "
                f"restart={outcome.restart_required} desync={outcome.desync} "
                f"late={outcome.late_upload} reason={outcome.reason!r}"
            )
        assert outcome.train_result is not None and outcome.state_root_after is not None
        windows.append(window)
        finals.append(outcome.train_result.final_loss)
        entries.append(outcome.train_result.entry_loss)
        utils.append(outcome.train_result.capacity_util_max)
        roots.append(outcome.state_root_after)
        state = outcome.state_after

    determinism = True
    if determinism_probe:
        determinism = _probe_determinism(
            model,
            cfg,
            manifest,
            uid=uid,
            window=start_window + n_windows,
            state=state,
            shard_path=shard_path,
            device=device,
        )

    return CalibrationReport(
        windows=tuple(windows),
        loss_curve=tuple(finals),
        entry_losses=tuple(entries),
        window_wall_times=tuple(walls),
        capacity_utils=tuple(utils),
        state_roots=tuple(roots),
        determinism_check=determinism,
    )


def _probe_determinism(
    model: MoKTransformer,
    cfg: RunConfig,
    manifest: RunManifest,
    *,
    uid: int,
    window: int,
    state: RunState,
    shard_path: Callable[[int], Path],
    device: str | torch.device,
) -> bool:
    """Train ``window`` twice from the model's current θ; roots must match.

    ``run_training_phase`` restores θ bitwise afterwards (asserted inside it),
    so the probe leaves the model untouched — and without compression state
    the run is a pure function, making root inequality a determinism failure.
    """
    phase = resolve_phase(manifest, cfg, window)
    run_seed = bytes.fromhex(manifest.prf.run_seed_hex)
    plan = build_window_plan(
        manifest, phase, run_seed=run_seed, uid=uid, window=window, rank=0, world_size=1
    )
    roots: list[str | None] = []
    for _ in range(2):
        readers = {i: ShardReader(shard_path(i), phase.seq_len) for i in set(plan.shard_ids)}
        try:
            artifacts = run_training_phase(
                model,
                cfg,
                manifest,
                phase,
                uid=uid,
                window=window,
                rank=0,
                world_size=1,
                comm=SingleNodeComm(),
                shard_lookup=readers.__getitem__,
                global_state=state,
                device=device,
                plan=plan,
            )
        finally:
            for reader in readers.values():
                reader.close()
        roots.append(artifacts.theta_end_root)
    return roots[0] is not None and roots[0] == roots[1]
