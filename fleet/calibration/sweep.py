"""MoKConfig sweep — tune (comm SMs, minibatch) on the real fleet.

The sweep times K toy training windows per grid point through the UNMODIFIED
``run_training_phase`` (compression-free: only the knobs under test affect
the timed region) and emits the winner as ``subnet/configs/mok_tuned.yaml`` — an
overlay every node loads, so the tuned values are part of the blessed
configuration rather than a local choice.

Structure is complete for the GPU run (each point rebuilds the model through
``model_factory`` with its ``MoKRuntimeConfig`` so the mok backend re-plans
its workspace); CPU tests run 2 tiny points with the reference backend and an
injected fake timer — the ranking/selection/emission logic is identical.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

import subnet
from mok_core.config import MoKRuntimeConfig, RunConfig
from mok_core.config.manifest import RunManifest
from mok_core.config.schemas import FrozenModel
from mok_core.data import ShardReader
from mok_core.model import MoKTransformer
from subnet.core.phase import resolve_phase
from subnet.core.window_runner import (
    SingleNodeComm,
    build_window_plan,
    run_state_at,
    run_training_phase,
)

__all__ = [
    "DEFAULT_TUNED_PATH",
    "SweepPoint",
    "SweepResult",
    "apply_point",
    "default_grid",
    "emit_tuned_yaml",
    "run_sweep",
    "select_best",
]

DEFAULT_TUNED_PATH = Path(subnet.__file__).resolve().parent / "configs" / "mok_tuned.yaml"


class SweepPoint(FrozenModel):
    """One grid point: forward/backward comm SMs + the schedule minibatch."""

    fwd_num_comm_sms: int
    bwd_num_comm_sms: int
    minibatch_size: int


@dataclass(frozen=True)
class SweepResult:
    point: SweepPoint
    mok: MoKRuntimeConfig          # the fully-validated runtime config of this point
    mean_window_s: float
    final_loss: float


def default_grid() -> tuple[SweepPoint, ...]:
    """The production grid: comm SMs {24, 36, 48} × minibatch {2048, 4096, 8192}
    (fwd == bwd; bench_mok.py's 36/36 + 4096 sits at the center)."""
    return tuple(
        SweepPoint(fwd_num_comm_sms=sms, bwd_num_comm_sms=sms, minibatch_size=mb)
        for sms in (24, 36, 48)
        for mb in (2048, 4096, 8192)
    )


def apply_point(cfg: RunConfig, point: SweepPoint) -> RunConfig:
    """cfg with the point's mok knobs applied — fully re-validated (an illegal
    combination fails here, not at kernel launch)."""
    mok = MoKRuntimeConfig(
        **{
            **cfg.mok.model_dump(),
            "fwd_num_comm_sms": point.fwd_num_comm_sms,
            "bwd_num_comm_sms": point.bwd_num_comm_sms,
            "minibatch_size": point.minibatch_size,
        }
    )
    return RunConfig(**{**cfg.model_dump(), "mok": mok.model_dump()})


def run_sweep(
    cfg: RunConfig,
    manifest: RunManifest,
    *,
    model_factory: Callable[[RunConfig], MoKTransformer],
    shard_path: Callable[[int], Path],
    points: Sequence[SweepPoint] | None = None,
    windows_per_point: int = 2,
    start_window: int = 0,
    uid: int = 0,
    device: str | torch.device = "cpu",
    timer: Callable[[], float] = time.perf_counter,
) -> list[SweepResult]:
    """Time ``windows_per_point`` windows for every grid point.

    ``model_factory(cfg_point)`` must return a θ-identical model per call
    (e.g. ``init_model`` from a fixed seed, or a checkpoint load) so points
    are compared on identical work. The training phase restores θ after each
    window, so every timed window starts from the same weights — pure timing,
    no confound from training progress.
    """
    if windows_per_point <= 0:
        raise ValueError(f"windows_per_point must be positive, got {windows_per_point}")
    grid = tuple(points) if points is not None else default_grid()
    if not grid:
        raise ValueError("empty sweep grid")

    results: list[SweepResult] = []
    for point in grid:
        cfg_p = apply_point(cfg, point)
        model = model_factory(cfg_p)
        state = run_state_at(cfg_p, manifest, start_window, world_size=1)
        run_seed = bytes.fromhex(manifest.prf.run_seed_hex)
        times: list[float] = []
        final_loss = float("nan")
        for window in range(start_window, start_window + windows_per_point):
            phase = resolve_phase(manifest, cfg_p, window)
            plan = build_window_plan(
                manifest, phase, run_seed=run_seed, uid=uid, window=window, rank=0, world_size=1
            )
            readers = {i: ShardReader(shard_path(i), phase.seq_len) for i in set(plan.shard_ids)}
            try:
                t0 = timer()
                artifacts = run_training_phase(
                    model,
                    cfg_p,
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
                times.append(timer() - t0)
            finally:
                for reader in readers.values():
                    reader.close()
            final_loss = artifacts.result.final_loss
        results.append(
            SweepResult(
                point=point,
                mok=cfg_p.mok,
                mean_window_s=sum(times) / len(times),
                final_loss=final_loss,
            )
        )
    return results


def select_best(results: Sequence[SweepResult]) -> SweepResult:
    """Fastest mean window wins; ties break to the earlier grid point (stable)."""
    if not results:
        raise ValueError("no sweep results to select from")
    best = results[0]
    for r in results[1:]:
        if r.mean_window_s < best.mean_window_s:
            best = r
    return best


def emit_tuned_yaml(
    mok: MoKRuntimeConfig, path: str | Path = DEFAULT_TUNED_PATH, *, provenance: str
) -> Path:
    """Write the tuned overlay (``mok:`` section) with a provenance comment.

    The file is a plain RunConfig overlay: ``load_run_config(base, ...,
    mok_tuned.yaml)`` applies it on every node.
    """
    out = Path(path)
    body = yaml.safe_dump({"mok": mok.model_dump()}, sort_keys=True, default_flow_style=False)
    header = (
        "# mok_tuned.yaml — MoK runtime knobs pinned by the fleet calibration sweep.\n"
        f"# provenance: {provenance}\n"
        "# Regenerate with `mok-calibrate sweep`; hand edits desync the fleet.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + body, encoding="utf-8")
    return out
