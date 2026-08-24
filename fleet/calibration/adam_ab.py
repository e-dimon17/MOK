"""The Adam-reset A/B — the calibration gate behind protocol decision #1.

The protocol resets the inner AdamW every window so a window is a pure
function of (θ_start, uid, window, manifest) and audits replay exactly one
window. The risk is optimizer-warmup drag: fresh second moments every H
steps. This experiment runs the SAME windows (same θ_start, same seeds, same
data plans) under two optimizer lifetimes — reset every window vs reset every
K windows (fallback K=5) — and compares the loss trajectories.

Decision rule: if the final-loss penalty of resetting every
window is under ``threshold_nats`` (default 0.01 nats), keep reset=1 and its
single-window audit property; otherwise pin
``inner.adam_reset_every_windows=K`` in the run config before window 0.

Implementation note: both arms run through ``_run_window_with_optimizer`` — a
mirror of ``subnet.core.inner_loop.InnerLoop.run_window`` with exactly ONE degree
of freedom injected: the optimizer's lifetime. The mirror reuses InnerLoop's
own plan validation and grad-clip method, and the reset-every-window arm is
pinned BITWISE against the real ``InnerLoop`` in
``tests/unit/test_fleet_adam_ab.py``, so the A/B measures optimizer dynamics,
never implementation drift.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import torch

from mok_core.config import RunConfig
from mok_core.config.manifest import RunManifest
from mok_core.config.schemas import FrozenModel
from mok_core.data import ShardReader, WindowBatchPlan
from mok_core.model import MoKTransformer, loss_head
from subnet.core.inner_loop import IGNORE_INDEX, InnerLoop
from subnet.core.moe_health import MoeHealth
from subnet.core.phase import accum_at, lr_at, resolve_phase
from subnet.core.window_runner import build_window_plan, run_state_at
from subnet.core.zero1 import Comm, SingleProcessComm, Zero1Adam, flat_grad_all_reduce

__all__ = ["ABReport", "run_adam_ab", "run_arm"]

DEFAULT_K = 5
DEFAULT_THRESHOLD_NATS = 0.01


class ABReport(FrozenModel):
    """Loss trajectories of both arms + the calibration recommendation."""

    n_windows: int
    k: int
    threshold_nats: float
    losses_reset_every_window: tuple[float, ...]   # final_loss per window, reset=1 arm
    losses_reset_every_k: tuple[float, ...]        # final_loss per window, reset=K arm
    delta_final_loss: float                        # final(reset=1) - final(reset=K), nats
    keep_reset_every_window: bool                  # delta < threshold -> keep reset=1
    recommendation: str                            # the config line to pin


def _run_window_with_optimizer(
    model: MoKTransformer,
    cfg: RunConfig,
    loop: InnerLoop,
    plan: WindowBatchPlan,
    shard_lookup: Callable[[int], ShardReader],
    optimizer: Zero1Adam,
    *,
    window: int,
    comm: Comm,
    device: torch.device,
    global_inner_step0: int,
    tokens_consumed0: int,
) -> tuple[list[float], int]:
    """InnerLoop.run_window with the optimizer injected (see module docstring).

    Returns (per-step mean losses, tokens consumed). Everything else — plan
    validation, target shift, fixed reductions, clip, LR, MoeHealth order —
    mirrors ``InnerLoop.run_window`` statement for statement; ``loop`` supplies
    the shared ``_check_plan`` / ``_clip_grad_norm`` implementations so the
    consensus-sensitive pieces exist exactly once.
    """
    loop._check_plan(plan, window)  # noqa: SLF001 — deliberate reuse, pinned by the bitwise test
    named_params = dict(model.named_parameters())
    replicated = {n: p for n, p in named_params.items() if not model.is_expert_local(n)}
    health = MoeHealth(
        model,
        cfg,
        capacity_multiplier=loop.phase.capacity_multiplier,
        tokens_per_launch=loop.phase.tokens_per_rank_microbatch,
    )
    num_layers = len(model.moe_layers())
    num_experts = cfg.model.num_experts
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    seqs = plan.seqs_per_microbatch
    tokens_consumed = tokens_consumed0
    step_losses: list[float] = []

    for s in range(plan.inner_steps):
        accum = max(1, min(accum_at(cfg.window, tokens_consumed), plan.grad_accum))
        for p in named_params.values():
            p.grad = None
        step_loads = [torch.zeros(num_experts, dtype=torch.int64) for _ in range(num_layers)]
        loss_sum = 0.0
        for m in range(accum):
            tokens = plan.microbatch_tokens(s, m, shard_lookup).to(device)
            inputs = tokens.view(seqs, plan.seq_len)
            targets = torch.full_like(inputs, IGNORE_INDEX)
            targets[:, :-1] = inputs[:, 1:]
            with autocast:
                output = model(inputs)
                losses = loss_head(output.logits, targets, output.loss_inputs, cfg.model)
            (losses.total / accum).backward()
            loss_sum += float(losses.total.detach())
            with torch.no_grad():
                for i, stats in enumerate(output.loss_inputs):
                    step_loads[i] += stats.load.detach().cpu()

        flat_grad_all_reduce(replicated, comm, loop.world_size)
        loop._clip_grad_norm(named_params)  # noqa: SLF001 — deliberate reuse (see above)
        lr = lr_at(loop.phase.lr, global_inner_step0 + s, cfg.tokens_per_inner_step)
        optimizer.step(lr)
        health.post_step(step_loads)
        step_losses.append(loss_sum / accum)
        tokens_consumed += accum * plan.tokens_per_rank_microbatch * loop.world_size

    return step_losses, tokens_consumed - tokens_consumed0


def run_arm(
    model: MoKTransformer,
    cfg: RunConfig,
    manifest: RunManifest,
    *,
    n_windows: int,
    reset_every: int,
    shard_path: Callable[[int], Path],
    uid: int = 0,
    start_window: int = 0,
    device: str | torch.device = "cpu",
) -> list[float]:
    """One arm: ``n_windows`` consecutive windows, resetting the inner AdamW
    every ``reset_every`` windows. Returns the final loss of each window.
    The model advances in place window over window (inner steps compound —
    this is the trajectory under test, without the outer loop's compression
    so the arms differ only in optimizer lifetime).
    """
    if reset_every <= 0:
        raise ValueError(f"reset_every must be positive, got {reset_every}")
    comm = SingleProcessComm()
    device_t = torch.device(device)
    state = run_state_at(cfg, manifest, start_window, world_size=1)
    global_inner_step = state.global_inner_step
    tokens_consumed = state.tokens_consumed
    run_seed = bytes.fromhex(manifest.prf.run_seed_hex)

    optimizer: Zero1Adam | None = None
    finals: list[float] = []
    for offset in range(n_windows):
        window = start_window + offset
        phase = resolve_phase(manifest, cfg, window)
        loop = InnerLoop(model, cfg, phase, rank=0, world_size=1, comm=comm, device=device_t)
        plan = build_window_plan(
            manifest, phase, run_seed=run_seed, uid=uid, window=window, rank=0, world_size=1
        )
        if optimizer is None or offset % reset_every == 0:
            optimizer = Zero1Adam.fresh(
                dict(model.named_parameters()),
                cfg.inner,
                rank=0,
                world_size=1,
                is_expert_local=model.is_expert_local,
                comm=comm,
            )
        readers = {i: ShardReader(shard_path(i), phase.seq_len) for i in set(plan.shard_ids)}
        try:
            step_losses, tokens = _run_window_with_optimizer(
                model,
                cfg,
                loop,
                plan,
                readers.__getitem__,
                optimizer,
                window=window,
                comm=comm,
                device=device_t,
                global_inner_step0=global_inner_step,
                tokens_consumed0=tokens_consumed,
            )
        finally:
            for reader in readers.values():
                reader.close()
        finals.append(step_losses[-1])
        global_inner_step += plan.inner_steps
        tokens_consumed += tokens
    return finals


def run_adam_ab(
    n_windows: int,
    cfg: RunConfig,
    manifest: RunManifest,
    *,
    model: MoKTransformer,
    shard_path: Callable[[int], Path],
    k: int = DEFAULT_K,
    threshold_nats: float = DEFAULT_THRESHOLD_NATS,
    uid: int = 0,
    start_window: int = 0,
    device: str | torch.device = "cpu",
) -> ABReport:
    """The full A/B: both arms from bitwise-identical θ_start over the same
    windows/seeds/data; report + recommendation per the decision rule."""
    if n_windows <= 0:
        raise ValueError(f"n_windows must be positive, got {n_windows}")
    if k <= 1:
        raise ValueError(f"k must be > 1 for a meaningful A/B, got {k}")
    common: dict[str, object] = {
        "n_windows": n_windows,
        "shard_path": shard_path,
        "uid": uid,
        "start_window": start_window,
        "device": device,
    }
    losses_1 = run_arm(copy.deepcopy(model), cfg, manifest, reset_every=1, **common)  # type: ignore[arg-type]
    losses_k = run_arm(copy.deepcopy(model), cfg, manifest, reset_every=k, **common)  # type: ignore[arg-type]
    delta = losses_1[-1] - losses_k[-1]
    keep = delta < threshold_nats
    return ABReport(
        n_windows=n_windows,
        k=k,
        threshold_nats=threshold_nats,
        losses_reset_every_window=tuple(losses_1),
        losses_reset_every_k=tuple(losses_k),
        delta_final_loss=delta,
        keep_reset_every_window=keep,
        recommendation=(
            "inner.adam_reset_every_windows=1"
            if keep
            else f"inner.adam_reset_every_windows={k}"
        ),
    )
