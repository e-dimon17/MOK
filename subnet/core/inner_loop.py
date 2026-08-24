"""The inner training loop of one window — H inner steps, bitwise replayable.

`InnerLoop.run_window` executes the training side of the window protocol
: a fixed `WindowBatchPlan` drives every microbatch, a FRESH
`Zero1Adam` is constructed for the window (protocol decision #1), gradients
are reduced with the explicit fixed-order `flat_grad_all_reduce` (no DDP
hooks), the global grad norm is clipped in fp32, the LR is the closed-form
`lr_at` of the global inner step, and `MoeHealth.post_step` runs after every
step (requant on mok backend + balance-bias update + capacity telemetry).

Determinism contract:
  - No RNG anywhere in the loop (the model has no dropout; data order is
    pinned by the plan). Two runs from identical θ_start produce identical
    state_roots — the CPU determinism gate (test_inner_loop.py) and the GPU
    launch gate both assert exactly this.
  - The only collectives per inner step are the FIXED reductions: one flat
    fp32 all-reduce over replicated grads and one 1-element all-reduce for
    the expert-local grad-norm contribution. Nothing else touches the wire
    mid-window.
  - bf16 autocast wraps forward/backward on CUDA; on CPU the model runs its
    native dtypes (plain fp32/bf16 math) — same code path via nullcontext.

Next-token targets keep the kernel token-count invariant: the model always
sees the full [B, S] microbatch (MoK requires T = B*S >= 512 and % 256 == 0),
and targets are the within-sequence shift with the final position set to
-100 — `loss_head`'s cross-entropy ignores it (F.cross_entropy default
ignore_index), so each sequence contributes S-1 supervised positions.

`null_round=True` (warmup windows for late joiners) runs EVERYTHING —
forward, backward, reductions, clip, MoeHealth — but skips `Zero1Adam.step`,
so the node exercises the full hot path without moving trainable weights
(balance biases still evolve; the caller discards/restores θ afterwards).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from mok_core.config import RunConfig
from mok_core.data import ShardReader, WindowBatchPlan
from mok_core.model import MoKTransformer, RouterOutput, loss_head
from subnet.core.moe_health import MoeHealth
from subnet.core.phase import PhaseConfig, accum_at, lr_at
from subnet.core.zero1 import Comm, Zero1Adam, flat_grad_all_reduce

__all__ = ["IGNORE_INDEX", "InnerLoop", "WindowResult"]

IGNORE_INDEX = -100  # F.cross_entropy's default ignore_index (used by loss_head)

_NORM_EPS = 1e-6  # torch.nn.utils.clip_grad_norm_ convention


@dataclass(frozen=True)
class WindowResult:
    """Telemetry + accounting of one window run (all consensus-free floats).

    entry_loss:  mean total loss of inner step 0 (computed at θ_start).
    mean_loss:   mean over inner steps of the per-step mean total loss.
    final_loss:  per-step mean total loss of the last inner step.
    tokens:      tokens consumed by this miner across ALL ranks this window.
    grad_norm_mean:     mean over steps of the PRE-clip fp32 global grad norm.
    capacity_util_max:  max over steps of MoeHealth utilization.
    router_entropy_mean: mean over steps/microbatches/layers of the token-mean
                         router softmax entropy (nats).
    expert_load: int64 [E] CPU — dispatch counts summed over layers and steps.
    global_inner_steps_done: the global inner-step counter AFTER this window
                             (= global_inner_step0 + plan.inner_steps).
    """

    entry_loss: float
    mean_loss: float
    final_loss: float
    tokens: int
    grad_norm_mean: float
    capacity_util_max: float
    router_entropy_mean: float
    expert_load: torch.Tensor
    global_inner_steps_done: int


class InnerLoop:
    """Runs windows for one (model, cfg, phase) on one rank. Stateless between
    windows — everything a window needs arrives through `run_window`."""

    def __init__(
        self,
        model: MoKTransformer,
        cfg: RunConfig,
        phase: PhaseConfig,
        *,
        rank: int,
        world_size: int,
        comm: Comm,
        device: str | torch.device,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")
        self.model = model
        self.cfg = cfg
        self.phase = phase
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        self.device = torch.device(device)

    # ------------------------------------------------------------------ #

    def run_window(
        self,
        plan: WindowBatchPlan,
        shard_lookup: Callable[[int], ShardReader],
        window: int,
        global_inner_step0: int,
        tokens_consumed0: int,
        null_round: bool = False,
    ) -> WindowResult:
        """Execute one full window per the module-docstring contract."""
        self._check_plan(plan, window)
        named_params = dict(self.model.named_parameters())
        replicated = {
            name: p for name, p in named_params.items() if not self.model.is_expert_local(name)
        }
        optimizer = Zero1Adam.fresh(
            named_params,
            self.cfg.inner,
            rank=self.rank,
            world_size=self.world_size,
            is_expert_local=self.model.is_expert_local,
            comm=self.comm,
        )
        health = MoeHealth(
            self.model,
            self.cfg,
            capacity_multiplier=self.phase.capacity_multiplier,
            tokens_per_launch=self.phase.tokens_per_rank_microbatch,
        )

        num_layers = len(self.model.moe_layers())
        num_experts = self.cfg.model.num_experts

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        seqs = plan.seqs_per_microbatch
        tokens_consumed = tokens_consumed0
        step_losses: list[float] = []
        grad_norms: list[float] = []
        entropies: list[float] = []
        window_load = torch.zeros(num_experts, dtype=torch.int64)

        for s in range(plan.inner_steps):
            # ramp value, clamped into [1, plan.grad_accum]: the plan's schedule
            # dimension is the ceiling, and a step always runs >= 1 microbatch
            accum = max(1, min(accum_at(self.cfg.window, tokens_consumed), plan.grad_accum))
            for p in named_params.values():
                p.grad = None
            step_loads = [torch.zeros(num_experts, dtype=torch.int64) for _ in range(num_layers)]
            loss_sum = 0.0
            entropy_sum = 0.0

            for m in range(accum):
                tokens = plan.microbatch_tokens(s, m, shard_lookup).to(self.device)
                inputs = tokens.view(seqs, plan.seq_len)
                targets = torch.full_like(inputs, IGNORE_INDEX)
                targets[:, :-1] = inputs[:, 1:]
                with autocast:
                    output = self.model(inputs)
                    losses = loss_head(output.logits, targets, output.loss_inputs, self.cfg.model)
                (losses.total / accum).backward()
                loss_sum += float(losses.total.detach())
                with torch.no_grad():
                    for i, stats in enumerate(output.loss_inputs):
                        step_loads[i] += stats.load.detach().cpu()
                    entropy_sum += self._router_entropy(output.loss_inputs)

            flat_grad_all_reduce(replicated, self.comm, self.world_size)
            grad_norms.append(self._clip_grad_norm(named_params))
            lr = lr_at(self.phase.lr, global_inner_step0 + s, self.cfg.tokens_per_inner_step)
            if not null_round:
                optimizer.step(lr)
            step_loads = self._all_reduce_loads(step_loads)
            health.post_step(step_loads, microbatches=accum)

            for load in step_loads:
                window_load += load
            step_losses.append(loss_sum / accum)
            entropies.append(entropy_sum / accum)
            tokens_consumed += accum * plan.tokens_per_rank_microbatch * self.world_size

        return WindowResult(
            entry_loss=step_losses[0],
            mean_loss=sum(step_losses) / len(step_losses),
            final_loss=step_losses[-1],
            tokens=tokens_consumed - tokens_consumed0,
            grad_norm_mean=sum(grad_norms) / len(grad_norms),
            capacity_util_max=health.max_util,
            router_entropy_mean=sum(entropies) / len(entropies),
            expert_load=window_load,
            global_inner_steps_done=global_inner_step0 + plan.inner_steps,
        )

    # ------------------------------------------------------------------ #

    def _check_plan(self, plan: WindowBatchPlan, window: int) -> None:
        checks = (
            ("window", plan.window, window),
            ("rank", plan.rank, self.rank),
            ("world_size", plan.world_size, self.world_size),
            ("inner_steps", plan.inner_steps, self.phase.inner_steps),
            ("grad_accum", plan.grad_accum, self.phase.grad_accum),
            ("seq_len", plan.seq_len, self.phase.seq_len),
            (
                "tokens_per_rank_microbatch",
                plan.tokens_per_rank_microbatch,
                self.phase.tokens_per_rank_microbatch,
            ),
        )
        for field, got, want in checks:
            if got != want:
                raise ValueError(f"plan.{field} = {got} does not match expected {want}")

    def _clip_grad_norm(self, named_params: dict[str, torch.nn.Parameter]) -> float:
        """fp32 global grad-norm clip at cfg.inner.grad_clip. Returns the
        PRE-clip norm.

        The norm spans all ranks: expert-local squared sums are summed with a
        1-element all-reduce (the second fixed reduction), replicated squared
        sums are computed identically on every rank from the already-reduced
        grads. Accumulation walks sorted names — fixed order, every rank
        computes the same coefficient and scales bitwise identically.
        """
        expert_sq = torch.zeros(1, dtype=torch.float32, device=self.device)
        replicated_sq = torch.zeros(1, dtype=torch.float32, device=self.device)
        for name in sorted(named_params):
            grad = named_params[name].grad
            if grad is None:
                continue
            sq = grad.detach().to(torch.float32).pow(2).sum()
            if self.model.is_expert_local(name):
                expert_sq += sq
            else:
                replicated_sq += sq
        self.comm.all_reduce(expert_sq)
        total_norm = float(torch.sqrt(expert_sq + replicated_sq))

        clip = self.cfg.inner.grad_clip
        clip_coef = clip / (total_norm + _NORM_EPS)
        if clip_coef < 1.0:
            with torch.no_grad():
                for name in sorted(named_params):
                    grad = named_params[name].grad
                    if grad is not None:
                        grad.mul_(clip_coef)
        return total_norm

    def _all_reduce_loads(self, step_loads: list[torch.Tensor]) -> list[torch.Tensor]:
        """Sum the per-layer expert dispatch counts across the EP ranks.

        Each rank only routes its own microbatch tokens, so without this the
        aux-free balance bias is driven by one rank's sample: the `balance_bias`
        buffers (a state_root member, never broadcast by Zero1Adam because they
        are buffers rather than parameters) silently diverge across the 8 ranks,
        and the capacity metric measures 1/ep_size of the real dispatch. Counts
        are exact integers and fp64 represents them exactly below 2**53, so the
        sum is bitwise deterministic. One 32 KB reduction per inner step.
        """
        if self.world_size == 1:
            return step_loads
        flat = torch.cat([load.reshape(-1) for load in step_loads]).to(
            device=self.device, dtype=torch.float64
        )
        self.comm.all_reduce(flat)
        num_experts = self.cfg.model.num_experts
        return [chunk.round().to(torch.int64).cpu() for chunk in flat.split(num_experts)]

    @staticmethod
    def _router_entropy(loss_inputs: tuple[RouterOutput, ...]) -> float:
        """Mean over layers of the token-mean router softmax entropy (nats)."""
        total = 0.0
        for stats in loss_inputs:
            probs = torch.softmax(stats.router_logits.detach().float(), dim=-1)
            total += float(-(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean())
        return total / len(loss_inputs)
