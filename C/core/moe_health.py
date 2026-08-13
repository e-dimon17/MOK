"""Post-step MoE health: MXFP8 requant + aux-free bias update + capacity watchdog.

Runs once after every optimizer step, in a FIXED order (consensus-relevant —
the balance-bias buffers are in the state_root domain):

  1. mok backend only: re-quantize every layer's routed weights from the
     freshly-stepped BF16 masters (`MXFP8WeightManager.requantize_`, lazy
     `mok` import inside quant.py). On the reference backend this is a no-op —
     `mok` is never imported and `quant_cache` stays None, so the surrounding
     code path is identical on CPU tests and B300 nodes.
  2. Per layer, in layer order: `router.update_balance_bias_(load, rate)` with
     `rate = cfg.model.bias_update_rate` — the DeepSeek-V3-style aux-free sign
     nudge, deterministic given identical loads.
  3. Capacity-utilization telemetry (NEVER acts locally — capacity anneals
     arrive as manifest amendments).

Capacity-utilization formula (documented contract):

    Experts are EP-sharded in contiguous blocks: rank r hosts experts
    [r*E_local, (r+1)*E_local) with E_local = E / ep_size (MoK layout), using
    the PROTOCOL EP geometry from cfg.model — a validator running the ep=1
    reference replica computes the same util a miner sees on real hardware.

    For one layer with GLOBAL dispatch counts load[E] accumulated over M
    kernel launches (sum == M * tokens_per_launch * ep_size * top_k):
        rows_r     = sum(load[r*E_local : (r+1)*E_local])     rows destined to rank r
        factor     = max(2, ceil(ep_size * capacity_multiplier))   MoK's own formula
        capacity   = tokens_per_launch * top_k * factor * M    rows MoK provisions
        util_layer = max_r rows_r / capacity
    util(step) = max over layers; ~1/factor at perfect balance (0.111 at
    cm=1.05), 1.0 is the GPU trap. Compare against
    cfg.mok.capacity_anneal_util_threshold (0.4) directly.
    `capacity_alert(threshold)` reports whether the RUNNING MAX util has
    reached `threshold` (>=) — the operator signal to propose a capacity
    amendment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from mok_core.config import RunConfig
from mok_core.model import MoKTransformer, MXFP8WeightManager

__all__ = ["MoeHealth"]


class MoeHealth:
    """Owns the post-optimizer-step MoE maintenance for one model instance.

    `capacity_multiplier` defaults to `cfg.mok.schedule_capacity_multiplier`;
    the inner loop passes the phase-resolved value (manifest amendments anneal
    it over the run).
    """

    def __init__(
        self,
        model: MoKTransformer,
        cfg: RunConfig,
        *,
        capacity_multiplier: float | None = None,
        tokens_per_launch: int | None = None,
    ) -> None:
        if model.cfg.num_experts != cfg.model.num_experts:
            raise ValueError(
                f"model has {model.cfg.num_experts} experts but cfg.model declares "
                f"{cfg.model.num_experts}"
            )
        self._model = model
        self._cfg = cfg
        self._layers = model.moe_layers()
        self._quant = MXFP8WeightManager(self._layers)
        cm = cfg.mok.schedule_capacity_multiplier if capacity_multiplier is None else capacity_multiplier
        if cm <= 0.0:
            raise ValueError(f"capacity_multiplier must be positive, got {cm}")
        self._capacity_multiplier = float(cm)
        self._num_experts = cfg.model.num_experts
        self._ep_size = cfg.model.ep_size
        self._local_experts = cfg.model.num_local_experts
        self._top_k = cfg.model.top_k
        self._tokens_per_launch = (
            cfg.window.tokens_per_rank_microbatch if tokens_per_launch is None else int(tokens_per_launch)
        )
        if self._tokens_per_launch <= 0:
            raise ValueError(f"tokens_per_launch must be positive, got {self._tokens_per_launch}")
        self._max_util = 0.0

    # -- properties ----------------------------------------------------------

    @property
    def capacity_multiplier(self) -> float:
        return self._capacity_multiplier

    @property
    def max_util(self) -> float:
        """Running max of per-step utilization since construction."""
        return self._max_util

    # -- the post-step hook --------------------------------------------------

    def post_step(self, router_loads: Sequence[torch.Tensor], *, microbatches: int = 1) -> float:
        """Run requant + bias updates + capacity tracking for one inner step.

        `router_loads` is the PER-LAYER int64 [E] dispatch counts of the step
        (summed over the step's accum microbatches), in layer order. Returns
        this step's utilization (max over layers, see module docstring).
        """
        if len(router_loads) != len(self._layers):
            raise ValueError(f"expected {len(self._layers)} per-layer loads, got {len(router_loads)}")
        if self._model.backend == "mok":
            for layer in self._layers:
                self._quant.requantize_(layer)

        rate = self._cfg.model.bias_update_rate
        util = 0.0
        for i, (layer, load) in enumerate(zip(self._layers, router_loads, strict=True)):
            if load.shape != (self._num_experts,):
                raise ValueError(
                    f"layer {i}: load shape {tuple(load.shape)} != ({self._num_experts},)"
                )
            layer.router.update_balance_bias_(load, rate)
            util = max(util, self._layer_util(load, microbatches))
        self._max_util = max(self._max_util, util)
        return util

    def capacity_alert(self, threshold: float) -> bool:
        """True once the running max utilization has reached `threshold`."""
        return self._max_util >= threshold

    # -- internals -----------------------------------------------------------

    def _layer_util(self, load: torch.Tensor, microbatches: int = 1) -> float:
        """Fraction of MoK's schedule capacity used by the hottest rank.

        `load` is the GLOBAL dispatch count (all EP ranks) accumulated over
        `microbatches` kernel launches. MoK provisions, per launch,
            capacity = num_local_tokens * top_k * max(2, ceil(ep_size * cm))
        rows (mok/functional.py:196,206) and TRAPS the GPU once the padded row
        count exceeds it, so utilisation is measured against exactly that:
        ~1/factor at perfect balance, 1.0 == the trap. (The previous formula
        returned a max/mean SKEW ratio bounded below by 1/cm, which could never
        be compared against the configured 0.4 threshold.)
        """
        rows_per_rank = (
            load.detach()
            .to(device="cpu", dtype=torch.float64)
            .reshape(self._ep_size, self._local_experts)
            .sum(dim=1)
        )
        if float(rows_per_rank.sum()) <= 0.0:
            return 0.0
        factor = max(2, math.ceil(self._ep_size * self._capacity_multiplier))
        capacity = self._tokens_per_launch * self._top_k * factor * max(1, int(microbatches))
        return float(rows_per_rank.max()) / capacity
