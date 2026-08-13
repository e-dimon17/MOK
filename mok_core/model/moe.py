"""The MoK boundary: autograd wrapper + BF16 master params + reference backend.

This is the ONLY module (with quant.py) that touches the `mok` package, and
always lazily inside functions — the SM103-only wheel never loads on CPU hosts.
Wrapper design per mok-training-subnet-report §8.3 and the mixture-of-kittens
README/functional docstrings (Apache-2.0, github.com/cursor/mixture-of-kittens):

  - `build_schedule` runs per forward (routing changes every microbatch);
    schedule + forward_context are reused fwd->bwd.
  - The MXFP8 backward mutates the saved fp8 activations IN PLACE, so a
    forward context is single-use: a consumed-flag guard raises on any second
    backward through the same forward.
  - `functional.backward` returns 8 grads; d_x and d_router_weights flow back
    through autograd (the router is ordinary PyTorch), the 6 weight grads are
    accumulated straight into `param.grad`.

The `reference` backend is the pure-PyTorch replica of the kernel math
(mirrors mixture-of-kittens/tests/utils.py `run_reference_bf16` for the
single-rank case). Reference models are constructed with ep_size forced to 1
so they hold ALL experts — scoring validators and parity tests use this.

Parameter naming: routed expert params are registered as `routed_gate/up/down`
so every expert-local tensor's qualified name contains ".routed_" — the
ownership marker consumed by `is_expert_local` (ZeRO-1 bucketing and the outer
step partition on it). MoK-doc-style aliases (`w_routed_gate`, ...) are
provided as read-only properties.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mok_core.config import ModelConfig, MoKRuntimeConfig

from .quant import QuantizedRoutedWeights
from .router import Router, RouterOutput

_BACKENDS = ("mok", "reference")

EXPERT_MARKER = ".routed_"


def is_expert_local(name: str) -> bool:
    """True iff `name` is an expert-sharded (EP-local) parameter.

    Everything else (attention, norms, router, shared expert, embeddings, LM
    head, balance biases) is replicated across the EP group.
    """
    return EXPERT_MARKER in name


class _MoKFunction(torch.autograd.Function):
    """Manual-API bridge: mok.functional.forward/backward as one autograd node."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        top_experts: torch.Tensor,
        layer: MoKMoELayer,
    ) -> torch.Tensor:
        from mok import functional  # noqa: PLC0415 — SM103-only wheel, lazy by design

        mok_cfg = layer._functional_config()
        workspace = layer._workspace(x, top_experts.shape[1])
        schedule = functional.build_schedule(
            workspace, mok_cfg, top_experts, num_local_experts=layer.num_local_experts
        )
        gate_w, up_w, down_w = layer._forward_weight_args()
        output, fwd_ctx = functional.forward(
            mok_cfg,
            workspace,
            schedule,
            x,
            router_weights,
            layer.shared_gate,
            layer.shared_up,
            layer.shared_down,
            gate_w,
            up_w,
            down_w,
        )
        ctx.save_for_backward(x, router_weights)
        ctx.mok_state = (mok_cfg, workspace, schedule, fwd_ctx, layer)
        ctx.mok_consumed = False
        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, None, None]:
        if ctx.mok_consumed:
            raise RuntimeError(
                "MoK forward context is single-use: the MXFP8 backward mutates saved "
                "activations in place. Second backward through the same forward is invalid "
                "(retain_graph / double-backward are unsupported on the MoE layer)."
            )
        ctx.mok_consumed = True
        from mok import functional  # noqa: PLC0415

        x, router_weights = ctx.saved_tensors
        mok_cfg, workspace, schedule, fwd_ctx, layer = ctx.mok_state
        gate_w, up_w, down_w = layer._backward_weight_args()
        (
            d_x,
            d_router_weights,
            d_routed_gate,
            d_routed_up,
            d_routed_down,
            d_shared_gate,
            d_shared_up,
            d_shared_down,
        ) = functional.backward(
            mok_cfg,
            workspace,
            schedule,
            fwd_ctx,
            grad_output.contiguous(),
            x,
            router_weights,
            layer.shared_gate,
            layer.shared_up,
            layer.shared_down,
            gate_w,
            up_w,
            down_w,
        )
        layer._accumulate_weight_grads(
            d_routed_gate, d_routed_up, d_routed_down, d_shared_gate, d_shared_up, d_shared_down
        )
        return d_x, d_router_weights, None, None


class MoKMoELayer(nn.Module):
    """One MoE layer: shared expert + EP-sharded routed experts + fp32 router.

    BF16 master parameters (state_root domain):
      shared_gate [I, H] · shared_up [I, H] · shared_down [H, I]
      routed_gate [E_local, I, H] · routed_up [E_local, I, H] · routed_down [E_local, H, I]
    plus `router.proj.weight` (fp32) and `router.balance_bias` (fp32 buffer).
    E_local = num_experts // ep_size. `quant_cache` (MXFP8 copies) is derived
    data — a plain attribute, invisible to state_dict and state_root.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        mok_runtime: MoKRuntimeConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.mok_runtime = mok_runtime if mok_runtime is not None else MoKRuntimeConfig()
        self.num_local_experts = cfg.num_local_experts
        hidden, inter, e_local = cfg.hidden_size, cfg.intermediate_size, cfg.num_local_experts
        self.shared_gate = nn.Parameter(torch.empty(inter, hidden, dtype=dtype))
        self.shared_up = nn.Parameter(torch.empty(inter, hidden, dtype=dtype))
        self.shared_down = nn.Parameter(torch.empty(hidden, inter, dtype=dtype))
        self.routed_gate = nn.Parameter(torch.empty(e_local, inter, hidden, dtype=dtype))
        self.routed_up = nn.Parameter(torch.empty(e_local, inter, hidden, dtype=dtype))
        self.routed_down = nn.Parameter(torch.empty(e_local, hidden, inter, dtype=dtype))
        self.router = Router(cfg)
        self.quant_cache: QuantizedRoutedWeights | None = None
        self._mok_cfg_cache: Any = None

    # MoK-doc-style aliases (read-only) --------------------------------------
    @property
    def w_shared_gate(self) -> nn.Parameter:
        return self.shared_gate

    @property
    def w_shared_up(self) -> nn.Parameter:
        return self.shared_up

    @property
    def w_shared_down(self) -> nn.Parameter:
        return self.shared_down

    @property
    def w_routed_gate(self) -> nn.Parameter:
        return self.routed_gate

    @property
    def w_routed_up(self) -> nn.Parameter:
        return self.routed_up

    @property
    def w_routed_down(self) -> nn.Parameter:
        return self.routed_down

    # ------------------------------------------------------------------------

    def forward(self, x: torch.Tensor, backend: str) -> tuple[torch.Tensor, RouterOutput]:
        """x: bf16 [T, H] -> (y bf16 [T, H], RouterOutput). Router runs outside MoK."""
        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        route = self.router(x)
        if backend == "reference":
            y = self._reference_forward(x, route)
        else:
            y = _MoKFunction.apply(
                x.contiguous(), route.weights.contiguous(), route.experts.contiguous(), self
            )
        return y, route

    # -- reference backend ---------------------------------------------------

    def _reference_forward(self, x: torch.Tensor, route: RouterOutput) -> torch.Tensor:
        """Pure-PyTorch replica of the kernel math (single-rank run_reference_bf16).

        Requires ep_size == 1 (all experts local): build reference models via
        `mok_core.model.reference.build_reference_model`.
        """
        if self.num_local_experts != self.cfg.num_experts:
            raise RuntimeError(
                "reference backend requires all experts local (ep_size == 1); "
                f"this layer holds {self.num_local_experts}/{self.cfg.num_experts} experts"
            )
        num_tokens, hidden = x.shape
        topk = route.experts.shape[1]
        flat_experts = route.experts.reshape(-1)                       # [T*topk]
        token_idx = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)

        flat_output = torch.zeros(num_tokens * topk, hidden, dtype=x.dtype, device=x.device)
        for expert_idx in range(self.num_local_experts):
            rows = (flat_experts == expert_idx).nonzero().flatten()
            expert_x = x[token_idx[rows]]
            gate = expert_x @ self.routed_gate[expert_idx].T
            up = expert_x @ self.routed_up[expert_idx].T
            hidden_act = torch.nn.functional.silu(gate) * up
            flat_output = flat_output.index_copy(
                0, rows, hidden_act @ self.routed_down[expert_idx].T
            )
        routed_output = (
            flat_output.view(num_tokens, topk, hidden).float() * route.weights.unsqueeze(2)
        ).sum(1)

        gate_shared = x @ self.shared_gate.T
        up_shared = x @ self.shared_up.T
        shared_output = (torch.nn.functional.silu(gate_shared) * up_shared) @ self.shared_down.T

        return (routed_output + shared_output.float()).to(x.dtype)

    # -- mok backend plumbing ------------------------------------------------

    def _functional_config(self) -> Any:
        """Cached mok.functional.MoKConfig built from MoKRuntimeConfig (lazy import)."""
        if self._mok_cfg_cache is None:
            self._mok_cfg_cache = self.mok_runtime.to_mok()
        return self._mok_cfg_cache

    def _workspace(self, x: torch.Tensor, topk: int) -> Any:
        """Model-wide cached symmetric-memory workspace (get_workspace caches by key)."""
        import torch.distributed as dist  # noqa: PLC0415
        from mok import functional  # noqa: PLC0415

        return functional.get_workspace(
            self._functional_config(),
            dist.group.WORLD,
            device=x.device,
            num_local_tokens=x.shape[0],
            hidden_size=x.shape[1],
            topk=topk,
        )

    def _forward_weight_args(self) -> tuple[Any, Any, Any]:
        """Routed weight args for functional.forward: (fp8, sc) pairs or BF16 masters."""
        if self.cfg.routed_precision == "bf16":
            return self.routed_gate, self.routed_up, self.routed_down
        if self.quant_cache is None:
            raise RuntimeError(
                "MXFP8 cache missing — run MXFP8WeightManager.requantize_ after "
                "init / checkpoint load and after every optimizer step"
            )
        return self.quant_cache.forward_args()

    def _backward_weight_args(self) -> tuple[Any, Any, Any]:
        """Routed weight args for functional.backward: 4-tuples gate/up + transposed pair down."""
        if self.cfg.routed_precision == "bf16":
            return self.routed_gate, self.routed_up, self.routed_down
        if self.quant_cache is None:
            raise RuntimeError("MXFP8 cache missing at backward — requantize_ was never called")
        return self.quant_cache.backward_args()

    def _accumulate_weight_grads(
        self,
        d_routed_gate: torch.Tensor,
        d_routed_up: torch.Tensor,
        d_routed_down: torch.Tensor,
        d_shared_gate: torch.Tensor,
        d_shared_up: torch.Tensor,
        d_shared_down: torch.Tensor,
    ) -> None:
        """Accumulate kernel-produced weight grads into param.grad (fixed order)."""
        pairs = (
            (self.routed_gate, d_routed_gate),
            (self.routed_up, d_routed_up),
            (self.routed_down, d_routed_down),
            (self.shared_gate, d_shared_gate),
            (self.shared_up, d_shared_up),
            (self.shared_down, d_shared_down),
        )
        for param, grad in pairs:
            if param.grad is None:
                param.grad = grad
            else:
                param.grad.add_(grad)
