"""MoK-54B transformer: embedding, N pre-norm blocks (attention + MoE), LM head.

Everything outside the MoE layer is ordinary PyTorch in BF16 (fp32 router and
fp32-master LM head excepted). The forward runs under the pinned SDPA-backend
context so attention is deterministic on every host.

`iter_master_params()` defines the state_root domain: ALL trainable parameters
plus the (non-trainable) router balance-bias buffers. MXFP8 copies, optimizer
state and error-feedback buffers are excluded by construction.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from mok_core.config import ModelConfig, MoKRuntimeConfig

from .attention import CausalSelfAttention, sdpa_backend
from .moe import MoKMoELayer, is_expert_local
from .router import RouterOutput

COMPILE_ENV = "MOK_COMPILE"


@dataclass(frozen=True)
class ModelOutput:
    """logits: [B, S, V] model dtype (loss head casts fp32);
    loss_inputs: per-layer RouterOutput in layer order (feeds `loss_head` and
    the post-step balance-bias update / capacity telemetry)."""

    logits: torch.Tensor
    loss_inputs: tuple[RouterOutput, ...]

    def total_load(self) -> torch.Tensor:
        """Summed per-expert dispatch counts across layers, int64 [E]."""
        load = self.loss_inputs[0].load.clone()
        for stats in self.loss_inputs[1:]:
            load += stats.load
        return load


class LMHead(nn.Module):
    """Untied LM head with an fp32 master weight, cast to bf16 for the matmul."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight.to(x.dtype))


class DenseSwiGLU(nn.Module):
    """Dense SwiGLU FFN for the first num_dense_layers blocks (no router, no MoK)."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        h, i = cfg.hidden_size, cfg.dense_intermediate_size
        self.w_gate = nn.Linear(h, i, bias=False, dtype=dtype)
        self.w_up = nn.Linear(h, i, bias=False, dtype=dtype)
        self.w_down = nn.Linear(i, h, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(torch.nn.functional.silu(self.w_gate(x)) * self.w_up(x))


class MoKBlock(nn.Module):
    """Pre-norm block: x + attn(norm(x)); x + moe(norm(x)) with [B,S,H]->[T,H] reshape."""

    def __init__(
        self,
        cfg: ModelConfig,
        layer_idx: int,
        rope_theta: float | None = None,
        mok_runtime: MoKRuntimeConfig | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.attn = CausalSelfAttention(cfg, rope_theta=rope_theta, dtype=dtype)
        self.moe_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.is_dense = layer_idx < cfg.num_dense_layers
        if self.is_dense:
            self.moe = DenseSwiGLU(cfg, dtype=dtype)
        else:
            self.moe = MoKMoELayer(cfg, layer_idx, mok_runtime=mok_runtime, dtype=dtype)


    def forward(self, x: torch.Tensor, backend: str) -> tuple[torch.Tensor, RouterOutput]:
        x = x + self.attn(self.attn_norm(x))
        batch, seq, hidden = x.shape
        tokens = batch * seq
        if backend == "mok" and (tokens < 512 or tokens % 256 != 0):
            raise ValueError(
                f"T = B*S = {tokens} violates MoK num_local_tokens rules (>= 512 and % 256 == 0)"
            )
        h = self.moe_norm(x).reshape(tokens, hidden)
        if self.is_dense:
            return x + self.moe(h).view(batch, seq, hidden), None
        y, route = self.moe(h.contiguous(), backend)
        return x + y.view(batch, seq, hidden), route



class MoKTransformer(nn.Module):
    """The full model. backend: 'mok' (B300 megakernel) or 'reference' (pure PyTorch)."""

    def __init__(
        self,
        cfg: ModelConfig,
        backend: str = "reference",
        mok_runtime: MoKRuntimeConfig | None = None,
        rope_theta: float | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if backend not in ("mok", "reference"):
            raise ValueError(f"backend must be 'mok' or 'reference', got {backend!r}")
        self.cfg = cfg
        self.backend = backend
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=dtype)
        self.blocks = nn.ModuleList(
            MoKBlock(cfg, i, rope_theta=rope_theta, mok_runtime=mok_runtime, dtype=dtype)
            for i in range(cfg.num_layers)
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        self.lm_head = LMHead(cfg)
        self._maybe_compile()

    def _maybe_compile(self) -> None:
        """torch.compile per NON-MoE submodule, only for the mok backend and only
        when MOK_COMPILE=1 (never in tests). In-place `Module.compile()` keeps
        parameter names intact — no `_orig_mod.` prefix in the state_root domain."""
        if self.backend != "mok" or os.environ.get(COMPILE_ENV) != "1":
            return
        self.embed.compile()
        for block in self.blocks:
            block.attn_norm.compile()
            block.attn.compile()
            block.moe_norm.compile()
        self.final_norm.compile()
        self.lm_head.compile()

    def forward(self, tokens: torch.Tensor) -> ModelOutput:
        """tokens: int64 [B, S] -> ModelOutput."""
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, S], got {tuple(tokens.shape)}")
        router_outputs: list[RouterOutput] = []
        with sdpa_backend():
            x = self.embed(tokens)
            for block in self.blocks:
                x, route = block(x, self.backend)
                if route is not None:  # dense blocks (first num_dense_layers) have no router
                    router_outputs.append(route)
            x = self.final_norm(x)
            logits = self.lm_head(x)
        return ModelOutput(logits=logits, loss_inputs=tuple(router_outputs))

    # -- protocol surface ----------------------------------------------------

    def iter_master_params(self) -> Iterator[tuple[str, torch.Tensor]]:
        """The state_root domain: all trainable params + balance_bias buffers.

        Hash with `mok_core.determinism.hash_named_tensors(model.iter_master_params())`
        (sorted-name order is applied inside the hasher).
        """
        yield from self.named_parameters()
        for name, buf in self.named_buffers():
            if name.endswith("balance_bias"):
                yield name, buf

    def is_expert_local(self, name: str) -> bool:
        """True iff `name` (from iter_master_params) is EP-sharded on this rank."""
        return is_expert_local(name)

    def param_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: tuple(t.shape) for name, t in self.iter_master_params()}

    def moe_layers(self) -> list[MoKMoELayer]:
        """Layer-order MoE layers (MXFP8WeightManager / balance-bias updates).

        Dense (first num_dense_layers) blocks are excluded — they have no
        router, no experts and no MXFP8 cache."""
        return [block.moe for block in self.blocks if not block.is_dense]
