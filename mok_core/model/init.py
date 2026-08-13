"""Deterministic model construction — one init, replayed identically everywhere.

`init_model(cfg, seed)` is a pure function of its arguments: `seed_everything`
pins all RNG, then parameters are initialized in a FIXED layer order (module
definition order), each drawn into an fp32 staging buffer and copied into the
master dtype — so the draw sequence is independent of parameter dtype. Two
processes calling with the same seed produce identical state_roots (the
publisher's seed-42 init in step B relies on this).

Scheme: truncated normal (±2σ) std 0.02 for embeddings, QKV, gate/up and the
LM head; std 0.02/sqrt(2·num_layers) for residual-facing projections (attention
out-proj, MoE down-proj); fp32 normal std 0.006 for the router; zeros for
balance_bias; ones for norms.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mok_core.config import ModelConfig, MoKRuntimeConfig
from mok_core.determinism import seed_everything

from .transformer import MoKTransformer

ROUTER_INIT_STD = 0.006
BASE_INIT_STD = 0.02


def _fill_trunc_normal(param: torch.Tensor, std: float) -> None:
    """Draw trunc-normal(0, std, ±2σ) in fp32, then copy into the param's dtype."""
    staging = torch.empty(param.shape, dtype=torch.float32, device=param.device)
    nn.init.trunc_normal_(staging, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    with torch.no_grad():
        param.copy_(staging)


def init_model(
    cfg: ModelConfig,
    seed: int,
    device: str | torch.device = "cpu",
    backend: str = "reference",
    mok_runtime: MoKRuntimeConfig | None = None,
) -> MoKTransformer:
    """Seed, build and deterministically initialize a MoKTransformer.

    On device='meta' the structure is built but no values are written (shape /
    memory-planning use only — a meta model has no defined state_root).
    """
    seed_everything(seed)
    dev = torch.device(device)
    with torch.device(dev):
        model = MoKTransformer(cfg, backend=backend, mok_runtime=mok_runtime)
    if dev.type == "meta":
        return model

    residual_std = BASE_INIT_STD / math.sqrt(2.0 * cfg.num_layers)
    with torch.no_grad():
        _fill_trunc_normal(model.embed.weight, BASE_INIT_STD)
        for block in model.blocks:
            block.attn_norm.weight.fill_(1.0)
            _fill_trunc_normal(block.attn.qkv.weight, BASE_INIT_STD)
            _fill_trunc_normal(block.attn.o_proj.weight, residual_std)
            block.moe_norm.weight.fill_(1.0)
            moe = block.moe
            if block.is_dense:
                _fill_trunc_normal(moe.w_gate.weight, BASE_INIT_STD)
                _fill_trunc_normal(moe.w_up.weight, BASE_INIT_STD)
                _fill_trunc_normal(moe.w_down.weight, residual_std)
                continue
            _fill_trunc_normal(moe.shared_gate, BASE_INIT_STD)
            _fill_trunc_normal(moe.shared_up, BASE_INIT_STD)
            _fill_trunc_normal(moe.shared_down, residual_std)
            _fill_trunc_normal(moe.routed_gate, BASE_INIT_STD)
            _fill_trunc_normal(moe.routed_up, BASE_INIT_STD)
            _fill_trunc_normal(moe.routed_down, residual_std)
            moe.router.proj.weight.normal_(mean=0.0, std=ROUTER_INIT_STD)
            moe.router.balance_bias.zero_()
        model.final_norm.weight.fill_(1.0)
        _fill_trunc_normal(model.lm_head.weight, BASE_INIT_STD)
    return model
