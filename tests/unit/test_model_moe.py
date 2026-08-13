"""MoKMoELayer reference backend vs the mixture-of-kittens reference MoE math.

The expected values replicate tests/utils.py::run_reference_bf16 for the
single-rank case (identity dispatch), op for op — the comparison is exact.
No `mok` import anywhere in this file (CPU suite).
"""

from __future__ import annotations

import pytest
import torch

from mok_core.config import ModelConfig
from mok_core.model import MoKMoELayer, is_expert_local


def tiny_cfg(ep_size: int = 4) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        num_dense_layers=0,
        hidden_size=256,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=128,
        vocab_size=512,
        seq_len=256,
        num_experts=8,
        top_k=2,
        intermediate_size=256,
        ep_size=ep_size,
    )


def _ref_cfg() -> ModelConfig:
    return tiny_cfg().model_copy(update={"ep_size": 1})  # all experts local


def _randomized_layer(seed: int = 0) -> MoKMoELayer:
    torch.manual_seed(seed)
    layer = MoKMoELayer(_ref_cfg(), layer_idx=0)
    with torch.no_grad():
        for param in layer.parameters():
            param.copy_(torch.randn(param.shape, dtype=torch.float32) * 0.05)
        layer.router.balance_bias.zero_()
    return layer


def test_reference_forward_matches_kernel_reference_math() -> None:
    layer = _randomized_layer()
    cfg = layer.cfg
    num_tokens = 16
    x = (torch.randn(num_tokens, cfg.hidden_size) * 0.1).to(torch.bfloat16)

    y, route = layer(x, backend="reference")
    assert y.shape == (num_tokens, cfg.hidden_size)
    assert y.dtype == torch.bfloat16

    # mixture-of-kittens tests/utils.py math, single rank (identity dispatch):
    with torch.no_grad():
        topk = route.experts.shape[1]
        flat_experts = route.experts.reshape(-1)
        flat_output = torch.zeros(num_tokens * topk, cfg.hidden_size, dtype=torch.bfloat16)
        for expert in range(cfg.num_experts):
            rows = (flat_experts == expert).nonzero().flatten()
            expert_x = x[rows // topk]
            gate = expert_x @ layer.w_routed_gate[expert].T
            up = expert_x @ layer.w_routed_up[expert].T
            hidden = torch.nn.functional.silu(gate) * up
            flat_output[rows] = hidden @ layer.w_routed_down[expert].T
        routed = (
            flat_output.view(num_tokens, topk, cfg.hidden_size).float() * route.weights.unsqueeze(2)
        ).sum(1)
        gate_s = x @ layer.w_shared_gate.T
        up_s = x @ layer.w_shared_up.T
        shared = (torch.nn.functional.silu(gate_s) * up_s) @ layer.w_shared_down.T
        expected = (routed + shared.float()).to(torch.bfloat16)

    torch.testing.assert_close(y.detach(), expected.detach(), rtol=0, atol=0)


def test_reference_backward_produces_grads_for_all_params() -> None:
    layer = _randomized_layer(seed=3)
    x = (torch.randn(32, layer.cfg.hidden_size) * 0.1).to(torch.bfloat16).requires_grad_()
    y, _ = layer(x, backend="reference")
    y.float().pow(2).sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad.float()).all(), f"non-finite grad for {name}"
    assert x.grad is not None and torch.isfinite(x.grad.float()).all()
    assert layer.router.balance_bias.grad is None  # non-trainable buffer


def test_reference_backend_requires_all_experts_local() -> None:
    layer = MoKMoELayer(tiny_cfg(ep_size=4), layer_idx=0)  # holds 2 of 8 experts
    x = torch.zeros(4, 256, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="ep_size == 1"):
        layer(x, backend="reference")


def test_invalid_backend_rejected() -> None:
    layer = _randomized_layer()
    with pytest.raises(ValueError, match="backend"):
        layer(torch.zeros(4, 256, dtype=torch.bfloat16), backend="triton")


def test_expert_ownership_marker() -> None:
    layer = _randomized_layer()
    names = [f"blocks.0.moe.{n}" for n, _ in layer.named_parameters()]
    expert = sorted(n for n in names if is_expert_local(n))
    assert expert == [
        "blocks.0.moe.routed_down",
        "blocks.0.moe.routed_gate",
        "blocks.0.moe.routed_up",
    ]
    for name in names:
        if "routed" not in name:
            assert not is_expert_local(name)
    assert not is_expert_local("blocks.0.moe.router.proj.weight")  # router is replicated
    assert not is_expert_local("blocks.0.moe.shared_gate")


def test_mok_doc_aliases_point_at_master_params() -> None:
    layer = _randomized_layer()
    assert layer.w_routed_gate is layer.routed_gate
    assert layer.w_routed_up is layer.routed_up
    assert layer.w_routed_down is layer.routed_down
    assert layer.w_shared_gate is layer.shared_gate
    assert layer.w_shared_up is layer.shared_up
    assert layer.w_shared_down is layer.shared_down
    assert layer.routed_gate.shape == (8, 256, 256)
    assert layer.routed_down.shape == (8, 256, 256)
    assert layer.routed_gate.dtype == torch.bfloat16


def test_quant_cache_is_not_in_state_dict() -> None:
    layer = _randomized_layer()
    assert layer.quant_cache is None
    assert all("quant" not in k for k in layer.state_dict())
