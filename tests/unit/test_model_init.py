"""init_model determinism: identical seeds -> identical state_root, cross-process by design."""

from __future__ import annotations

import torch

from mok_core.config import ModelConfig
from mok_core.determinism import hash_named_tensors
from mok_core.model import init_model


def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        num_layers=4,
        num_dense_layers=0,
        hidden_size=512,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        vocab_size=512,
        seq_len=256,
        num_experts=8,
        top_k=2,
        intermediate_size=256,
        ep_size=4,
    ).model_copy(update={"ep_size": 1})


def test_same_seed_same_state_root() -> None:
    cfg = tiny_cfg()
    root_a = hash_named_tensors(init_model(cfg, seed=42).iter_master_params())
    root_b = hash_named_tensors(init_model(cfg, seed=42).iter_master_params())
    assert root_a == root_b


def test_different_seed_different_state_root() -> None:
    cfg = tiny_cfg()
    root_a = hash_named_tensors(init_model(cfg, seed=42).iter_master_params())
    root_b = hash_named_tensors(init_model(cfg, seed=43).iter_master_params())
    assert root_a != root_b


def test_init_values_follow_scheme() -> None:
    cfg = tiny_cfg()
    model = init_model(cfg, seed=1)
    # balance biases start at exactly zero
    for block in model.blocks:
        assert torch.equal(block.moe.router.balance_bias, torch.zeros(cfg.num_experts))
        # norms start at ones
        assert torch.equal(block.attn_norm.weight, torch.ones_like(block.attn_norm.weight))
        assert torch.equal(block.moe_norm.weight, torch.ones_like(block.moe_norm.weight))
    assert torch.equal(model.final_norm.weight, torch.ones_like(model.final_norm.weight))
    # router fp32 normal(0, 0.006): loose std check on ~131k draws
    router_w = model.blocks[0].moe.router.proj.weight.detach()
    assert router_w.dtype == torch.float32
    assert 0.004 < float(router_w.std()) < 0.008
    # trunc-normal(0.02, +-2 sigma): bounded up to bf16 rounding, std in the ballpark
    bf16_slack = 1 + 2**-7
    emb = model.embed.weight.detach().float()
    assert float(emb.abs().max()) <= 0.04 * bf16_slack
    assert 0.01 < float(emb.std()) < 0.03
    # residual-facing projections are scaled down by sqrt(2*num_layers)
    down = model.blocks[0].moe.shared_down.detach().float()
    assert float(down.abs().max()) <= 2 * 0.02 / (2 * cfg.num_layers) ** 0.5 * bf16_slack


def test_meta_device_init_skips_values() -> None:
    model = init_model(tiny_cfg(), seed=5, device="meta")
    assert all(p.device.type == "meta" for p in model.parameters())
