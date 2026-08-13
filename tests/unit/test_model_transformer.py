"""MoKTransformer end-to-end on CPU (reference backend) — the spec's tiny config."""

from __future__ import annotations

import sys

import pytest
import torch

from mok_core.config import ModelConfig
from mok_core.model import MoKTransformer, loss_head


def tiny_cfg() -> ModelConfig:
    # spec tiny config: 4 layers, H=512 (4 q-heads x 128), 2 kv, E=8 top-2, I=256,
    # ep_size forced 1 via model_copy (all experts local), vocab 512, seq 256
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


def _tiny_model(seed: int = 11) -> MoKTransformer:
    from mok_core.model import init_model  # noqa: PLC0415

    return init_model(tiny_cfg(), seed=seed, backend="reference")


def test_forward_loss_backward_all_params() -> None:
    cfg = tiny_cfg()
    model = _tiny_model()
    batch, seq = 2, 256
    tokens = torch.randint(0, cfg.vocab_size, (batch, seq))
    targets = torch.randint(0, cfg.vocab_size, (batch, seq))

    out = model(tokens)
    assert out.logits.shape == (batch, seq, cfg.vocab_size)
    assert out.logits.dtype == torch.bfloat16
    assert len(out.loss_inputs) == cfg.num_layers

    loss = loss_head(out.logits, targets, out.loss_inputs, cfg)
    assert torch.isfinite(loss.total)
    loss.total.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"no grad for {name}"
        assert torch.isfinite(param.grad.float()).all(), f"non-finite grad for {name}"

    # dispatched load: T*topk tokens per layer, summed over layers
    assert int(out.total_load().sum()) == batch * seq * cfg.top_k * cfg.num_layers


def test_master_param_domain_includes_balance_bias() -> None:
    cfg = tiny_cfg()
    model = _tiny_model()
    master = dict(model.iter_master_params())
    n_params = sum(1 for _ in model.named_parameters())
    bias_names = [n for n in master if n.endswith("balance_bias")]
    assert len(master) == n_params + cfg.num_layers
    assert len(bias_names) == cfg.num_layers
    for name in bias_names:
        assert master[name].dtype == torch.float32
        assert master[name].shape == (cfg.num_experts,)

    shapes = model.param_shapes()
    assert shapes["lm_head.weight"] == (cfg.vocab_size, cfg.hidden_size)
    assert shapes["embed.weight"] == (cfg.vocab_size, cfg.hidden_size)
    assert shapes["blocks.0.moe.routed_gate"] == (
        cfg.num_experts,
        cfg.intermediate_size,
        cfg.hidden_size,
    )
    assert shapes["blocks.0.moe.router.balance_bias"] == (cfg.num_experts,)


def test_is_expert_local_partition() -> None:
    cfg = tiny_cfg()
    model = _tiny_model()
    expert = [n for n, _ in model.iter_master_params() if model.is_expert_local(n)]
    assert len(expert) == 3 * cfg.num_layers  # routed gate/up/down per layer
    assert all(".routed_" in n for n in expert)
    replicated = [n for n, _ in model.iter_master_params() if not model.is_expert_local(n)]
    assert "lm_head.weight" in replicated
    assert "blocks.0.moe.shared_gate" in replicated
    assert "blocks.0.moe.router.proj.weight" in replicated
    assert "blocks.0.moe.router.balance_bias" in replicated


def test_dtype_layout() -> None:
    model = _tiny_model()
    assert model.embed.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.float32          # fp32 master
    assert model.blocks[0].attn.qkv.weight.dtype == torch.bfloat16
    assert model.blocks[0].moe.router.proj.weight.dtype == torch.float32
    assert model.blocks[0].moe.routed_gate.dtype == torch.bfloat16
    assert not any(isinstance(m, torch.nn.Dropout) for m in model.modules())


def test_mok_backend_asserts_token_count_rules() -> None:
    # construction never imports mok; the T = B*S check fires before the kernel path
    cfg = ModelConfig(
        num_layers=1,
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
    )
    model = MoKTransformer(cfg, backend="mok")
    with pytest.raises(ValueError, match="num_local_tokens"):
        model(torch.randint(0, cfg.vocab_size, (1, 128)))  # T=128 < 512
    assert "mok" not in sys.modules


def test_backend_validated_at_construction() -> None:
    with pytest.raises(ValueError, match="backend"):
        MoKTransformer(tiny_cfg(), backend="nope")


def test_no_mok_import_in_cpu_suite() -> None:
    assert "mok" not in sys.modules
