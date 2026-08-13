"""Reference-model helpers: forced ep_size=1, deterministic loss evaluation."""

from __future__ import annotations

import pytest
import torch

from mok_core.config import ModelConfig
from mok_core.model import build_reference_model, evaluate_sequences, reference_config


def tiny_cfg() -> ModelConfig:
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
        ep_size=4,
    )


def test_reference_config_forces_ep1() -> None:
    cfg = tiny_cfg()
    ref = reference_config(cfg)
    assert ref.ep_size == 1
    assert ref.num_local_experts == cfg.num_experts
    assert reference_config(ref) is ref  # idempotent
    # everything else untouched
    assert ref.model_dump(exclude={"ep_size"}) == cfg.model_dump(exclude={"ep_size"})


def test_build_reference_model_holds_all_experts() -> None:
    model = build_reference_model(tiny_cfg(), seed=9)
    assert model.backend == "reference"
    assert model.cfg.ep_size == 1
    assert model.blocks[0].moe.routed_gate.shape[0] == tiny_cfg().num_experts


def test_evaluate_sequences_token_weighted_mean_ce() -> None:
    torch.manual_seed(0)
    model = build_reference_model(tiny_cfg(), seed=9)
    batches = [
        torch.randint(0, 512, (1, 65)),
        torch.randint(0, 512, (2, 33)),
        torch.randint(0, 512, (17,)),  # 1-D accepted
    ]
    loss = evaluate_sequences(model, batches)
    assert isinstance(loss, float)
    assert 0.0 < loss < 20.0
    # deterministic: same model + same batches -> identical value
    assert evaluate_sequences(model, batches) == loss
    # no grad side effects
    assert all(p.grad is None for p in model.parameters())


def test_evaluate_sequences_rejects_empty() -> None:
    model = build_reference_model(tiny_cfg(), seed=9)
    with pytest.raises(ValueError, match="no tokens"):
        evaluate_sequences(model, [])
