"""Router: aux-free bias steers SELECTION only; gate weights stay unbiased; sign rule."""

from __future__ import annotations

import math

import torch

from mok_core.config import ModelConfig
from mok_core.model import Router


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


def _controlled_router() -> tuple[Router, torch.Tensor]:
    """Router whose logits for the probe token are exactly [3, 2, 1, 0, -1, ...]."""
    cfg = tiny_cfg()
    router = Router(cfg)
    with torch.no_grad():
        router.proj.weight.zero_()
        router.proj.weight[:, 0] = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    x = torch.zeros(1, cfg.hidden_size)
    x[0, 0] = 1.0
    return router, x


def test_unbiased_selection_and_gate_weights() -> None:
    router, x = _controlled_router()
    out = router(x)
    assert out.experts.tolist() == [[0, 1]]
    assert out.experts.dtype == torch.int64
    assert out.weights.dtype == torch.float32
    expected = torch.softmax(torch.tensor([[3.0, 2.0]]), dim=-1)
    torch.testing.assert_close(out.weights, expected)
    torch.testing.assert_close(out.weights.sum(-1), torch.ones(1))
    assert out.router_logits.dtype == torch.float32
    assert out.router_logits.shape == (1, 8)


def test_bias_flips_selection_but_not_gate_weights() -> None:
    router, x = _controlled_router()
    with torch.no_grad():
        router.balance_bias[2] = 10.0  # biased logits: [3, 2, 11, 0, ...] -> top2 {2, 0}
    out = router(x)
    assert out.experts.tolist() == [[2, 0]]
    # Gate weights come from the UNBIASED logits of the selected experts: [1, 3]
    expected = torch.softmax(torch.tensor([[1.0, 3.0]]), dim=-1)
    torch.testing.assert_close(out.weights, expected)
    w_kept = float(out.weights[0, 1].detach())
    assert math.isclose(w_kept, math.exp(3) / (math.exp(1) + math.exp(3)), rel_tol=1e-6)
    # The bias never leaks into the reported logits (z-loss / aux-loss input)
    torch.testing.assert_close(
        out.router_logits[0], torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0])
    )


def test_load_counts() -> None:
    router, x = _controlled_router()
    out = router(x.repeat(3, 1))  # 3 identical tokens -> experts {0,1} each
    assert out.load.dtype == torch.int64
    assert out.load.tolist() == [3, 3, 0, 0, 0, 0, 0, 0]
    assert int(out.load.sum()) == 3 * router.top_k


def test_update_balance_bias_sign_rule() -> None:
    router, _ = _controlled_router()
    load = torch.tensor([4, 0, 2, 2, 2, 2, 2, 2], dtype=torch.int64)  # mean = 2
    rate = 1e-3
    router.update_balance_bias_(load, rate)
    expected = torch.tensor([-rate, rate, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    torch.testing.assert_close(router.balance_bias, expected)
    # applied in place, cumulatively, deterministically
    router.update_balance_bias_(load, rate)
    torch.testing.assert_close(router.balance_bias, 2 * expected)


def test_balance_bias_is_persistent_nontrainable_fp32() -> None:
    router, _ = _controlled_router()
    assert router.balance_bias.dtype == torch.float32
    assert not router.balance_bias.requires_grad
    assert "balance_bias" in router.state_dict()  # persistent buffer -> state_root domain
    assert router.proj.weight.dtype == torch.float32


def test_gate_weights_backprop_into_projection() -> None:
    router, x = _controlled_router()
    out = router(x)
    out.weights.sum().backward()
    assert router.proj.weight.grad is not None
    assert torch.isfinite(router.proj.weight.grad).all()
