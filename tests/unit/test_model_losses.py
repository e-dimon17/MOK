"""Loss formulas vs hand-computed values; fixed reduction order of loss_head."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mok_core.config import ModelConfig
from mok_core.model import RouterOutput, aux_load_loss, loss_head, router_z_loss, z_loss


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


def test_z_loss_hand_computed() -> None:
    # single row [0, 0]: logsumexp = ln 2 -> z = (ln 2)^2
    logits = torch.zeros(1, 2)
    assert math.isclose(float(z_loss(logits)), math.log(2.0) ** 2, rel_tol=1e-6)
    # two rows: mean of squared logsumexps
    logits = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    expected = (math.log(2.0) ** 2 + (1.0 + math.log(2.0)) ** 2) / 2.0
    assert math.isclose(float(z_loss(logits)), expected, rel_tol=1e-6)


def test_router_z_loss_is_z_loss_on_router_logits() -> None:
    logits = torch.randn(4, 8)
    assert float(router_z_loss(logits)) == float(z_loss(logits))


def test_aux_load_loss_hand_computed() -> None:
    # T=1, E=2, topk=1: probs = [0.75, 0.25], load = [1, 0]
    # f = [1, 0]; aux = E * (f . P) = 2 * 0.75 = 1.5
    logits = torch.tensor([[math.log(3.0), 0.0]])
    load = torch.tensor([1, 0], dtype=torch.int64)
    assert math.isclose(float(aux_load_loss(logits, load, top_k=1)), 1.5, rel_tol=1e-6)
    # perfectly balanced uniform routing -> exactly 1.0
    logits = torch.zeros(4, 2)
    load = torch.tensor([4, 4], dtype=torch.int64)
    assert math.isclose(float(aux_load_loss(logits, load, top_k=2)), 1.0, rel_tol=1e-6)


def test_loss_head_fixed_reduction_order() -> None:
    torch.manual_seed(0)
    cfg = tiny_cfg()
    batch, seq, vocab = 2, 4, cfg.vocab_size
    lm_logits = torch.randn(batch, seq, vocab, dtype=torch.bfloat16)
    targets = torch.randint(0, vocab, (batch, seq))

    stats = []
    for layer_seed in (1, 2):  # two layers, fixed order
        gen = torch.Generator().manual_seed(layer_seed)
        router_logits = torch.randn(batch * seq, cfg.num_experts, generator=gen)
        _, experts = torch.topk(router_logits, cfg.top_k, dim=-1)
        load = (experts.reshape(-1, 1) == torch.arange(cfg.num_experts)).sum(0)
        weights = torch.softmax(router_logits.gather(1, experts), dim=-1)
        stats.append(
            RouterOutput(weights=weights, experts=experts, router_logits=router_logits, load=load)
        )

    out = loss_head(lm_logits, targets, stats, cfg)

    # ce first: fp32 mean over all tokens
    ce = F.cross_entropy(lm_logits.float().reshape(-1, vocab), targets.reshape(-1))
    torch.testing.assert_close(out.ce, ce)
    # per-layer terms averaged in layer order
    aux = (
        aux_load_loss(stats[0].router_logits, stats[0].load, cfg.top_k)
        + aux_load_loss(stats[1].router_logits, stats[1].load, cfg.top_k)
    ) / 2
    router_z = (z_loss(stats[0].router_logits) + z_loss(stats[1].router_logits)) / 2
    output_z = z_loss(lm_logits.float())
    torch.testing.assert_close(out.aux, aux)
    torch.testing.assert_close(out.router_z, router_z)
    torch.testing.assert_close(out.output_z, output_z)
    # total: coefficients applied in schema order (aux, router_z, output_z)
    total = ce + cfg.aux_loss_coef * aux + cfg.router_z_coef * router_z + cfg.output_z_coef * output_z
    torch.testing.assert_close(out.total, total)
    assert out.total.dtype == torch.float32
    assert torch.isfinite(out.total)
