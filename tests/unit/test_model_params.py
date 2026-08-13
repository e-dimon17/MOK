"""Full-scale parameter accounting on the meta device (zero memory).

Exact decomposition with the default ModelConfig (L=32, first D=3 layers dense
SwiGLU at Id=9216, H=4096, V=65536, E=128, top-8, I=1024, GQA 32q/8kv/128):
  routed experts  E*3*I*H*(L-D)          = 46,707,769,344
  attention       ((nq+2nkv)*hd*H + nq*hd*H)*L =  1,342,177,280
  emb + LM head   2*V*H                  =    536,870,912
  shared experts  3*I*H*(L-D)            =    364,904,448
  dense FFNs      3*Id*H*D               =    339,738,624
  routers         (E*H + E)*(L-D)        =     15,208,064   (incl. balance biases)
  norms           (2L+1)*H               =        266,240
  TOTAL                                  = 49,306,934,912  (~49.3B)
  ACTIVE per token (top-8 of 128)        =  5,518,401,152  (~5.5B)

The dense width Id=9216 = 9*I matches the activated MoE width (shared + top-8
routed), so the active count is unchanged from the all-MoE layout (~5.5B).
"""

from __future__ import annotations

import torch

from mok_core.config import ModelConfig
from mok_core.model import MoKTransformer, reference_config


def _expected_total(cfg: ModelConfig, num_experts_held: int) -> int:
    attn = ((cfg.num_q_heads + 2 * cfg.num_kv_heads) * cfg.head_dim * cfg.hidden_size
            + cfg.num_q_heads * cfg.head_dim * cfg.hidden_size)
    shared = 3 * cfg.intermediate_size * cfg.hidden_size
    routed = num_experts_held * 3 * cfg.intermediate_size * cfg.hidden_size
    router = cfg.num_experts * cfg.hidden_size + cfg.num_experts  # proj + balance_bias
    dense_ffn = 3 * cfg.dense_intermediate_size * cfg.hidden_size
    moe_layers = cfg.num_layers - cfg.num_dense_layers
    per_layer_common = attn + 2 * cfg.hidden_size  # attn + the two block norms
    return (
        cfg.num_layers * per_layer_common
        + moe_layers * (shared + routed + router)
        + cfg.num_dense_layers * dense_ffn
        + 2 * cfg.vocab_size * cfg.hidden_size  # untied embedding + LM head
        + cfg.hidden_size                       # final norm
    )


def test_default_config_totals_49b_and_5p5b_active() -> None:
    cfg = reference_config(ModelConfig())  # ep_size 1 -> the model holds ALL experts
    with torch.device("meta"):
        model = MoKTransformer(cfg, backend="reference")
    total = sum(t.numel() for _, t in model.iter_master_params())

    expected = _expected_total(cfg, num_experts_held=cfg.num_experts)
    assert total == expected == 49_306_934_912  # exact closed form

    # active-per-token: swap E for top_k in the routed term
    active = _expected_total(cfg, num_experts_held=cfg.top_k)
    assert active == 5_518_401_152
    assert abs(active - 5.5e9) / 5.5e9 < 0.01


def test_ep8_rank_holds_local_expert_shard() -> None:
    cfg = ModelConfig()  # default ep_size=8 -> 16 experts per rank
    with torch.device("meta"):
        model = MoKTransformer(cfg, backend="mok")
    total = sum(t.numel() for _, t in model.iter_master_params())
    assert total == _expected_total(cfg, num_experts_held=cfg.num_local_experts)
    # expert shard is exactly 1/ep_size of the routed weights (MoE layers only)
    expert = sum(
        t.numel() for n, t in model.iter_master_params() if model.is_expert_local(n)
    )
    full_routed = (
        (cfg.num_layers - cfg.num_dense_layers)
        * cfg.num_experts * 3 * cfg.intermediate_size * cfg.hidden_size
    )
    assert expert * cfg.ep_size == full_routed
