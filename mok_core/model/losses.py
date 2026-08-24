"""Loss head with a FIXED reduction order (replay-bitwise contract).

Reduction order (consensus-relevant — never reorder):
  1. ce       = fp32 mean cross-entropy over all tokens (logits cast fp32 first)
  2. aux      = mean over layers (in layer order) of the switch-style aux loss
  3. router_z = mean over layers (in layer order) of z_loss(router_logits)
  4. output_z = z_loss(lm_logits fp32)
  5. total    = ce + aux_loss_coef*aux + router_z_coef*router_z + output_z_coef*output_z
     (coefficients added in ModelConfig schema field order)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mok_core.config import ModelConfig

from .router import RouterOutput


@dataclass(frozen=True)
class LossOutput:
    """All fp32 scalars; `total` is the backward target, the rest are telemetry."""

    total: torch.Tensor
    ce: torch.Tensor
    aux: torch.Tensor
    router_z: torch.Tensor
    output_z: torch.Tensor


def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """mean(logsumexp(logits, -1)^2) in fp32 — pulls the partition function to 1."""
    return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """z-loss over the router's unbiased fp32 logits [T, E]."""
    return z_loss(router_logits)


def aux_load_loss(router_logits: torch.Tensor, load: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-style backup aux loss: E * sum_i f_i * P_i.

    Equals 1.0 at perfect balance. Reduction is MEAN over MoE layers (loss_head),
    matching DeepSeek-V3's complementary-loss magnitude class (coef 1e-4), NOT
    Switch's summed convention (coef 1e-2): this term is telemetry-grade pressure
    only — early-training balance protection comes from the schedule-capacity
    warmup margin (multiplier 1.05, see subnet/configs/base.yaml), never from this loss.
    """
    num_tokens, num_experts = router_logits.shape
    probs = torch.softmax(router_logits.float(), dim=-1)
    mean_prob = probs.mean(dim=0)                                   # [E]
    frac = load.float().to(mean_prob.device) / float(num_tokens * top_k)
    return num_experts * torch.sum(frac * mean_prob)


def loss_head(
    lm_logits: torch.Tensor,
    targets: torch.Tensor,
    router_stats: Sequence[RouterOutput],
    cfg: ModelConfig,
) -> LossOutput:
    """Combined training loss. `lm_logits` [B, S, V] (any float dtype, cast fp32
    once up front); `targets` int64 [B, S]; `router_stats` in layer order."""
    vocab = lm_logits.shape[-1]
    logits32 = lm_logits.float()
    ce = F.cross_entropy(logits32.reshape(-1, vocab), targets.reshape(-1))

    num_layers = len(router_stats)
    if num_layers == 0:
        raise ValueError("router_stats must contain at least one layer")
    aux_sum = router_stats[0].router_logits.new_zeros(())
    router_z_sum = router_stats[0].router_logits.new_zeros(())
    for stats in router_stats:  # fixed layer order
        aux_sum = aux_sum + aux_load_loss(stats.router_logits, stats.load, cfg.top_k)
        router_z_sum = router_z_sum + router_z_loss(stats.router_logits)
    aux = aux_sum / num_layers
    router_z = router_z_sum / num_layers

    output_z = z_loss(logits32)
    total = (
        ce
        + cfg.aux_loss_coef * aux
        + cfg.router_z_coef * router_z
        + cfg.output_z_coef * output_z
    )
    return LossOutput(total=total, ce=ce, aux=aux, router_z=router_z, output_z=output_z)
