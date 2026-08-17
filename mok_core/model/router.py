"""FP32 router with DeepSeek-V3-style aux-free load balancing.

The router projection stays fp32 end-to-end (playbook: "fp32 router") — it is
the smallest, most divergence-prone tensor in the model. Balancing uses a
persistent non-trainable `balance_bias` buffer added to the logits FOR TOP-K
SELECTION ONLY; the gate weights are a softmax over the SELECTED experts'
UNBIASED logits, so the bias steers dispatch without perturbing the mixture.
`balance_bias` is part of the state_root domain (it evolves deterministically
via the sign rule and must replay bitwise).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mok_core.config import ModelConfig


@dataclass(frozen=True)
class RouterOutput:
    """Per-microbatch routing decision.

    weights:       fp32 [T, topk] — renormalized softmax over the selected
                   experts' unbiased logits (differentiable).
    experts:       int64 [T, topk] — selected expert ids (bias-influenced).
    router_logits: fp32 [T, E] — unbiased logits (z-loss / aux-loss input).
    load:          int64 [E] — tokens dispatched per expert this microbatch.
    """

    weights: torch.Tensor
    experts: torch.Tensor
    router_logits: torch.Tensor
    load: torch.Tensor


class Router(nn.Module):
    """hidden -> num_experts fp32 projection + aux-free biased top-k selection."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.num_experts = cfg.num_experts
        self.top_k = cfg.top_k
        self.proj = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False, dtype=torch.float32)
        self.register_buffer(
            "balance_bias", torch.zeros(cfg.num_experts, dtype=torch.float32), persistent=True
        )

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """x: [T, H] any float dtype -> RouterOutput (all routing math in fp32)."""
        if x.dim() != 2:
            raise ValueError(f"router input must be [T, H], got {tuple(x.shape)}")
        # bf16 autocast would silently demote the projection matmul, breaking
        # the fp32-router guarantee — routing math runs outside any autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.proj(x.float())                            # fp32 [T, E]
            biased = logits + self.balance_bias                      # selection only
            _, experts = torch.topk(biased, self.top_k, dim=-1)      # int64 [T, topk]
            selected = logits.gather(1, experts)                     # UNBIASED logits
            weights = torch.softmax(selected, dim=-1)                # fp32 [T, topk]
            # Not torch.bincount: its CUDA kernel is flagged nondeterministic and
            # raises under use_deterministic_algorithms(True). Equality-matrix sum
            # is deterministic on every device.
            expert_ids = torch.arange(self.num_experts, device=experts.device)
            load = (experts.reshape(-1, 1) == expert_ids).sum(dim=0)
        return RouterOutput(weights=weights, experts=experts, router_logits=logits, load=load)

    @torch.no_grad()
    def update_balance_bias_(self, load: torch.Tensor, rate: float) -> None:
        """Aux-free bias nudge: b -= rate * sign(load - mean(load)), in place.

        Deterministic (sign of exact integer counts vs their fp32 mean); called
        after each optimizer step from the same `load` on every replica.
        """
        load_f = load.to(device=self.balance_bias.device, dtype=torch.float32)
        self.balance_bias.sub_(rate * torch.sign(load_f - load_f.mean()))
