"""Reference-backend helpers for validators, auditors and parity tests.

A reference model is the pure-PyTorch replica of the run: ep_size forced to 1
(via `model_copy`, deliberately bypassing the EP-size validator) so a single
process holds ALL experts. Scoring validators evaluate miner deltas with it;
the GPU parity gate compares it against the mok backend.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F

from mok_core.config import ModelConfig

from .init import init_model
from .transformer import MoKTransformer


def reference_config(cfg: ModelConfig) -> ModelConfig:
    """cfg with ep_size forced to 1 (all experts local). `model_copy` skips
    re-validation by design — 1 is not a legal kernel EP size, only a
    reference-backend one."""
    if cfg.ep_size == 1:
        return cfg
    return cfg.model_copy(update={"ep_size": 1})


def build_reference_model(
    cfg: ModelConfig, seed: int, device: str | torch.device = "cpu"
) -> MoKTransformer:
    """Deterministically initialized full-expert reference model."""
    return init_model(reference_config(cfg), seed, device=device, backend="reference")


def evaluate_sequences(
    model: MoKTransformer,
    token_batches: Iterable[torch.Tensor],
    device: str | torch.device = "cpu",
) -> float:
    """Token-weighted mean next-token CE (nats) over `token_batches`, no grad.

    Each batch is int64 [B, S+1] (or [S+1]); inputs are batch[:, :-1] and
    targets batch[:, 1:]. This is the loss-delta primitive behind validator
    eval bins.
    """
    dev = torch.device(device)
    total_nats = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in token_batches:
            tokens = batch.to(dev)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            output = model(inputs)
            vocab = output.logits.shape[-1]
            ce = F.cross_entropy(
                output.logits.float().reshape(-1, vocab), targets.reshape(-1), reduction="sum"
            )
            total_nats += float(ce)
            total_tokens += targets.numel()
    if total_tokens == 0:
        raise ValueError("evaluate_sequences received no tokens")
    return total_nats / total_tokens
