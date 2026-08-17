"""CausalSelfAttention: GQA math vs a hand-rolled reference, RoPE cache, backend pinning."""

from __future__ import annotations

import math

import pytest
import torch

from mok_core.config import ModelConfig
from mok_core.model import (
    ATTENTION_BACKEND_ENV,
    CausalSelfAttention,
    resolve_attention_backend,
    rope_cos_sin,
    sdpa_backend,
)


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


def _manual_rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """Independent RoPE reimplementation: x [B, S, n, d], rotate-half convention."""
    _, seq, _, dim = x.shape
    half = dim // 2
    freqs = theta ** (-torch.arange(0, half, dtype=torch.float32) * 2.0 / dim)  # [d/2]
    angles = torch.arange(seq, dtype=torch.float32)[:, None] * freqs[None, :]   # [S, d/2]
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


def test_gqa_matches_manual_reference_math() -> None:
    torch.manual_seed(0)
    cfg = tiny_cfg()
    attn = CausalSelfAttention(cfg, dtype=torch.float32)
    batch, seq = 2, 16
    x = torch.randn(batch, seq, cfg.hidden_size)

    y = attn(x)
    assert y.shape == (batch, seq, cfg.hidden_size)

    # Manual: fused qkv -> split -> rope -> expand kv -> masked softmax -> o_proj
    nq, nkv, hd = cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    qkv = x @ attn.qkv.weight.T
    q, k, v = torch.split(qkv, [nq * hd, nkv * hd, nkv * hd], dim=-1)
    q = _manual_rope(q.view(batch, seq, nq, hd), cfg.rope_theta).transpose(1, 2)
    k = _manual_rope(k.view(batch, seq, nkv, hd), cfg.rope_theta).transpose(1, 2)
    v = v.view(batch, seq, nkv, hd).transpose(1, 2)
    k = k.repeat_interleave(nq // nkv, dim=1)
    v = v.repeat_interleave(nq // nkv, dim=1)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
    ref = torch.softmax(scores + mask, dim=-1) @ v
    ref = ref.transpose(1, 2).reshape(batch, seq, nq * hd) @ attn.o_proj.weight.T

    torch.testing.assert_close(y, ref, rtol=1e-4, atol=1e-5)


def test_rope_theta_override_changes_output_and_cache() -> None:
    torch.manual_seed(1)
    cfg = tiny_cfg()
    base = CausalSelfAttention(cfg, dtype=torch.float32)
    long_ctx = CausalSelfAttention(cfg, rope_theta=500_000.0, dtype=torch.float32)
    long_ctx.load_state_dict(base.state_dict())  # identical weights, different theta

    x = torch.randn(1, 8, cfg.hidden_size)
    assert not torch.allclose(base(x), long_ctx(x))

    cos_a, sin_a = rope_cos_sin(8, cfg.rope_theta, torch.device("cpu"), torch.float32, cfg.head_dim)
    cos_b, _ = rope_cos_sin(8, 500_000.0, torch.device("cpu"), torch.float32, cfg.head_dim)
    assert not torch.equal(cos_a, cos_b)
    # cache hit returns the same tensors
    cos_a2, sin_a2 = rope_cos_sin(8, cfg.rope_theta, torch.device("cpu"), torch.float32, cfg.head_dim)
    assert cos_a is cos_a2 and sin_a is sin_a2


def test_backend_resolver_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ATTENTION_BACKEND_ENV, raising=False)
    assert resolve_attention_backend() in ("cudnn_det", "flash_det")
    # CUDA-less hosts must default to flash_det (suite runs on both kinds).
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_attention_backend() == "flash_det"

    monkeypatch.setenv(ATTENTION_BACKEND_ENV, "cudnn_det")
    assert resolve_attention_backend() == "cudnn_det"
    monkeypatch.setenv(ATTENTION_BACKEND_ENV, "bogus")
    with pytest.raises(ValueError, match="MOK_ATTENTION_BACKEND"):
        resolve_attention_backend()


def test_sdpa_context_falls_back_to_math_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit CPU device always pins math, even on CUDA hosts.
    with sdpa_backend(device="cpu") as pinned:
        assert pinned == "math"
        q = torch.randn(1, 2, 4, 8)
        y = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    assert y.shape == (1, 2, 4, 8)
    # No device given: CUDA-less hosts fall back to math too.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with sdpa_backend() as pinned:
        assert pinned == "math"


def test_no_dropout_anywhere() -> None:
    attn = CausalSelfAttention(tiny_cfg())
    assert not any(isinstance(m, torch.nn.Dropout) for m in attn.modules())
