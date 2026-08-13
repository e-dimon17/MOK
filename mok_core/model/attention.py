"""Causal GQA self-attention with a pinned, deterministic SDPA backend.

Attention runs in plain PyTorch (MoK covers only the MoE layer). Determinism
contract: exactly one SDPA kernel is ever eligible — resolved once via
`resolve_attention_backend()` (env-overridable) and pinned with
`sdpa_backend()` around every forward/backward. On CPU the math backend is
the (only) deterministic fallback. No dropout anywhere in the model.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Literal

import torch
from torch import nn

from mok_core.config import ModelConfig

ATTENTION_BACKEND_ENV = "MOK_ATTENTION_BACKEND"
AttentionBackend = Literal["cudnn_det", "flash_det"]
_VALID_BACKENDS: tuple[str, ...] = ("cudnn_det", "flash_det")

# (seq_len, theta, device, dtype) -> (cos [S, head_dim], sin [S, head_dim])
_ROPE_CACHE: dict[tuple[int, float, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def resolve_attention_backend() -> AttentionBackend:
    """Pick the deterministic attention kernel: env override, else cuDNN when
    available, else flash. The choice is pinned in the run manifest — every
    node in a run must resolve identically."""
    override = os.environ.get(ATTENTION_BACKEND_ENV)
    if override is not None:
        if override not in _VALID_BACKENDS:
            raise ValueError(
                f"{ATTENTION_BACKEND_ENV}={override!r} invalid; expected one of {_VALID_BACKENDS}"
            )
        return override  # type: ignore[return-value]
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        return "cudnn_det"
    return "flash_det"


@contextlib.contextmanager
def sdpa_backend(backend: AttentionBackend | None = None) -> Iterator[str]:
    """Pin `torch.nn.attention.sdpa_kernel` to exactly one backend.

    On CPU-only hosts (tests, verify_bundle) SDPA falls back to the math
    backend — the pinned-kernel requirement only binds on CUDA. Yields the
    name of the backend actually pinned.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

    if not torch.cuda.is_available():
        with sdpa_kernel([SDPBackend.MATH]):
            yield "math"
        return
    resolved = backend if backend is not None else resolve_attention_backend()
    mapped = {
        "cudnn_det": SDPBackend.CUDNN_ATTENTION,
        "flash_det": SDPBackend.FLASH_ATTENTION,
    }[resolved]
    with sdpa_kernel([mapped]):
        yield resolved


def rope_cos_sin(
    seq_len: int,
    theta: float,
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lazily-built, cached RoPE cos/sin tables, keyed (seq_len, theta, device, dtype)."""
    key = (int(seq_len), float(theta), str(device), dtype)
    cached = _ROPE_CACHE.get(key)
    if cached is not None and cached[0].shape[-1] == head_dim:
        return cached
    inv_freq = 1.0 / (
        float(theta)
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                       # [S, head_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)                # [S, head_dim]
    cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
    _ROPE_CACHE[key] = (cos, sin)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x` [B, S, n_heads, head_dim] by cached cos/sin [S, head_dim]."""
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """GQA causal attention: fused bias-free QKV, RoPE, SDPA with is_causal=True.

    `rope_theta` overrides `cfg.rope_theta` (step E raises it to 5e5 for 16k
    context without touching the rest of the architecture).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        rope_theta: float | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_q_heads = cfg.num_q_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size
        self.rope_theta = float(rope_theta if rope_theta is not None else cfg.rope_theta)
        qkv_dim = (cfg.num_q_heads + 2 * cfg.num_kv_heads) * cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, qkv_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(cfg.num_q_heads * cfg.head_dim, cfg.hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        nq, nkv, hd = self.num_q_heads, self.num_kv_heads, self.head_dim
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [nq * hd, nkv * hd, nkv * hd], dim=-1)
        q = q.view(batch, seq, nq, hd)
        k = k.view(batch, seq, nkv, hd)
        v = v.view(batch, seq, nkv, hd)

        cos, sin = rope_cos_sin(seq, self.rope_theta, x.device, x.dtype, hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)  # [B, nq, S, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=(nq != nkv)
        )
        y = y.transpose(1, 2).reshape(batch, seq, nq * hd)
        return self.o_proj(y)

    def extra_repr(self) -> str:
        return (
            f"q_heads={self.num_q_heads}, kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, rope_theta={self.rope_theta}"
        )
