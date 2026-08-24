"""MoK's own sanity on this node: workspace creation + MXFP8 quantize math.

The MXFP8 reference below is the 32-element-block quantize math from
mixture-of-kittens `tests/utils.py::run_mxfp8_quantize_normal_reference`
(Apache-2.0, github.com/cursor/mixture-of-kittens), inlined so this suite is
self-contained on nodes that install only the `mok` wheel. Upstream pins the
kernel EXACTLY equal to this reference (their EXACT_TOLERANCE = (0.0, 0.0));
we assert `torch.equal` accordingly.
"""

from __future__ import annotations

import pytest
import torch

from mok_core.config import EP_SIZES

pytestmark = pytest.mark.usefixtures("mok_available")


# --------------------------------------------------------------------------- #
# Inline 32-block MXFP8 quantize reference (self-contained)
# --------------------------------------------------------------------------- #


def mxfp8_quantize_reference(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(fp8, swizzled-scale-bytes) for a [rows, cols] or [E, rows, cols] bf16 tensor.

    Per 32-element block along the last dim: amax -> decode scale amax/448 ->
    exponent = clamp(ceil(log2(scale)), -127) -> fp8 = x / 2^exp (e4m3), scale
    byte = exp + 127, swizzled into the kernel's [128, columns//128, 4] layout.
    """
    is_2d = x_bf16.ndim == 2
    x3 = x_bf16.unsqueeze(0) if is_2d else x_bf16
    num_experts, rows, columns = x3.shape
    if columns % 32 != 0 or (num_experts * rows) % 128 != 0 or columns % 128 != 0:
        raise ValueError("reference layout needs cols % 128 == 0 and total rows % 128 == 0")

    x_float = x3.float().reshape(num_experts * rows, columns)
    dest_max = torch.tensor(448.0, dtype=torch.float32, device=x_bf16.device)
    min_exp = torch.tensor(-127.0, dtype=torch.float32, device=x_bf16.device)
    fp8e8m0_bias = torch.tensor(127.0, dtype=torch.float32, device=x_bf16.device)

    block_amax = x_float.abs().view(num_experts * rows, columns // 32, 32).amax(dim=-1)
    decode_scale = block_amax / dest_max
    scale_exponent = torch.clamp(torch.ceil(torch.log2(decode_scale)), min=min_exp)
    x_fp8 = (x_float / (2 ** scale_exponent.repeat_interleave(32, dim=-1))).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(num_experts, rows, columns)

    scale_bytes = (scale_exponent + fp8e8m0_bias).to(torch.uint8)
    scales = scale_bytes.reshape(num_experts * rows // 128, 128, columns // 128, 4).transpose(1, 2)
    scales = scales.reshape(num_experts * rows // 128, columns // 128, 4, 32, 4).transpose(-2, -3)
    scales = scales.reshape(num_experts * rows // 128, columns // 128, 32, 16).contiguous()

    return (x_fp8.squeeze(0) if is_2d else x_fp8), scales


def _dequantize_reference(x_bf16: torch.Tensor) -> torch.Tensor:
    """Round-trip decode using the same block exponents (for error bounding)."""
    x3 = x_bf16.unsqueeze(0) if x_bf16.ndim == 2 else x_bf16
    num_experts, rows, columns = x3.shape
    x_float = x3.float().reshape(num_experts * rows, columns)
    block_amax = x_float.abs().view(num_experts * rows, columns // 32, 32).amax(dim=-1)
    exponent = torch.clamp(torch.ceil(torch.log2(block_amax / 448.0)), min=-127.0)
    scale = (2 ** exponent).repeat_interleave(32, dim=-1)
    decoded = (x_float / scale).to(torch.float8_e4m3fn).float() * scale
    return decoded.reshape(x3.shape).squeeze(0) if x_bf16.ndim == 2 else decoded.reshape(x3.shape)


# --------------------------------------------------------------------------- #
# Workspace creation
# --------------------------------------------------------------------------- #


def test_workspace_creation_and_cache_for_toy_shapes(dist_ctx, mok_available, toy_cfg) -> None:
    """get_workspace succeeds for the toy4L shapes and caches by shape key."""
    import torch.distributed as dist

    functional = mok_available.functional
    if dist_ctx.world_size not in EP_SIZES:
        pytest.skip(f"MoK EP size must be one of {EP_SIZES}; world_size={dist_ctx.world_size}")

    mok_cfg = toy_cfg.mok.to_mok()
    kwargs = {
        "device": dist_ctx.device,
        "num_local_tokens": toy_cfg.window.tokens_per_rank_microbatch,
        "hidden_size": toy_cfg.model.hidden_size,
        "topk": toy_cfg.model.top_k,
    }
    workspace = functional.get_workspace(mok_cfg, dist.group.WORLD, **kwargs)
    assert workspace.num_local_tokens == toy_cfg.window.tokens_per_rank_microbatch
    assert workspace.topk == toy_cfg.model.top_k
    # Cached: the second request for the same key is the same object.
    again = functional.get_workspace(mok_cfg, dist.group.WORLD, **kwargs)
    assert again is workspace
    # A different shape key creates (and caches) a distinct workspace — the
    # per-shape caching contract the 16k context restart relies on.
    other = functional.get_workspace(
        mok_cfg,
        dist.group.WORLD,
        device=dist_ctx.device,
        num_local_tokens=toy_cfg.window.tokens_per_rank_microbatch // 2,
        hidden_size=toy_cfg.model.hidden_size,
        topk=toy_cfg.model.top_k,
    )
    assert other is not workspace
    dist_ctx.barrier()


def test_workspace_rejects_misaligned_tokens(dist_ctx, mok_available, toy_cfg) -> None:
    import torch.distributed as dist

    functional = mok_available.functional
    if dist_ctx.world_size not in EP_SIZES:
        pytest.skip(f"MoK EP size must be one of {EP_SIZES}; world_size={dist_ctx.world_size}")
    with pytest.raises(ValueError):
        functional.get_workspace(
            toy_cfg.mok.to_mok(),
            dist.group.WORLD,
            device=dist_ctx.device,
            num_local_tokens=8192 + 128,  # violates % 256
            hidden_size=toy_cfg.model.hidden_size,
            topk=toy_cfg.model.top_k,
        )
    dist_ctx.barrier()


# --------------------------------------------------------------------------- #
# mxfp8_quantize vs the inlined reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("2d_activations", (512, 1024)),
        ("3d_routed_experts", (2, 256, 1024)),  # toy4L [E_local, I, H]
    ],
)
def test_mxfp8_quantize_matches_reference(dist_ctx, mok_available, name: str, shape: tuple[int, ...]) -> None:
    ops = mok_available.ops
    generator = torch.Generator(device=dist_ctx.device).manual_seed(1234 + dist_ctx.rank)
    x = torch.randn(shape, generator=generator, device=dist_ctx.device, dtype=torch.bfloat16)

    fp8, sc, t_fp8, t_sc = ops.mxfp8_quantize(x, True, True)
    ref_fp8, ref_sc = mxfp8_quantize_reference(x)
    assert torch.equal(fp8.view(torch.uint8), ref_fp8.view(torch.uint8)), f"{name}: fp8 payload differs"
    assert torch.equal(sc, ref_sc), f"{name}: scale bytes differ"

    x_t = x.transpose(-2, -1).contiguous()
    ref_t_fp8, ref_t_sc = mxfp8_quantize_reference(x_t)
    assert torch.equal(t_fp8.view(torch.uint8), ref_t_fp8.view(torch.uint8)), f"{name}: transposed fp8"
    assert torch.equal(t_sc, ref_t_sc), f"{name}: transposed scale bytes"
    dist_ctx.barrier()


def test_mxfp8_round_trip_error_bounded(dist_ctx, mok_available) -> None:
    """Decode(quantize(w)) stays within MXFP8's per-block relative error budget
    (e4m3: 3 mantissa bits -> worst-case block-relative error ~2^-3 of amax)."""
    generator = torch.Generator(device=dist_ctx.device).manual_seed(7)
    w = torch.randn((2, 256, 1024), generator=generator, device=dist_ctx.device, dtype=torch.bfloat16)
    w = (w * 0.02).to(torch.bfloat16)  # trained-weight scale
    decoded = _dequantize_reference(w)
    err = (decoded - w.float()).abs()
    block_amax = (
        w.float().abs().reshape(2 * 256, 1024 // 32, 32).amax(dim=-1).repeat_interleave(32, -1)
    ).reshape(w.shape)
    assert torch.all(err <= block_amax * (2.0**-3) + 1e-12)
    rel_frob = err.norm() / w.float().norm()
    assert float(rel_frob) < 0.05
    dist_ctx.barrier()


def test_mxfp8_quantize_is_deterministic(dist_ctx, mok_available) -> None:
    """Same input twice -> byte-identical outputs (requant after every step
    depends on this)."""
    ops = mok_available.ops
    generator = torch.Generator(device=dist_ctx.device).manual_seed(99)
    w = torch.randn((2, 256, 1024), generator=generator, device=dist_ctx.device, dtype=torch.bfloat16)
    first = ops.mxfp8_quantize(w, True, True)
    second = ops.mxfp8_quantize(w.clone(), True, True)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a.view(torch.uint8) if a.dtype == torch.float8_e4m3fn else a,
                           b.view(torch.uint8) if b.dtype == torch.float8_e4m3fn else b)
    dist_ctx.barrier()
