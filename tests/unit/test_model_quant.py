"""MXFP8 cache wire formats + bf16 passthrough — all without importing `mok`.

The quantization kernel itself is SM103-only (covered by the GPU suite); what
must hold on CPU is the consensus-relevant plumbing: the exact README recipe
tuple shapes (fwd pairs; bwd 4-tuples for gate/up, transposed-only pair for
down), tensor identity for buffer reuse, and the bf16 passthrough.
"""

from __future__ import annotations

import sys

import pytest
import torch

from mok_core.config import ModelConfig
from mok_core.model import MoKMoELayer, MXFP8WeightManager, QuantizedRoutedWeights


def _ids(pairs: object) -> object:
    """Map nested tensor tuples to id() so comparisons check identity, not values."""
    if isinstance(pairs, tuple):
        return tuple(_ids(p) for p in pairs)
    return id(pairs)


def _dummy_cache() -> tuple[QuantizedRoutedWeights, list[torch.Tensor]]:
    tensors = [torch.full((1,), float(i)) for i in range(12)]
    return QuantizedRoutedWeights(*tensors), tensors


def test_forward_args_are_fp8_sc_pairs_in_gate_up_down_order() -> None:
    cache, t = _dummy_cache()
    gate, up, down = cache.forward_args()
    assert _ids(gate) == _ids((t[0], t[1]))    # gate_fp8, gate_sc
    assert _ids(up) == _ids((t[4], t[5]))      # up_fp8, up_sc
    assert _ids(down) == _ids((t[8], t[9]))    # down_fp8, down_sc


def test_backward_args_use_transposed_only_down() -> None:
    # consensus wire format — mixture-of-kittens README lines 204-227: 4-tuples
    # for gate/up, (t_fp8, t_sc) ONLY for down. Change requires SPEC_VERSION bump.
    cache, t = _dummy_cache()
    gate, up, down = cache.backward_args()
    assert _ids(gate) == _ids((t[0], t[1], t[2], t[3]))
    assert _ids(up) == _ids((t[4], t[5], t[6], t[7]))
    assert _ids(down) == _ids((t[10], t[11]))  # down_t_fp8, down_t_sc — NOT the normal layout
    assert len(down) == 2


def test_tensors_returns_identical_objects_for_buffer_reuse() -> None:
    cache, t = _dummy_cache()
    for held, original in zip(cache.tensors(), t, strict=True):
        assert held is original  # copy_-into-cache depends on identity, not equality


def _bf16_cfg() -> ModelConfig:
    return ModelConfig(
        num_layers=1,
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
        routed_precision="bf16",
    )


def test_bf16_mode_is_a_passthrough_without_mok() -> None:
    layer = MoKMoELayer(_bf16_cfg(), layer_idx=0)
    manager = MXFP8WeightManager([layer])
    manager.requantize_all_()
    assert layer.quant_cache is None
    # bf16 mode hands the BF16 masters straight to the kernel
    fwd = layer._forward_weight_args()
    bwd = layer._backward_weight_args()
    assert _ids(fwd) == _ids((layer.routed_gate, layer.routed_up, layer.routed_down))
    assert _ids(bwd) == _ids((layer.routed_gate, layer.routed_up, layer.routed_down))
    assert "mok" not in sys.modules


def test_mxfp8_mode_requires_requantize_before_kernel_use() -> None:
    cfg = _bf16_cfg().model_copy(update={"routed_precision": "mxfp8"})
    assert cfg.routed_precision == "mxfp8"
    layer = MoKMoELayer(cfg, layer_idx=0)
    with pytest.raises(RuntimeError, match="requantize_"):
        layer._forward_weight_args()
    with pytest.raises(RuntimeError, match="requantize_"):
        layer._backward_weight_args()
