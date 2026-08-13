"""MXFP8 routed-weight management — the exact recipe from the MoK README.

After every optimizer step the BF16 master routed weights are re-quantized:
`mok.ops.mxfp8_quantize(w, True, True)` per routed matrix yields
(fp8, sc, t_fp8, t_sc). The forward consumes (fp8, sc) pairs for gate/up/down;
the backward consumes 4-tuples for gate/up and ONLY the transposed pair for
down (mixture-of-kittens README lines 153–226, tests/test_functional_e2e.py).
Shared-expert weights stay BF16 always.

Quantized copies are DERIVED data: excluded from the state_root domain by
construction (held as a plain attribute, never registered as buffers). Cached
tensors are reused across steps via `copy_` so their storage pointers stay
stable (CUDA-graph / compile friendly). `mok` is imported lazily — this module
loads fine on CPU-only hosts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover — avoid a runtime moe<->quant cycle
    from .moe import MoKMoELayer

_FwdPair = tuple[torch.Tensor, torch.Tensor]
_BwdQuad = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class QuantizedRoutedWeights:
    """One layer's MXFP8 copies, in `mxfp8_quantize` output order per matrix."""

    gate_fp8: torch.Tensor
    gate_sc: torch.Tensor
    gate_t_fp8: torch.Tensor
    gate_t_sc: torch.Tensor
    up_fp8: torch.Tensor
    up_sc: torch.Tensor
    up_t_fp8: torch.Tensor
    up_t_sc: torch.Tensor
    down_fp8: torch.Tensor
    down_sc: torch.Tensor
    down_t_fp8: torch.Tensor
    down_t_sc: torch.Tensor

    def tensors(self) -> tuple[torch.Tensor, ...]:
        # NOT dataclasses.astuple (it deep-copies tensors); identity matters for copy_.
        return tuple(getattr(self, f.name) for f in fields(self))

    def forward_args(self) -> tuple[_FwdPair, _FwdPair, _FwdPair]:
        """(fp8, sc) pairs for functional.forward: gate, up, down."""
        return (
            (self.gate_fp8, self.gate_sc),
            (self.up_fp8, self.up_sc),
            (self.down_fp8, self.down_sc),
        )

    def backward_args(self) -> tuple[_BwdQuad, _BwdQuad, _FwdPair]:
        """functional.backward wire format: 4-tuples for gate/up, transposed-only pair for down."""
        return (
            (self.gate_fp8, self.gate_sc, self.gate_t_fp8, self.gate_t_sc),
            (self.up_fp8, self.up_sc, self.up_t_fp8, self.up_t_sc),
            (self.down_t_fp8, self.down_t_sc),
        )


class MXFP8WeightManager:
    """Owns re-quantization of every MoE layer's routed weights.

    Call `requantize_all_()` once after model init / checkpoint load and after
    every optimizer step (and after every outer step). In bf16 mode
    (`cfg.routed_precision == "bf16"`) this is a passthrough — the kernels run
    directly on the BF16 masters.
    """

    def __init__(self, layers: Sequence[MoKMoELayer]) -> None:
        self._layers = list(layers)

    @property
    def layers(self) -> list[MoKMoELayer]:
        return list(self._layers)

    def requantize_all_(self) -> None:
        for layer in self._layers:
            self.requantize_(layer)

    def requantize_(self, layer: MoKMoELayer) -> None:
        """Refresh (or first-build) one layer's MXFP8 cache from its BF16 masters."""
        if layer.cfg.routed_precision == "bf16":
            return  # passthrough: kernels consume the BF16 masters directly
        from mok import ops  # noqa: PLC0415 — SM103-only wheel, lazy by design

        fresh: list[torch.Tensor] = []
        for weight in (layer.routed_gate, layer.routed_up, layer.routed_down):
            fp8, sc, t_fp8, t_sc = ops.mxfp8_quantize(weight.detach(), True, True)
            fresh.extend((fp8, sc, t_fp8, t_sc))

        cache = layer.quant_cache
        if cache is None:
            layer.quant_cache = QuantizedRoutedWeights(*fresh)
            return
        for dst, src in zip(cache.tensors(), fresh, strict=True):
            dst.copy_(src)  # reuse buffers: stable storage across steps
