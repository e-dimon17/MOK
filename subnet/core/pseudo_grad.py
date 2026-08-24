"""Pseudo-gradient extraction: CPU snapshot of θ_start, restore + delta.

The DiLoCo outer loop needs Δ = θ_start − θ_end at the end of every window,
AND the parameters put back to θ_start so the deterministic outer step
(subnet/core/outer_opt.py) is the only thing that moves the master weights.

Deliberately non-DTensor: our EP sharding is expressed by
plain per-rank parameter NAMES (`.routed_` marker), never by DTensor, so the
mesh/placement bookkeeping and the grad-storage staging trick drop out. What
remains: reusable pinned CPU buffers with async D2H copies fenced before use,
and the exact `Δ = saved − current; p ← saved` restore algebra (computed here
into fresh fp32 CPU tensors instead of `p.grad`, which the inner loop owns).

Snapshots keep the MASTER dtype (bf16/fp32) so `restore_` is a bitwise
round-trip; deltas are always fp32 CPU (the compression pipeline's input).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

__all__ = ["CpuSnapshot", "restore_and_extract_delta"]


def _named_items(
    named_params: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    items = list(named_params.items()) if isinstance(named_params, Mapping) else list(named_params)
    names = [name for name, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("duplicate names in named_params")
    return items


@dataclass(frozen=True, eq=False)
class CpuSnapshot:
    """θ_start frozen on CPU: name -> CPU clone in the master dtype."""

    tensors: dict[str, torch.Tensor]
    pinned: bool

    @classmethod
    def take(
        cls,
        named_params: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        *,
        pin: bool | None = None,
    ) -> CpuSnapshot:
        """Snapshot every named tensor into fresh CPU buffers.

        `pin=None` (default) resolves to True only when snapshotting CUDA
        tensors: pinned buffers + async D2H copies (fenced with one synchronize
        before returning). CPU-resident sources take plain pageable copies even
        on CUDA hosts — allocating pinned memory would spin up a CUDA context,
        which pure-CPU callers (and enforce_determinism after them) must not do.
        """
        items = [(name, param) for name, param in _named_items(named_params)]
        if pin is None:
            use_pin = torch.cuda.is_available() and any(p.is_cuda for _, p in items)
        else:
            use_pin = bool(pin)
        out: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in items:
                src = param.detach()
                buf = torch.empty_like(src, device="cpu", pin_memory=use_pin)
                buf.copy_(src, non_blocking=use_pin)
                out[name] = buf
        if use_pin and torch.cuda.is_available():
            torch.cuda.synchronize()  # fence: all async D2H copies landed
        return cls(tensors=out, pinned=use_pin)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.tensors))

    def __len__(self) -> int:
        return len(self.tensors)

    def __contains__(self, name: str) -> bool:
        return name in self.tensors


@torch.no_grad()
def restore_and_extract_delta(
    named_params: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    snapshot: CpuSnapshot,
) -> dict[str, torch.Tensor]:
    """Δ = θ_start − θ_end as fp32 CPU tensors, AND θ ← θ_start in place.

    `named_params` must cover exactly the snapshot's names (shape and dtype
    checked); after the call every param is a bitwise copy of its snapshot
    value (the round-trip is exact because snapshots keep the master dtype).
    Returned deltas are contiguous fp32 CPU, keyed by name.
    """
    items = _named_items(named_params)
    if {name for name, _ in items} != set(snapshot.tensors):
        missing = sorted(set(snapshot.tensors) - {n for n, _ in items})
        extra = sorted({n for n, _ in items} - set(snapshot.tensors))
        raise ValueError(f"snapshot/param name mismatch: missing={missing} extra={extra}")

    deltas: dict[str, torch.Tensor] = {}
    for name, param in items:
        saved = snapshot.tensors[name]
        current = param.detach()
        if saved.shape != current.shape or saved.dtype != current.dtype:
            raise ValueError(
                f"{name}: snapshot {tuple(saved.shape)}/{saved.dtype} vs "
                f"param {tuple(current.shape)}/{current.dtype}"
            )
        current_cpu = current.cpu()  # no-op view on CPU hosts, D2H copy on GPU
        # Materialized BEFORE the restore below mutates the underlying storage.
        deltas[name] = (saved.to(torch.float32) - current_cpu.to(torch.float32)).contiguous()
        param.copy_(saved)  # θ ← θ_start, bitwise
    return deltas
