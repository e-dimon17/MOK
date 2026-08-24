"""Deterministic replicated outer step — the bitwise-lockstep linchpin.

Every node applies the outer optimizer to the SAME certified peer set in the
SAME order and must land on byte-identical parameters (protocol decision #3).
Hardened for determinism:

  - sort-based segment mean on CPU replaces `scatter_reduce_("mean")` — the
    CPU bincount kernel is a serial in-order loop, so there are no atomics and
    no reduction-order races anywhere in the merge;
  - all merge/momentum/update math runs in fp32 (accumulation in fp64) on CPU
    master copies, then casts back to the parameter dtype;
  - peers are consumed strictly in certificate-UID order and parameters in
    sorted-name order — both fixed by the window certificate, never by dict
    iteration or network arrival order.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from mok_core.config import OuterOptConfig

__all__ = [
    "OuterReport",
    "ReplicatedOuterStep",
    "deterministic_segment_mean",
    "median_norm_clip_factors",
]


def deterministic_segment_mean(
    indices: list[torch.Tensor],
    values: list[torch.Tensor],
    numel: int,
) -> torch.Tensor:
    """Mean of sparse contributions per index, bitwise repeatable.

    `indices[k]` / `values[k]` is peer k's flat (int64 index, value) pairs into a
    parameter of `numel` elements; peers appear in certificate-UID order. The
    result is a dense fp32 CPU tensor: mean over all contributions at each index,
    zero where nobody contributed. Duplicate indices (across or within peers)
    each count once in the mean.

    Determinism: concatenate in list order -> stable argsort by index -> segment
    sums via CPU bincount (serial in-order kernel, fp64 accumulation) -> divide
    by counts -> cast fp32. No scatter/atomic ops; identical inputs give
    identical bytes on every node.
    """
    if numel <= 0:
        raise ValueError(f"numel must be positive, got {numel}")
    if len(indices) != len(values):
        raise ValueError(f"{len(indices)} index tensors vs {len(values)} value tensors")
    if not indices:
        return torch.zeros(numel, dtype=torch.float32)

    idx_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    for k, (i, v) in enumerate(zip(indices, values, strict=True)):
        i = i.detach().reshape(-1).cpu().to(torch.int64)
        v = v.detach().reshape(-1).cpu().to(torch.float64)
        if i.numel() != v.numel():
            raise ValueError(f"peer {k}: {i.numel()} indices vs {v.numel()} values")
        idx_parts.append(i)
        val_parts.append(v)

    idx = torch.cat(idx_parts)
    val = torch.cat(val_parts)
    if idx.numel() == 0:
        return torch.zeros(numel, dtype=torch.float32)
    if int(idx.min()) < 0 or int(idx.max()) >= numel:
        raise ValueError(f"index out of range for numel={numel}")

    order = torch.argsort(idx, stable=True)
    idx = idx[order]
    val = val[order]
    sums = torch.bincount(idx, weights=val, minlength=numel)  # fp64, serial CPU kernel
    counts = torch.bincount(idx, minlength=numel)
    return (sums / counts.clamp(min=1)).to(torch.float32)


def median_norm_clip_factors(norms: torch.Tensor) -> torch.Tensor:
    """Per-peer clip factors: factor_i = min(1, median(norms) / norm_i).

    Caps outlier peers at the cohort's median gradient norm (the standard defense
    against norm-inflation). Zero/non-positive norms get factor 1.0 — their
    contribution is all zeros anyway, and this avoids 0/0. For an even peer
    count torch.median returns the lower middle value (deterministic). Result
    is fp32 CPU, same length as `norms`.
    """
    n = norms.detach().reshape(-1).cpu().to(torch.float32)
    if n.numel() == 0:
        return n
    med = n.median()
    ones = torch.ones_like(n)
    safe = torch.where(n > 0, n, ones)
    return torch.where(n > 0, (med / safe).clamp(max=1.0), ones)


@dataclass(frozen=True)
class OuterReport:
    """Telemetry fingerprint of one outer step (outside the deterministic path)."""

    global_grad_l2: float
    per_param_l2: dict[str, float] = field(default_factory=dict)
    applied_peers: int = 0


class ReplicatedOuterStep:
    """Outer optimizer replicated on every node; must produce identical bytes everywhere.

    Holds one fp32 CPU momentum buffer per parameter (checkpointed via
    state_dict). `apply` merges the certified peers' sparse pseudo-gradients
    with median-norm clipping and a deterministic segment mean, then takes a
    Nesterov (m = mu*m + g; d = g + mu*m) or plain-SGD (d = g) step in fp32
    before casting back to each parameter's dtype. Fleet calibration pins
    the final `kind`.
    """

    def __init__(self, cfg: OuterOptConfig, param_shapes: dict[str, torch.Size]) -> None:
        self.cfg = cfg
        self._shapes: dict[str, torch.Size] = {
            name: torch.Size(param_shapes[name]) for name in sorted(param_shapes)
        }
        self._momentum: dict[str, torch.Tensor] = {
            name: torch.zeros(shape, dtype=torch.float32) for name, shape in self._shapes.items()
        }

    # ------------------------------------------------------------------ #

    def _clip_factors(self, name: str, n_peers: int, peer_norms: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.cfg.clip == "none":
            return torch.ones(n_peers, dtype=torch.float32)
        norms = peer_norms.get(name)
        if norms is None:
            raise ValueError(f"median_norm clip enabled but no peer norms for param {name!r}")
        factors = median_norm_clip_factors(norms)
        if factors.numel() != n_peers:
            raise ValueError(f"param {name!r}: {factors.numel()} norms for {n_peers} peers")
        return factors

    @torch.no_grad()
    def apply(
        self,
        named_params: dict[str, torch.Tensor],
        peer_sparse: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
        dense_contribs: dict[str, list[torch.Tensor]],
        peer_norms: dict[str, torch.Tensor],
    ) -> OuterReport:
        """One outer step, in place on `named_params`.

        peer_sparse: per param, per-peer (flat int64 indices, dequantized values)
        IN CERTIFICATE UID ORDER. dense_contribs: per dense param (fp32 router /
        balance biases), per-peer full tensors in the same order. peer_norms: per
        param, one pre-clip l2 norm per peer, same order. Params with no entry
        in either dict are left untouched.
        """
        mu = self.cfg.momentum
        lr = self.cfg.lr
        per_param_l2: dict[str, float] = {}
        total_sq = 0.0
        applied_peers = 0

        for name in sorted(named_params):
            p = named_params[name]
            shape = self._shapes.get(name)
            if shape is None:
                raise KeyError(f"param {name!r} not registered in ReplicatedOuterStep")
            if tuple(p.shape) != tuple(shape):
                raise ValueError(f"param {name!r}: shape {tuple(p.shape)} != registered {tuple(shape)}")

            sparse = peer_sparse.get(name)
            dense = dense_contribs.get(name)
            if sparse:
                factors = self._clip_factors(name, len(sparse), peer_norms)
                idx_list = [i for i, _ in sparse]
                val_list = [
                    v.detach().reshape(-1).cpu().to(torch.float32) * factors[k]
                    for k, (_, v) in enumerate(sparse)
                ]
                g = deterministic_segment_mean(idx_list, val_list, shape.numel()).view(shape)
                n_peers = len(sparse)
            elif dense:
                factors = self._clip_factors(name, len(dense), peer_norms)
                g = torch.zeros(shape, dtype=torch.float32)
                for k, contrib in enumerate(dense):  # fixed certificate-UID accumulation order
                    if tuple(contrib.shape) != tuple(shape):
                        raise ValueError(f"dense contrib for {name!r} has shape {tuple(contrib.shape)}")
                    g += contrib.detach().cpu().to(torch.float32) * factors[k]
                g /= len(dense)
                n_peers = len(dense)
            else:
                continue

            m = self._momentum[name]
            if self.cfg.kind == "nesterov":
                m.mul_(mu).add_(g)
                d = g.add(m, alpha=mu)
            else:  # plain sgd: momentum buffer intentionally untouched
                d = g

            p_f32 = p.detach().to(device="cpu", dtype=torch.float32, copy=True)
            p_f32.sub_(d, alpha=lr)
            p.copy_(p_f32.to(dtype=p.dtype, device=p.device))

            l2 = float(torch.linalg.vector_norm(g))
            per_param_l2[name] = l2
            total_sq += l2 * l2
            applied_peers = max(applied_peers, n_peers)

        return OuterReport(
            global_grad_l2=math.sqrt(total_sq),
            per_param_l2=per_param_l2,
            applied_peers=applied_peers,
        )

    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Momentum buffers (cloned), keyed by param name — goes into the DCP checkpoint."""
        return {name: buf.detach().clone() for name, buf in self._momentum.items()}

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        if set(state) != set(self._momentum):
            missing = sorted(set(self._momentum) - set(state))
            extra = sorted(set(state) - set(self._momentum))
            raise ValueError(f"momentum state mismatch: missing={missing} extra={extra}")
        for name, buf in self._momentum.items():
            t = state[name]
            if tuple(t.shape) != tuple(buf.shape):
                raise ValueError(f"momentum {name!r}: shape {tuple(t.shape)} != {tuple(buf.shape)}")
            buf.copy_(t.detach().cpu().to(torch.float32))

    @property
    def param_names(self) -> Sequence[str]:
        return tuple(self._shapes)
