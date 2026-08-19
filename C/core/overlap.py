"""Gradient-copy detection via top-k index overlap.

Honest SparseLoCo payloads from different miners share few top-k indices (each
miner compresses its own pseudo-gradient over its own data). A pair of payloads
whose index sets largely coincide means one peer republished another's work.
The offender is the peer that uploaded LATER — the copier had to wait for the
original — and severity escalates with the overlap fraction.

Pairwise top-k index-overlap detection with timestamp-based offender
attribution. Design notes: operates
on already-unpacked index tensors (payload.py owns 12-bit unpacking), returns a
typed report, deterministic tie-breaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

#: Severity ladder (plan decision): overlap -> (level, score multiplier, naughty).
_SEVERITY_LADDER: tuple[tuple[float, str, float, bool], ...] = (
    (0.6, "mega", 0.0, True),
    (0.5, "max", 0.0, False),
    (0.4, "high", 0.5, False),
)


@dataclass(frozen=True)
class OverlapPair:
    """One flagged peer pair; uid_a < uid_b; overlap = size-weighted mean index overlap."""

    uid_a: int
    uid_b: int
    overlap: float


@dataclass(frozen=True)
class OverlapReport:
    """Flagged pairs (overlap >= threshold) plus whole-window telemetry."""

    pairs: list[OverlapPair]
    pairs_checked: int = 0
    mean_overlap: float = 0.0


@dataclass(frozen=True)
class OverlapSeverity:
    level: str          # "none" | "high" | "max" | "mega"
    multiplier: float   # applied to the offender's final score this window
    naughty: bool       # True => offender also enters the naughty list

    # Convenience so callers can write `if sev:` for "any sanction".
    def __bool__(self) -> bool:
        return self.level != "none"


@dataclass
class _PairAccumulator:
    weighted_sum: float = 0.0
    weight: float = 0.0

    def merge(self, mean_frac: float, weight: float) -> None:
        self.weighted_sum += mean_frac * weight
        self.weight += weight

    @property
    def mean(self) -> float:
        return self.weighted_sum / self.weight if self.weight > 0 else 0.0


@dataclass
class _Totals:
    acc: dict[tuple[int, int], _PairAccumulator] = field(default_factory=dict)
    weighted_sum: float = 0.0
    weight: float = 0.0


#: Chunk rows compared per batch in `_pair_overlap` (bounds the K×K broadcast).
_PAIR_BATCH_CHUNKS = 16384


def _pair_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-chunk |setA ∩ setB| / k for two (..., k) index tensors.

    Indices within a chunk are distinct (top-k), so counting membership of a's
    indices in b's set equals the intersection size.
    """
    k = a.shape[-1]
    a_flat = a.reshape(-1, k)
    b_flat = b.reshape(-1, k)
    rows = a_flat.shape[0]
    # Batch the (C, K, K) comparison so peak memory stays bounded for
    # production-scale params (54B embed: C≈65k chunks) — result is exact.
    total = 0.0
    for i in range(0, rows, _PAIR_BATCH_CHUNKS):
        aa = a_flat[i : i + _PAIR_BATCH_CHUNKS]
        bb = b_flat[i : i + _PAIR_BATCH_CHUNKS]
        inter = (aa.unsqueeze(-1) == bb.unsqueeze(-2)).any(-1).sum(-1)  # (batch,)
        total += float((inter.to(torch.float32) / k).sum().item())
    return total / rows


def index_overlap_report(
    peer_indices: dict[int, dict[str, torch.Tensor]],
    threshold: float,
) -> OverlapReport:
    """All-pairs top-k index overlap over the gathered payloads.

    `peer_indices` maps uid -> {param_name: LongTensor(..., k)} of unpacked
    top-k indices (trailing dim = k, leading dims = chunks). For each pair the
    per-parameter overlap is averaged size-weighted (weight = index count);
    pairs at or above `threshold` are flagged. Only parameters present for both
    peers with matching shapes are compared. Deterministic: uids and parameter
    names are iterated in sorted order.
    """
    uids = sorted(peer_indices)
    totals = _Totals()

    for pos_a, uid_a in enumerate(uids):
        for uid_b in uids[pos_a + 1 :]:
            params_a, params_b = peer_indices[uid_a], peer_indices[uid_b]
            for name in sorted(params_a.keys() & params_b.keys()):
                a, b = params_a[name], params_b[name]
                if a.shape != b.shape or a.numel() == 0:
                    continue
                mean_frac = _pair_overlap(a, b)
                weight = float(a.numel())
                totals.acc.setdefault((uid_a, uid_b), _PairAccumulator()).merge(mean_frac, weight)
                totals.weighted_sum += mean_frac * weight
                totals.weight += weight

    flagged = [
        OverlapPair(uid_a=pair[0], uid_b=pair[1], overlap=acc.mean)
        for pair, acc in sorted(totals.acc.items())
        if acc.mean >= threshold
    ]
    return OverlapReport(
        pairs=flagged,
        pairs_checked=len(totals.acc),
        mean_overlap=totals.weighted_sum / totals.weight if totals.weight > 0 else 0.0,
    )


def determine_offender(pair: OverlapPair, upload_ts: dict[int, float]) -> int:
    """The peer that uploaded later is the copier; equal timestamps blame the
    higher uid (deterministic; registering later = strictly less trusted)."""
    ts_a, ts_b = upload_ts[pair.uid_a], upload_ts[pair.uid_b]
    if ts_a == ts_b:
        return max(pair.uid_a, pair.uid_b)
    return pair.uid_a if ts_a > ts_b else pair.uid_b


def severity(overlap: float) -> OverlapSeverity:
    """Map an overlap fraction to its sanction (see _SEVERITY_LADDER)."""
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be in [0, 1], got {overlap}")
    for floor, level, multiplier, naughty in _SEVERITY_LADDER:
        if overlap >= floor:
            return OverlapSeverity(level=level, multiplier=multiplier, naughty=naughty)
    return OverlapSeverity(level="none", multiplier=1.0, naughty=False)
