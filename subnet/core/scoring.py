"""Validator scoring: the trust pipeline from raw eval losses to chain weights.

Pipeline (one evaluated miner-window):
  loss deltas -> gradient_score / binary_indicator -> BinaryEMA
  window scores -> OpenSkillBook (PlackettLuce) -> ordinal
  debug-dict sync check -> sync_score
  final_score = max(0, ordinal) * max(0, ema) * sync
  compute_weights: gather/reserve emissions ladder over final scores.

EvalPools reconstructs eval data pools deterministically from the manifest so
every validator draws the identical eval slices (seeded by a post-commit block
hash — unpredictable to miners, identical across validators).

Gradient scoring, OpenSkill updates, final-score composition, and the
weights ladder. Design notes: no burn weight (plan decision: roll forward), sorted-uid
tie determinism everywhere, pure-dict outputs instead of a 256-slot tensor.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from openskill.models import PlackettLuce
from openskill.models.weng_lin.plackett_luce import PlackettLuceRating

from mok_core.config.manifest import RunManifest
from mok_core.config.schemas import ScoringConfig
from mok_core.determinism.seeding import eval_seed, philox

from .phase import PhaseConfig

#: Sentinel uid for the shared random eval pool (never a valid metagraph uid).
RANDOM_POOL_UID: int = 2**32 - 1

#: Bytes per token in the packed uint16 shard format (dataprep).
_TOKEN_BYTES: int = 2


# --------------------------------------------------------------------------- #
# Per-window score primitives
# --------------------------------------------------------------------------- #


def gradient_score(loss_before: float, loss_after: float) -> float:
    """Relative loss improvement (before - after) / before; 0.0 on degenerate input."""
    if not (math.isfinite(loss_before) and math.isfinite(loss_after)) or loss_before <= 0.0:
        return 0.0
    return (loss_before - loss_after) / loss_before


def binary_indicator(improvement_own: float, improvement_random: float) -> int:
    """+1 iff the miner improves its own assigned data more than held-out random data.

    A miner faking gradients (or overfitting a tiny slice) improves random data
    at least as much as its own -> -1. Equality counts against the miner.
    """
    return 1 if improvement_own > improvement_random else -1


class BinaryEMA:
    """Exponential moving average of the +/-1 binary indicator, with a per-uid warmup.

    Warmup is counted from the first update of each uid: `passes` is
    unconditionally True inside the warmup span, afterwards it requires
    max(0, ema) >= threshold (the binary-moving-average gate).
    """

    def __init__(self, alpha: float, threshold: float, warmup_windows: int) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.threshold = threshold
        self.warmup_windows = warmup_windows
        self._ema: dict[int, float] = {}
        self._first_window: dict[int, int] = {}

    def update(self, uid: int, indicator: int, window: int) -> float:
        """Fold one +/-1 indicator into uid's EMA; returns the new EMA value."""
        if indicator not in (-1, 1):
            raise ValueError(f"indicator must be +1 or -1, got {indicator}")
        self._first_window.setdefault(uid, window)
        new = (1.0 - self.alpha) * self._ema.get(uid, 0.0) + self.alpha * float(indicator)
        self._ema[uid] = new
        return new

    def value(self, uid: int) -> float:
        return self._ema.get(uid, 0.0)

    def passes(self, uid: int, window: int) -> bool:
        """Threshold gate; always True inside the uid's warmup span (or if unseen)."""
        first = self._first_window.get(uid)
        if first is None or window - first < self.warmup_windows:
            return True
        return max(0.0, self._ema.get(uid, 0.0)) >= self.threshold

    def reset(self, uid: int) -> None:
        self._ema.pop(uid, None)
        self._first_window.pop(uid, None)

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema": {str(uid): v for uid, v in self._ema.items()},
            "first_window": {str(uid): w for uid, w in self._first_window.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._ema = {int(uid): float(v) for uid, v in state.get("ema", {}).items()}
        self._first_window = {int(uid): int(w) for uid, w in state.get("first_window", {}).items()}


class OpenSkillBook:
    """PlackettLuce skill ratings over per-window gradient scores.

    Each window is one "match": participants are ranked by gradient score
    (higher is better) and rated jointly, so a rating reflects performance
    relative to the current peer set, with uncertainty (sigma) shrinking as
    evidence accumulates. `ordinal` = mu - 3*sigma (openskill default): a
    conservative skill floor that new peers start at 0.0.
    """

    def __init__(self, beta: float, tau: float) -> None:
        self.beta = beta
        self.tau = tau
        self._model = PlackettLuce(beta=beta, tau=tau)
        self._ratings: dict[int, PlackettLuceRating] = {}

    def _rating(self, uid: int) -> PlackettLuceRating:
        if uid not in self._ratings:
            self._ratings[uid] = self._model.rating(name=str(uid))
        return self._ratings[uid]

    def rate_window(self, scores: dict[int, float]) -> None:
        """Rate one window's participants against each other (no-op below 2 peers).

        Uids are processed in sorted order so score ties resolve identically on
        every validator regardless of dict insertion order.
        """
        if len(scores) < 2:
            for uid in scores:
                self._rating(uid)  # first-seen uids still get a default rating
            return
        uids = sorted(scores)
        teams = [[self._rating(uid)] for uid in uids]
        rated = self._model.rate(teams, scores=[float(scores[uid]) for uid in uids])
        for uid, team in zip(uids, rated, strict=True):
            self._ratings[uid] = team[0]

    def ordinal(self, uid: int) -> float:
        rating = self._ratings.get(uid)
        return float(rating.ordinal()) if rating is not None else 0.0

    def mu_sigma(self, uid: int) -> tuple[float, float]:
        rating = self._ratings.get(uid)
        if rating is None:
            rating = self._model.rating(name=str(uid))
        return float(rating.mu), float(rating.sigma)

    def reset(self, uid: int) -> None:
        """Drop uid's rating; it re-enters at the default on next appearance."""
        self._ratings.pop(uid, None)

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "tau": self.tau,
            "ratings": {
                str(uid): {"mu": float(r.mu), "sigma": float(r.sigma)}
                for uid, r in self._ratings.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._ratings = {
            int(uid): self._model.rating(mu=float(v["mu"]), sigma=float(v["sigma"]), name=str(uid))
            for uid, v in state.get("ratings", {}).items()
        }


def sync_score(avg_steps_behind: float, max_behind: float = 3.0) -> float:
    """Sync-score formula: max(0, (1 - min(x, max_behind)/max_behind))**2.5.

    1.0 when perfectly in lockstep, 0.0 at >= max_behind outer steps behind,
    with a superlinear penalty in between.
    """
    if max_behind <= 0.0:
        raise ValueError(f"max_behind must be positive, got {max_behind}")
    x = min(max(avg_steps_behind, 0.0), max_behind)
    return max(0.0, 1.0 - x / max_behind) ** 2.5


def final_score(uid: int, book: OpenSkillBook, bma: BinaryEMA, sync: float) -> float:
    """max(0, ordinal) * max(0, binary EMA) * sync — the emission-ranking score."""
    return max(0.0, book.ordinal(uid)) * max(0.0, bma.value(uid)) * sync


# --------------------------------------------------------------------------- #
# Emissions ladder
# --------------------------------------------------------------------------- #


def compute_weights(
    final_scores: dict[int, float],
    cfg: ScoringConfig,
    *,
    gather_count: int = 20,
    reserve_count: int = 10,
) -> dict[int, float]:
    """Piece-wise emissions ladder over positive final scores; sums to 1.0.

      gather set  (top `gather_count`):    share cfg.gather_share, linear ramp
                                           cfg.top_ratio : 1 from rank 1 to rank n
      reserve set (next `reserve_count`):  share cfg.reserve_share, geometric
                                           cfg.reserve_decay, capped below the
                                           smallest gather weight
      everyone else: 0 (omitted from the result)

    No burn (plan decision: roll forward). Fewer than 1 eligible -> {}. Ties in
    score rank deterministically by ascending uid. `gather_count`/`reserve_count`
    come from WindowConfig (gather_peer_count / reserve_peer_count).
    """
    eligible = [uid for uid, s in final_scores.items() if s > 0.0 and math.isfinite(s)]
    if not eligible:
        return {}
    ranked = sorted(eligible, key=lambda uid: (-final_scores[uid], uid))
    gather_uids = ranked[:gather_count]
    reserve_uids = ranked[gather_count : gather_count + reserve_count]

    weights: dict[int, float] = {}

    n = len(gather_uids)
    profile = (
        [cfg.top_ratio + (1.0 - cfg.top_ratio) * i / (n - 1) for i in range(n)] if n > 1 else [1.0]
    )
    profile_sum = sum(profile)
    for uid, p in zip(gather_uids, profile, strict=True):
        weights[uid] = cfg.gather_share * p / profile_sum

    if reserve_uids:
        r_profile = [cfg.reserve_decay**i for i in range(len(reserve_uids))]
        r_sum = sum(r_profile)
        for uid, p in zip(reserve_uids, r_profile, strict=True):
            weights[uid] = cfg.reserve_share * p / r_sum
        # Reserve must stay strictly below the gather floor.
        min_gather = min(weights[uid] for uid in gather_uids)
        max_reserve = max(weights[uid] for uid in reserve_uids)
        if max_reserve > min_gather:
            factor = min_gather / max_reserve * cfg.reserve_decay
            for uid in reserve_uids:
                weights[uid] *= factor

    total = sum(weights.values())
    return {uid: w / total for uid, w in weights.items()}


# --------------------------------------------------------------------------- #
# Deterministic eval pools
# --------------------------------------------------------------------------- #

PlanFactory = Callable[[RunManifest, bytes, int, int, PhaseConfig], Any]


def _plan_sequences(plan: Any) -> list[tuple[int, int]]:
    """(shard_idx, seq_idx) rows of a WindowBatchPlan (`global_pairs` — the
    miner's whole all-rank window pool) or of a raw (shard, seq) iterable."""
    pairs = getattr(plan, "global_pairs", None)
    if pairs is None:
        pairs = getattr(plan, "sequences", plan)
    return [(int(s), int(q)) for s, q in pairs]


def _sample_without_replacement(rng: np.random.Generator, population: int, k: int) -> list[int]:
    """First-k-distinct rejection sampling over [0, population).

    Deterministic given the Philox stream and memory-flat in `population`
    (numpy's choice(replace=False) materializes a full permutation). Consensus
    function: draw order is part of the protocol.
    """
    k = min(k, population)
    picked: list[int] = []
    seen: set[int] = set()
    while len(picked) < k:
        for v in rng.integers(0, population, size=max(k, 8)):
            iv = int(v)
            if iv not in seen:
                seen.add(iv)
                picked.append(iv)
                if len(picked) == k:
                    break
    return picked


class EvalPools:
    """Deterministic eval-slice construction, identical on every validator.

    Both pools are seeded by `eval_seed(run_seed, block_hash, uid)` where
    `block_hash` is a post-commit block: miners cannot predict their eval
    slices while training, yet all validators derive the same slices.

    `phase` is the resolved PhaseConfig (subnet/core/phase.resolve_phase) of the
    evaluated window. `plan_factory` builds a miner's WindowBatchPlan; the
    default lazily imports mok_core.data.window_dataset and rebuilds the
    miner's exact plan (rank 0 carries the full all-rank `global_pairs` pool).
    `world_size` is the miner's rank count (ModelConfig.ep_size).
    """

    def __init__(self, plan_factory: PlanFactory | None = None, *, world_size: int = 8) -> None:
        self.world_size = world_size
        self._plan_factory = plan_factory or self._build_plan

    def _build_plan(
        self, manifest: RunManifest, run_seed: bytes, uid: int, window: int, phase: PhaseConfig
    ) -> Any:
        from mok_core.data.window_dataset import WindowBatchPlan  # noqa: PLC0415

        return WindowBatchPlan.build(
            manifest,
            run_seed=run_seed,
            uid=uid,
            window=window,
            rank=0,
            world_size=self.world_size,
            tokens_per_rank_microbatch=phase.tokens_per_rank_microbatch,
            grad_accum=phase.grad_accum,
            inner_steps=phase.inner_steps,
            seq_len=phase.seq_len,
            dataset=phase.data,
        )

    def own_pool(
        self,
        manifest: RunManifest,
        run_seed: bytes,
        uid: int,
        window: int,
        phase: PhaseConfig,
        n_sequences: int,
        block_hash: bytes,
    ) -> list[tuple[int, int]]:
        """Eval subset of the miner's own assigned window data (anti-fake check)."""
        sequences = _plan_sequences(self._plan_factory(manifest, run_seed, uid, window, phase))
        if not sequences or n_sequences <= 0:
            return []
        rng = philox(eval_seed(run_seed, block_hash, uid))
        return [sequences[i] for i in _sample_without_replacement(rng, len(sequences), n_sequences)]

    def random_pool(
        self,
        manifest: RunManifest,
        run_seed: bytes,
        window: int,  # signature parity with own_pool; the pool keys on block_hash, not window
        phase: PhaseConfig,
        n_sequences: int,
        block_hash: bytes,
    ) -> list[tuple[int, int]]:
        """Held-out eval subset drawn from the whole active dataset tree.

        Seeded with the RANDOM_POOL_UID sentinel: one shared pool per
        (block_hash, tree), identical for every evaluated miner and validator.
        """
        dataset = manifest.dataset(phase.data)
        seqs_per_shard = dataset.shard_bytes // (dataset.seq_len * _TOKEN_BYTES)
        total = dataset.num_shards * seqs_per_shard
        if total <= 0 or n_sequences <= 0:
            return []
        rng = philox(eval_seed(run_seed, block_hash, RANDOM_POOL_UID))
        picks = _sample_without_replacement(rng, total, n_sequences)
        return [divmod(g, seqs_per_shard) for g in picks]
