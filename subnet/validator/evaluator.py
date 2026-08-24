"""Per-window miner evaluation: loss deltas on deterministic eval pools.

For every evaluated miner the validator, holding the reference replica at
θ_start(window):

  1. draws the miner's OWN pool (a subset of its assigned window data) and the
     shared RANDOM pool (held-out draw over the whole tree) via
     `subnet.core.scoring.EvalPools` — both seeded by a post-commit block hash, so
     every validator evaluates identical slices the miner could not predict;
  2. measures CE loss before and after applying the miner's decompressed
     pseudo-gradient at `outer.lr * scoring.eval_lr_factor` to a scratch copy
     (θ restored bitwise via the pinned-CPU snapshot machinery afterwards);
  3. reduces to `gradient_score` (relative own-pool improvement) and the
     anti-fake `binary_indicator` (own improvement must beat random improvement).

Nothing here is consensus-bearing for the training lineage — evaluation reads
θ and restores it — but pool draws are consensus functions across validators.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch

from mok_core.model import MoKTransformer, evaluate_sequences
from mok_core.telemetry import get_logger
from subnet.core.compress import TopKCompressor
from subnet.core.payload import WindowPayload
from subnet.core.phase import PhaseConfig
from subnet.core.pseudo_grad import CpuSnapshot, restore_and_extract_delta
from subnet.core.scoring import EvalPools, binary_indicator, gradient_score
from subnet.core.window_runner import DENSE_SUFFIX
from subnet.miner.bootstrap import NodeContext

__all__ = ["EvalRecord", "WindowEvaluator"]

log = get_logger("app.validator.evaluator")


@dataclass(frozen=True)
class EvalRecord:
    """One miner-window evaluation, ready for the trust pipeline."""

    uid: int
    own_before: float
    own_after: float
    random_before: float
    random_after: float
    score: float          # gradient_score on the miner's own pool
    indicator: int        # +1 / -1 anti-fake verdict
    own_sequences: int
    random_sequences: int


class WindowEvaluator:
    """Builds eval batches from the verified shard caches and scores payloads."""

    def __init__(self, ctx: NodeContext, *, pools: EvalPools | None = None) -> None:
        self.ctx = ctx
        self.pools = pools if pools is not None else EvalPools(world_size=ctx.protocol_world_size)

    # ------------------------------------------------------------------ #
    # Data plumbing
    # ------------------------------------------------------------------ #

    async def _batch_for_pairs(
        self, phase: PhaseConfig, pairs: Iterable[tuple[int, int]]
    ) -> torch.Tensor | None:
        """[B, seq_len] int64 batch for (shard_idx, seq_idx) pairs (cache-verified)."""
        pair_list = [(int(s), int(q)) for s, q in pairs]
        if not pair_list:
            return None
        cache = self.ctx.shard_caches[phase.data]
        await cache.prefetch({s for s, _ in pair_list}, self.ctx.fetch_fns[phase.data])
        from mok_core.data import ShardReader  # noqa: PLC0415 — keep module import surface tight

        rows: list[np.ndarray] = []
        readers: dict[int, ShardReader] = {}
        try:
            for shard_idx, seq_idx in pair_list:
                reader = readers.get(shard_idx)
                if reader is None:
                    reader = ShardReader(cache.path_for(shard_idx), phase.seq_len)
                    readers[shard_idx] = reader
                rows.append(reader.sequence(seq_idx % reader.num_sequences).astype(np.int64))
        finally:
            for reader in readers.values():
                reader.close()
        return torch.from_numpy(np.stack(rows))

    def _loss(self, model: MoKTransformer, batch: torch.Tensor | None) -> float:
        if batch is None:
            return float("nan")
        return evaluate_sequences(model, [batch], device=self.ctx.device)

    # ------------------------------------------------------------------ #
    # Scratch application of a miner's pseudo-gradient
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _apply_payload(
        self,
        master: Mapping[str, torch.Tensor],
        payload: WindowPayload,
        compressor: TopKCompressor,
        scale: float,
    ) -> None:
        """θ ← θ − scale · Δ for every parameter the payload carries."""
        for name, ct in payload.compressed.items():
            p = master.get(name)
            if p is None:
                continue
            delta = compressor.decompress(name, ct)
            p.data.add_(delta.to(device=p.device, dtype=p.dtype), alpha=-scale)
        for name, dense in payload.dense.items():
            p = master.get(name)
            if p is None or not name.endswith(DENSE_SUFFIX):
                continue
            p.data.add_(dense.to(device=p.device, dtype=p.dtype), alpha=-scale)

    # ------------------------------------------------------------------ #
    # The window evaluation
    # ------------------------------------------------------------------ #

    async def evaluate_window(
        self,
        model: MoKTransformer,
        window: int,
        phase: PhaseConfig,
        payloads: Mapping[int, WindowPayload],
        compressor: TopKCompressor,
        block_hash: bytes,
    ) -> dict[int, EvalRecord]:
        """Score every payload against the model at θ_start(window); θ restored."""
        if not payloads:
            return {}
        ctx = self.ctx
        n = ctx.cfg.scoring.eval_sequences
        scale = ctx.cfg.outer.lr * ctx.cfg.scoring.eval_lr_factor

        random_pairs = self.pools.random_pool(
            ctx.manifest, ctx.run_seed, window, phase, n, block_hash
        )
        random_batch = await self._batch_for_pairs(phase, random_pairs)
        random_before = self._loss(model, random_batch)

        master = dict(model.iter_master_params())
        snapshot = CpuSnapshot.take(master)
        records: dict[int, EvalRecord] = {}
        try:
            for uid in sorted(payloads):
                own_pairs = self.pools.own_pool(
                    ctx.manifest, ctx.run_seed, uid, window, phase, n, block_hash
                )
                own_batch = await self._batch_for_pairs(phase, own_pairs)
                own_before = self._loss(model, own_batch)

                self._apply_payload(master, payloads[uid], compressor, scale)
                own_after = self._loss(model, own_batch)
                random_after = self._loss(model, random_batch)
                restore_and_extract_delta(master, snapshot)  # bitwise θ_start restore

                score = gradient_score(own_before, own_after)
                imp_random = gradient_score(random_before, random_after)
                records[uid] = EvalRecord(
                    uid=uid,
                    own_before=own_before,
                    own_after=own_after,
                    random_before=random_before,
                    random_after=random_after,
                    score=score,
                    indicator=binary_indicator(score, imp_random),
                    own_sequences=len(own_pairs),
                    random_sequences=len(random_pairs),
                )
                log.info(
                    "miner evaluated",
                    window=window,
                    miner=uid,
                    score=round(score, 6),
                    indicator=records[uid].indicator,
                )
        finally:
            restore_and_extract_delta(master, snapshot)
        return records
