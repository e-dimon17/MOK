"""Chain weight submission: the emissions ladder over final scores.

Pure math lives in `C.core.scoring.compute_weights`; this module binds it to
the RunConfig knobs (gather/reserve counts come from WindowConfig — the same
constants the certificate uses) and to the chain client. `ChainClient.
set_weights` performs the u16 normalization; we submit the normalized floats.
"""

from __future__ import annotations

import asyncio
from typing import Any

from C.core.scoring import compute_weights
from mok_core.config import RunConfig
from mok_core.telemetry import get_logger

__all__ = ["submit_weights", "weights_for"]

log = get_logger("app.validator.weights")


def weights_for(final_scores: dict[int, float], cfg: RunConfig) -> dict[int, float]:
    """The gather/reserve emissions ladder for the current final scores."""
    return compute_weights(
        final_scores,
        cfg.scoring,
        gather_count=cfg.window.gather_peer_count,
        reserve_count=cfg.window.reserve_peer_count,
    )


async def submit_weights(
    chain: Any, final_scores: dict[int, float], cfg: RunConfig
) -> dict[int, float] | None:
    """Compute and submit weights; returns what was submitted (None if nothing).

    An empty ladder (no positive final scores yet — e.g. run start) is skipped
    rather than submitted: zero-weight extrinsics waste fees and reset nothing.
    """
    weights = weights_for(final_scores, cfg)
    if not weights:
        log.info("no positive final scores — skipping set_weights")
        return None
    ok = await asyncio.to_thread(chain.set_weights, weights)
    if not ok:
        log.warning("set_weights rejected by chain", peers=len(weights))
        return None
    log.info("weights submitted", peers=len(weights), top=max(weights.values()))
    return weights
