"""Seed-42 initialization publication and verification (playbook step B).

Owner side (``build_and_publish_init``): deterministically initialize the run
model (``mok_core.model.init_model`` — a pure function of (cfg, seed)), hash
the master weights into the run's ``init_checkpoint_hash`` (``state_root``),
save the window-0 checkpoint through ``C.core.checkpoint.Checkpointer`` (the
exact layout every later checkpoint uses: ``model/`` DCP + ``outer_state.pt``
with fresh zero outer momentum + canonical ``meta.json``), mirror it to the
owner's bucket, and commit the root on-chain as a ``ManifestCommit`` — the
same wire the run manifest hash uses (``ChainClient.commit_manifest_hash``).

Miner side (``fetch_and_verify_init``): download the newest complete
checkpoint from the owner's bucket via ``Checkpointer.load_latest``, recompute
``hash_named_tensors`` over the loaded masters, and refuse to start unless it
equals the expected root bitwise (and, when ``owner_uid`` is given, the root
committed on-chain). A miner that passes this check starts the run at the
same θ_start(0) as everyone else — the precondition of window-0 lockstep.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import torch

from C.core.checkpoint import Checkpointer, CheckpointMeta
from C.core.outer_opt import ReplicatedOuterStep
from mok_core.config import RunConfig
from mok_core.config.schemas import BucketCreds
from mok_core.determinism import hash_named_tensors
from mok_core.model import init_model, reference_config
from mok_core.storage import StorageClient

__all__ = [
    "DEFAULT_INIT_SEED",
    "INIT_WINDOW",
    "InitPublishError",
    "build_and_publish_init",
    "fetch_and_verify_init",
]

DEFAULT_INIT_SEED = 42  # consensus constant (playbook: "seed-42 init")
INIT_WINDOW = 0


class InitPublishError(RuntimeError):
    pass


async def build_and_publish_init(
    cfg: RunConfig,
    storage: StorageClient | None,
    chain: Any | None,
    *,
    local_dir: str | Path,
    seed: int = DEFAULT_INIT_SEED,
    device: str | torch.device = "cpu",
    backend: str = "reference",
    manifest_hash: str = "",
    spec_version: int = 1,
) -> str:
    """Build, checkpoint, upload and commit the run initialization; returns the
    init ``state_root``.

    ``manifest_hash`` is empty by design: the manifest cannot exist yet because
    it embeds this very root as ``init_checkpoint_hash`` (the owner re-commits
    the real manifest hash right after building the manifest). ``storage=None``
    keeps the checkpoint local only; ``chain=None`` skips the on-chain commit
    (both useful for offline/dry runs and tests of the pure path).
    """
    model_cfg = reference_config(cfg.model) if backend == "reference" else cfg.model
    model = init_model(model_cfg, seed, device=device, backend=backend)
    master = dict(model.iter_master_params())
    root = hash_named_tensors(master.items())

    outer = ReplicatedOuterStep(cfg.outer, {n: torch.Size(t.shape) for n, t in master.items()})
    meta = CheckpointMeta(
        window=INIT_WINDOW,
        global_step=0,
        tokens_consumed=0,
        state_root=root,
        manifest_hash=manifest_hash,
        spec_version=spec_version,
    )
    checkpointer = Checkpointer(storage, local_dir)
    await checkpointer.save(INIT_WINDOW, master, outer.state_dict(), meta)

    if chain is not None:
        # ManifestCommit wire (ChainClient.commit_manifest_hash); blocking+retrying,
        # so keep it off the event loop like exchange.put_window_payload does.
        await asyncio.to_thread(chain.commit_manifest_hash, root)
    return root


async def fetch_and_verify_init(
    storage: StorageClient,
    chain: Any | None,
    expected_root: str,
    *,
    local_dir: str | Path,
    bucket: BucketCreds | None = None,
    owner_uid: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], CheckpointMeta]:
    """Miner-side init loader: fetch, then verify BITWISE against ``expected_root``.

    Returns ``(master_state, outer_state, meta)`` exactly as
    ``Checkpointer.load_latest`` does; raises ``InitPublishError`` on any
    mismatch (wrong root, wrong window, or a chain commitment that disagrees).
    """
    expected = expected_root.lower()
    checkpointer = Checkpointer(storage, local_dir)
    loaded = await checkpointer.load_latest(bucket=bucket)
    if loaded is None:
        raise InitPublishError("no init checkpoint found locally or in the owner bucket")
    state, outer_state, meta = loaded
    if meta.window != INIT_WINDOW:
        raise InitPublishError(f"init checkpoint is for window {meta.window}, expected {INIT_WINDOW}")
    if meta.state_root != expected:
        raise InitPublishError(
            f"checkpoint meta.state_root {meta.state_root} != expected {expected}"
        )
    actual = hash_named_tensors(state.items())
    if actual != expected:
        raise InitPublishError(
            f"loaded init hashes to {actual}, expected {expected} — refusing to start desynced"
        )
    if chain is not None and owner_uid is not None:
        committed = await asyncio.to_thread(chain.get_manifest_hash, owner_uid)
        if committed != expected:
            raise InitPublishError(
                f"on-chain init commitment {committed!r} (uid {owner_uid}) != expected {expected}"
            )
    return state, outer_state, meta
