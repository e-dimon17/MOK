"""The attestation reference run — a deterministic toy window through the REAL
training path (playbook step B, "run a reference 20-step deterministic MoK
training snippet from a chain-published seed").

Self-contained by construction: the token stream is synthesized from
``philox(challenge.seed)`` (no real dataset shards needed), written to temp
uint16 shard files, indexed into a ``DatasetShardIndex`` and wrapped in a
minimal ``RunManifest`` — so ``WindowBatchPlan`` and ``C.core.inner_loop.
InnerLoop`` run UNMODIFIED. Attestation therefore exercises exactly the code
path that mines real windows; a node that passes attestation is a node whose
inner loop replays bitwise.

Consensus derivations (SPEC_VERSION-bound, golden-pinned in
``tests/unit/test_stepB_reference_step.py``):

  - data seed:   ``run_seed = blake2b-256(b"attest.data.v1" ‖ raw8(challenge_id)
    ‖ le64(seed))`` — the PRF run seed of the synthetic dataset.
  - token wire:  ``philox(seed).integers(0, vocab, size=(rows, seq_len),
    dtype=uint16)`` row-major; shard ``i`` holds rows
    ``[i*seqs_per_shard, (i+1)*seqs_per_shard)`` as little-endian bytes.
  - sizing:      the dataset always covers the full Tier-A geometry
    (``ATTEST_DATA_WORLD_SIZE = 8``) split over ``ATTEST_NUM_SHARDS = 4``
    shards, so shard bytes are identical no matter how many ranks replay.
  - state root:  ``attestation_state_root`` — for world_size 1 this is plainly
    ``hash_named_tensors(iter_master_params())``; for an EP-sharded run each
    rank's expert-local tensors are gathered and concatenated along dim 0 in
    rank order (the protocol EP geometry: rank r hosts experts
    ``[r*E_local, (r+1)*E_local)``), which reproduces the ``ep_size=1``
    reference layout — so an 8-rank mok run and a 1-rank reference run of the
    same challenge hash to THE SAME root.

``main()`` is the miner-side CLI: launched under ``torchrun --nproc-per-node=8``
inside the blessed container (``entrypoint.sh attest``); rank 0 emits the
``AttestationResponse`` as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from C.core.inner_loop import InnerLoop
from C.core.phase import resolve_phase
from C.core.window_runner import RunnerComm, SingleNodeComm, TorchDistRunnerComm, build_window_plan
from mok_core.config import RunConfig, config_hash
from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest
from mok_core.config.schemas import FrozenModel
from mok_core.data import DatasetShardIndex, ShardReader, verify_index_matches_ref
from mok_core.data.assignment import sequences_per_window
from mok_core.determinism import (
    enforce_determinism,
    environment_fingerprint,
    hash_named_tensors,
    philox,
)
from mok_core.model import MoKTransformer, init_model, reference_config

from .challenge import AttestationChallenge, challenge_run_config

__all__ = [
    "ATTEST_DATA_DOMAIN",
    "ATTEST_DATA_WORLD_SIZE",
    "ATTEST_NUM_SHARDS",
    "ATTEST_TOKENIZER_HASH",
    "ATTEST_UID",
    "ATTEST_WINDOW",
    "AttestationResponse",
    "attestation_manifest",
    "attestation_run_seed",
    "attestation_state_root",
    "build_parser",
    "main",
    "run_reference",
    "write_attestation_shards",
]

# Consensus constants — change requires SPEC_VERSION bump.
ATTEST_DATA_DOMAIN = b"attest.data.v1"
ATTEST_UID = 0
ATTEST_WINDOW = 0
ATTEST_NUM_SHARDS = 4
ATTEST_DATA_WORLD_SIZE = 8  # dataset sized for the full Tier-A node regardless of replay ranks
ATTEST_TOKENIZER_HASH = hashlib.blake2b(b"mok.attest.synthetic", digest_size=32).hexdigest()


class AttestationResponse(FrozenModel):
    """What a responder returns: the root, how long it took, and where it ran."""

    challenge_id: str
    state_root: str
    wall_time_s: float
    fingerprint: dict[str, Any]


def attestation_run_seed(challenge: AttestationChallenge) -> bytes:
    """The 32-byte PRF run seed of the challenge's synthetic dataset (see module docstring)."""
    h = hashlib.blake2b(digest_size=32)
    h.update(ATTEST_DATA_DOMAIN)
    h.update(bytes.fromhex(challenge.challenge_id))
    h.update(challenge.seed.to_bytes(8, "little"))
    return h.digest()


def write_attestation_shards(
    challenge: AttestationChallenge, cfg: RunConfig, out_dir: Path
) -> tuple[DatasetShardIndex, DatasetManifestRef]:
    """Synthesize the challenge dataset into ``out_dir`` (module-docstring wire).

    Every rank writes the same bytes (philox is counter-based, so this is
    platform- and rank-independent); the returned index/ref make
    ``WindowBatchPlan`` and ``ShardCache``-style verification work unmodified.
    """
    seq_len = cfg.model.seq_len
    total = sequences_per_window(
        tokens_per_rank_microbatch=cfg.window.tokens_per_rank_microbatch,
        grad_accum=cfg.window.grad_accum,
        inner_steps=cfg.window.inner_steps,
        ranks=ATTEST_DATA_WORLD_SIZE,
        seq_len=seq_len,
    )
    seqs_per_shard = math.ceil(total / ATTEST_NUM_SHARDS)
    gen = philox(challenge.seed)
    tokens = gen.integers(
        0, cfg.model.vocab_size, size=(ATTEST_NUM_SHARDS * seqs_per_shard, seq_len), dtype=np.uint16
    ).astype("<u2", copy=False)

    hashes: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(ATTEST_NUM_SHARDS):
        data = tokens[i * seqs_per_shard : (i + 1) * seqs_per_shard].tobytes()
        (out_dir / f"shard-{i:04d}.bin").write_bytes(data)
        hashes.append(hashlib.blake2b(data, digest_size=32).hexdigest())

    index = DatasetShardIndex(name="attest", seq_len=seq_len, shard_hashes=hashes)
    ref = DatasetManifestRef(
        name="attest",
        merkle_root=index.merkle().root.hex(),
        num_shards=ATTEST_NUM_SHARDS,
        shard_bytes=2 * seqs_per_shard * seq_len,
        seq_len=seq_len,
        tokens_total=ATTEST_NUM_SHARDS * seqs_per_shard * seq_len,
        tokenizer_hash=ATTEST_TOKENIZER_HASH,
    )
    verify_index_matches_ref(index, ref)
    return index, ref


def attestation_manifest(
    challenge: AttestationChallenge, cfg: RunConfig, ref: DatasetManifestRef
) -> RunManifest:
    """The minimal manifest wrapping the synthetic dataset — a pure function of
    (challenge, cfg), so every rank and every verifier builds the same plan."""
    return RunManifest(
        spec_version=1,
        run_id=f"attest-{challenge.challenge_id}",
        netuid=cfg.chain.netuid,
        network=cfg.chain.network,
        config_hash=config_hash(cfg),
        container_digest="attest",
        mok_commit="attest",
        tk_commit="attest",
        attention_backend="cudnn_det",
        start_block=challenge.issued_block,
        blocks_per_window=1,
        prf=PRFSpec(run_seed_hex=attestation_run_seed(challenge).hex()),
        datasets=(ref,),
        init_checkpoint_hash="00" * 32,
    )


def attestation_state_root(
    model: MoKTransformer, *, comm: RunnerComm, rank: int, world_size: int
) -> str | None:
    """The geometry-independent state root of ``model`` (rank 0; None elsewhere).

    See the module docstring: expert-local tensors are gathered across ranks and
    concatenated along the expert dim in rank order, replicated tensors come
    from rank 0's copy — reproducing the ``ep_size=1`` layout bitwise.
    """
    master = dict(model.iter_master_params())
    if world_size == 1:
        return hash_named_tensors(master.items())
    expert_local = {
        name: t.detach().cpu() for name, t in master.items() if model.is_expert_local(name)
    }
    gathered = comm.gather_object(expert_local)
    if gathered is None:
        return None
    combined = {name: t for name, t in master.items() if not model.is_expert_local(name)}
    for name in expert_local:
        parts = []
        for rank_idx, rank_tensors in enumerate(gathered):
            if name not in rank_tensors:
                raise ValueError(f"rank {rank_idx} did not contribute expert tensor {name!r}")
            parts.append(rank_tensors[name])
        combined[name] = torch.cat(parts, dim=0)
    return hash_named_tensors(combined.items())


def run_reference(
    challenge: AttestationChallenge,
    *,
    backend: str = "mok",
    device: str | torch.device,
    comm: RunnerComm | None = None,
) -> AttestationResponse:
    """Execute the challenge: init from ``challenge.seed``, run the toy window
    through the unmodified ``InnerLoop``, return the θ_end root.

    Rank/world_size come from ``torch.distributed`` when initialized (torchrun
    launch), else 1. The root is computed cooperatively and broadcast, so the
    returned response is identical on every rank. ``backend='reference'`` with
    a CPU device is the verifier/test path; miners run ``backend='mok'`` on
    the 8-GPU node.
    """
    rank, world_size = _dist_geometry()
    if comm is None:
        comm = SingleNodeComm() if world_size == 1 else TorchDistRunnerComm()
    cfg = challenge_run_config(challenge)
    model_cfg = reference_config(cfg.model) if backend == "reference" else cfg.model

    t0 = time.perf_counter()
    model = init_model(model_cfg, challenge.seed, device=device, backend=backend)
    with tempfile.TemporaryDirectory(prefix=f"mok-attest-{challenge.challenge_id}-") as tmp:
        shard_dir = Path(tmp)
        index, ref = write_attestation_shards(challenge, cfg, shard_dir)
        manifest = attestation_manifest(challenge, cfg, ref)
        phase = resolve_phase(manifest, cfg, ATTEST_WINDOW)
        plan = build_window_plan(
            manifest,
            phase,
            run_seed=attestation_run_seed(challenge),
            uid=ATTEST_UID,
            window=ATTEST_WINDOW,
            rank=rank,
            world_size=world_size,
        )
        readers = {
            i: ShardReader(shard_dir / f"shard-{i:04d}.bin", cfg.model.seq_len)
            for i in set(plan.shard_ids)
        }
        try:
            for i, reader in readers.items():
                if not reader.verify(index.leaf(i)):
                    raise RuntimeError(f"synthetic shard {i} failed its own hash check")
            inner = InnerLoop(
                model, cfg, phase, rank=rank, world_size=world_size, comm=comm, device=device
            )
            inner.run_window(
                plan, readers.__getitem__, ATTEST_WINDOW, global_inner_step0=0, tokens_consumed0=0
            )
        finally:
            for reader in readers.values():
                reader.close()

    root = attestation_state_root(model, comm=comm, rank=rank, world_size=world_size)
    root = comm.broadcast_object(root, 0)
    assert root is not None
    return AttestationResponse(
        challenge_id=challenge.challenge_id,
        state_root=root,
        wall_time_s=time.perf_counter() - t0,
        fingerprint=environment_fingerprint().to_json(),
    )


# --------------------------------------------------------------------------- #
# CLI (torchrun entry — `entrypoint.sh attest`)
# --------------------------------------------------------------------------- #


def _dist_geometry() -> tuple[int, int]:
    import torch.distributed as dist  # noqa: PLC0415 — optional at runtime

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m B.attestation.reference_step",
        description="Run an attestation challenge and emit the response JSON (rank 0).",
    )
    p.add_argument("--challenge", required=True, help="challenge JSON file, or '-' for stdin")
    p.add_argument("--backend", choices=("mok", "reference"), default="mok")
    p.add_argument("--device", default=None, help="default: cuda:<local_rank> if available, else cpu")
    p.add_argument("--out", default="-", help="response JSON destination (rank 0), '-' for stdout")
    return p


def _load_challenge(source: str) -> AttestationChallenge:
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    return AttestationChallenge.model_validate(json.loads(raw))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    challenge = _load_challenge(args.challenge)
    enforce_determinism()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    started_dist = False
    if world_size > 1:
        import torch.distributed as dist  # noqa: PLC0415

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
            started_dist = True
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = args.device or (f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))

    try:
        response = run_reference(challenge, backend=args.backend, device=device)
        rank, _ = _dist_geometry()
        if rank == 0:
            payload = json.dumps(response.model_dump(), sort_keys=True)
            if args.out == "-":
                print(payload)
            else:
                Path(args.out).write_text(payload + "\n", encoding="utf-8")
    finally:
        if started_dist:
            import torch.distributed as dist  # noqa: PLC0415

            dist.destroy_process_group()
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via subprocess in GPU runs
    raise SystemExit(main())
