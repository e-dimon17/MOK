"""Shared builders for the GPU suite: synthetic dataset, toy config, mok model.

Importable two ways:
  - by `tests/gpu/conftest.py` and the test modules (pytest puts this
    directory on sys.path in non-package mode), and
  - as a torchrun ENTRY SCRIPT for the compile-cache determinism check
    (tests/gpu/test_04_compile_cache.py launches
    `torchrun --standalone --nproc-per-node=8 tests/gpu/_synthetic.py ...`),
    so the subprocess runs byte-identical window logic to the fixtures.

The synthetic corpus is the proven pattern from tests/unit/test_inner_loop.py:
a learnable 4-token cycle (3 -> 5 -> 7 -> 11) with a per-shard phase shift plus
one unique constant row per shard, which keeps every shard's bytes (and hence
Merkle leaves) distinct. Geometry is sized for the toy4L phase at world_size 8:
8 shards x 128 sequences x 4096 tokens = 1024 sequences, enough for a full
20-step window (640 sequences across 8 ranks).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

# When torchrun executes this file as a script, sys.path[0] is tests/gpu (not
# the repo root), so the subnet/mok_core packages need an explicit path entry when
# the wheel is not pip-installed. A no-op under pytest and in the container.
_REPO_ROOT_STR = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

from mok_core.config import RunConfig  # noqa: E402
from mok_core.config.loader import load_run_config  # noqa: E402
from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest  # noqa: E402
from mok_core.data import DatasetShardIndex, ShardReader, shard_leaf_hash  # noqa: E402
from mok_core.model import MoKTransformer, init_model  # noqa: E402
from subnet.core.inner_loop import InnerLoop  # noqa: E402
from subnet.core.phase import PhaseConfig, resolve_phase  # noqa: E402
from subnet.core.window_runner import TorchDistRunnerComm, build_window_plan, shared_master_root  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TOY_BASE_YAML = REPO_ROOT / "subnet" / "configs" / "base.yaml"
TOY_OVERLAY_YAML = REPO_ROOT / "subnet" / "configs" / "toy4L.yaml"

# Consensus-style constants for the whole GPU suite (one window, one uid).
RUN_SEED = bytes(range(32))
INIT_SEED = 42
UID = 3
WINDOW = 2
SEQ_LEN = 4096          # == toy4L model.seq_len
NUM_SHARDS = 8
SEQS_PER_SHARD = 128
CYCLE = (3, 5, 7, 11)


# --------------------------------------------------------------------------- #
# Synthetic dataset
# --------------------------------------------------------------------------- #


def shard_array(shard_idx: int) -> np.ndarray:
    """[SEQS_PER_SHARD, SEQ_LEN] uint16; learnable cycle + one unique row."""
    rows = []
    for r in range(SEQS_PER_SHARD):
        if r == SEQS_PER_SHARD - 1:
            rows.append(np.full(SEQ_LEN, 100 + shard_idx, dtype="<u2"))  # unique bytes per shard
        else:
            phase = (shard_idx + r) % len(CYCLE)
            rows.append(np.array([CYCLE[(phase + j) % len(CYCLE)] for j in range(SEQ_LEN)], dtype="<u2"))
    return np.stack(rows)


def write_shard_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(NUM_SHARDS):
        path = root / f"shard-{i}.bin"
        if not path.exists():  # idempotent: rank 0 may re-enter across sessions
            path.write_bytes(shard_array(i).tobytes())


def build_index(data_dir: Path) -> DatasetShardIndex:
    hashes = [shard_leaf_hash(data_dir / f"shard-{i}.bin").hex() for i in range(NUM_SHARDS)]
    if len(set(hashes)) != NUM_SHARDS:
        raise RuntimeError("synthetic shards must be byte-distinct")
    return DatasetShardIndex(name="bulk", seq_len=SEQ_LEN, shard_hashes=hashes)


def build_manifest(index: DatasetShardIndex) -> RunManifest:
    ref = DatasetManifestRef(
        name="bulk",
        merkle_root=index.merkle().root.hex(),
        num_shards=NUM_SHARDS,
        shard_bytes=2 * SEQ_LEN * SEQS_PER_SHARD,
        seq_len=SEQ_LEN,
        tokens_total=NUM_SHARDS * SEQS_PER_SHARD * SEQ_LEN,
        tokenizer_hash="ab" * 32,
    )
    return RunManifest(
        spec_version=1,
        run_id="gpu-suite",
        netuid=11,
        network="test",
        config_hash="11" * 32,
        container_digest="sha256:" + "22" * 32,
        mok_commit="deadbeef",
        tk_commit="cafebabe",
        attention_backend="cudnn_det",
        start_block=100,
        blocks_per_window=225,
        prf=PRFSpec(run_seed_hex=RUN_SEED.hex()),
        datasets=(ref,),
        init_checkpoint_hash="33" * 32,
    )


def make_shard_lookup_factory(data_dir: Path):
    """WindowReplayer-compatible shard_lookup_factory over the local shard files."""

    @contextmanager
    def factory(plan: Any) -> Iterator[Callable[[int], ShardReader]]:
        readers = {i: ShardReader(data_dir / f"shard-{i}.bin", SEQ_LEN) for i in set(plan.shard_ids)}
        try:
            yield readers.__getitem__
        finally:
            for reader in readers.values():
                reader.close()

    return factory


# --------------------------------------------------------------------------- #
# Toy config + model
# --------------------------------------------------------------------------- #


def load_toy_run_config(
    *,
    inner_steps: int | None = None,
    routed_precision: str | None = None,
    adam_reset_every_windows: int | None = None,
) -> RunConfig:
    """subnet/configs/base.yaml + toy4L.yaml through the real loader, with test knobs."""
    cfg = load_run_config(TOY_BASE_YAML, TOY_OVERLAY_YAML)
    if inner_steps is not None:
        cfg = cfg.model_copy(update={"window": cfg.window.model_copy(update={"inner_steps": inner_steps})})
    if routed_precision is not None:
        cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"routed_precision": routed_precision})})
    if adam_reset_every_windows is not None:
        cfg = cfg.model_copy(
            update={"inner": cfg.inner.model_copy(update={"adam_reset_every_windows": adam_reset_every_windows})}
        )
    return cfg


def prepare_mok_model(model: MoKTransformer) -> None:
    """First MXFP8 quantization after init (the MoK README recipe). No-op for bf16."""
    if model.cfg.routed_precision == "mxfp8":
        from mok_core.model import MXFP8WeightManager  # mok imported lazily inside

        MXFP8WeightManager(model.moe_layers()).requantize_all_()


def build_mok_model(cfg: RunConfig, device: torch.device | str) -> MoKTransformer:
    """Deterministic seed-INIT_SEED mok-backend model, quant cache primed.

    Every rank calls this with the same seed, so every rank's expert shard
    holds identical values — a degenerate but perfectly legal θ for the
    determinism tests. Parity tests (test_02) instead reshard a full-expert
    reference init so experts are genuinely distinct.
    """
    model = init_model(cfg.model, INIT_SEED, device=device, backend="mok", mok_runtime=cfg.mok)
    prepare_mok_model(model)
    return model


def run_toy_window(
    cfg: RunConfig,
    manifest: RunManifest,
    data_dir: Path,
    *,
    rank: int,
    world_size: int,
    device: torch.device | str,
    comm: Any,
    window: int = WINDOW,
    uid: int = UID,
    model: MoKTransformer | None = None,
) -> tuple[str | None, MoKTransformer]:
    """One full toy inner-loop window; returns (state_root on rank 0, model)."""
    phase: PhaseConfig = resolve_phase(manifest, cfg, window)
    plan = build_window_plan(
        manifest, phase, run_seed=RUN_SEED, uid=uid, window=window, rank=rank, world_size=world_size
    )
    if model is None:
        model = build_mok_model(cfg, device)
    loop = InnerLoop(model, cfg, phase, rank=rank, world_size=world_size, comm=comm, device=device)
    with make_shard_lookup_factory(data_dir)(plan) as shard_lookup:
        loop.run_window(plan, shard_lookup, window, global_inner_step0=0, tokens_consumed0=0)
    root = shared_master_root(model, rank=rank, world_size=world_size, comm=comm)
    return root, model


# --------------------------------------------------------------------------- #
# torchrun entry point (the test_04 compile-cache probe)
# --------------------------------------------------------------------------- #


def _probe_main(argv: list[str] | None = None) -> int:
    """Run a short toy window under torchrun and print STATE_ROOT=<hex> on rank 0.

    test_04 launches this twice (fresh processes, shared inductor cache) with
    MOK_COMPILE=1 and asserts the printed roots are identical.
    """
    parser = argparse.ArgumentParser(description="MoK GPU compile-cache probe")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--inner-steps", type=int, default=5)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--uid", type=int, default=UID)
    args = parser.parse_args(argv)

    from mok_core.determinism import enforce_determinism

    enforce_determinism()  # before any CUDA context — the process is fresh

    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        cfg = load_toy_run_config(inner_steps=args.inner_steps)
        data_dir = Path(args.data_dir)
        manifest = build_manifest(build_index(data_dir))
        root, _ = run_toy_window(
            cfg,
            manifest,
            data_dir,
            rank=rank,
            world_size=world_size,
            device=torch.device("cuda", local_rank),
            comm=TorchDistRunnerComm(),
            window=args.window,
            uid=args.uid,
        )
        if rank == 0:
            print(f"STATE_ROOT={root}", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(_probe_main())
