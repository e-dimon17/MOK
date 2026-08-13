"""GPU-suite conftest: every test here is `@pytest.mark.gpu` automatically.

Launch on a Tier-A node (8x B300 SM103, NVLink):

    torchrun --standalone --nproc-per-node=8 -m pytest tests/gpu -m gpu -q

Design rules (enforced by review, exercised by `pytest --collect-only -m gpu`
on any CPU host):
  - collection never touches CUDA, torch.distributed init, or the `mok` wheel;
  - every CUDA/mok dependency is guarded by a fixture that SKIPS with a clear
    reason when the hardware/wheel is absent — no fake passes;
  - all ranks execute the same tests in the same order (pytest's deterministic
    collection), so collective calls inside tests stay in lockstep.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import _synthetic as synth
import pytest
import torch

from mok_core.config import RunConfig
from mok_core.config.manifest import RunManifest
from mok_core.data import DatasetShardIndex

_GPU_DIR = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: Any, items: list[pytest.Item]) -> None:
    """Auto-apply @pytest.mark.gpu to everything collected from this directory."""
    for item in items:
        try:
            path = Path(item.path).resolve()
        except (TypeError, ValueError):  # pragma: no cover — non-file items
            continue
        if _GPU_DIR in path.parents:
            item.add_marker(pytest.mark.gpu)


# --------------------------------------------------------------------------- #
# Distributed context
# --------------------------------------------------------------------------- #


@dataclass
class DistCtx:
    """One rank's view of the torchrun process group."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    comm: Any  # TorchDistRunnerComm — satisfies both Comm and RunnerComm protocols

    def barrier(self) -> None:
        self.comm.barrier()


@pytest.fixture(scope="session")
def dist_ctx() -> Any:
    """torch.distributed init from the torchrun environment + device per LOCAL_RANK.

    Runs `enforce_determinism()` (env pins, deterministic algorithms) before
    the CUDA context is created — the same order every miner process uses.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip(
            "no torchrun environment (RANK/WORLD_SIZE unset) — launch via "
            "`torchrun --standalone --nproc-per-node=8 -m pytest tests/gpu -m gpu`"
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable — the GPU suite needs a Tier-A (8x B300) node")

    from mok_core.determinism import enforce_determinism

    enforce_determinism(allow_uninitialized_cuda_check=not torch.cuda.is_initialized())

    import torch.distributed as dist

    from C.core.window_runner import TorchDistRunnerComm

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ["RANK"]))
    if local_rank >= torch.cuda.device_count():
        pytest.skip(f"LOCAL_RANK {local_rank} >= visible CUDA devices {torch.cuda.device_count()}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(backend="nccl", device_id=device)
    ctx = DistCtx(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        device=device,
        comm=TorchDistRunnerComm(),
    )
    yield ctx
    dist.barrier()
    if owns_group:
        dist.destroy_process_group()


@pytest.fixture(scope="session")
def mok_available() -> Any:
    """Skip-guard for the MoK megakernel: SM103 (B300) + the `mok` wheel.

    Returns the imported `mok` package so tests never import it at module level.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable — MoK needs an SM103 (B300) node")
    capability = torch.cuda.get_device_capability(0)
    if capability != (10, 3):
        pytest.skip(f"MoK requires SM103 (sm_103); this device reports sm_{capability[0]}{capability[1]}")
    try:
        import mok
    except ImportError as exc:  # pragma: no cover — wheel present on Tier-A nodes
        pytest.skip(f"mok wheel not importable: {exc}")
    return mok


# --------------------------------------------------------------------------- #
# Config + data fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def toy_cfg() -> RunConfig:
    """C/configs/base.yaml + toy4L.yaml through the real loader (CPU-safe)."""
    return synth.load_toy_run_config()


class ToyData(NamedTuple):
    manifest: RunManifest
    index: DatasetShardIndex
    data_dir: Path


@pytest.fixture(scope="session")
def shared_tmp(dist_ctx: DistCtx) -> Any:
    """A tmp directory with the SAME path on every rank of this torchrun job.

    Per-process pytest tmp factories give each rank a different directory, so
    shared artifacts (shard files) live under a job-keyed path instead:
    rank 0 creates and removes it, everyone else synchronizes on barriers.
    """
    job_key = os.environ.get("TORCHELASTIC_RUN_ID") or os.environ.get("MASTER_PORT") or "local"
    root = Path(tempfile.gettempdir()) / f"mok-gpu-suite-{job_key}"
    if dist_ctx.rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    dist_ctx.barrier()
    yield root
    dist_ctx.barrier()
    if dist_ctx.rank == 0:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def toy_dataset(dist_ctx: DistCtx, shared_tmp: Path) -> ToyData:
    """Tiny synthetic dataset on the shared tmp: rank 0 writes, everyone verifies.

    8 shards x 128 sequences x 4096 tokens — enough for a full toy4L window at
    world_size 8 and byte-distinct per shard (distinct Merkle leaves).
    """
    data_dir = shared_tmp / "shards"
    if dist_ctx.rank == 0:
        synth.write_shard_files(data_dir)
    dist_ctx.barrier()
    index = synth.build_index(data_dir)
    return ToyData(manifest=synth.build_manifest(index), index=index, data_dir=data_dir)
