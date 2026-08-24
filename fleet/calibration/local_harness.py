"""The production loopback rig — a moto-free, single-node dress-rehearsal stack.

This is the loopback fixture pattern of ``tests/unit/test_window_runner.py``
promoted into production code (task spec): a real ``subnet.core.window_runner.
WindowRunner`` in ``self_leader`` bootstrap mode, wired to

  - ``MemoryStorage`` — a filesystem-backed stand-in implementing the exact
    ``mok_core.storage.StorageClient`` surface that ``subnet.core.exchange`` and
    ``subnet.core.checkpoint`` consume (put/get/list/head/gather with the same
    error types and failure-reason prefixes), and
  - ``ScriptedChain`` — a recording chain double for the runner's single
    chain touchpoint (``commit_window`` inside ``put_window_payload``), plus
    the deterministic ``block_hash`` scheduling helpers, and
  - ``LoopbackClock`` — a stepped window clock whose ``enter_gate`` puts
    "now" inside the upload gate of any window.

Calibration (``fleet.calibration.rehearsal`` / fleet dress rehearsal) and tests
both build on this; nothing here is test-only code.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from mok_core.chain.schemas import WindowCommit
from mok_core.config import RunConfig, config_hash
from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest
from mok_core.config.schemas import BucketCreds, StorageConfig
from mok_core.data import DatasetShardIndex, ShardCache
from mok_core.determinism import hash_bytes
from mok_core.model import MoKTransformer
from mok_core.storage import (
    GatherResult,
    IntegrityError,
    ObjectMissingError,
    ObjectTooLargeError,
    StorageError,
)
from subnet.core.checkpoint import Checkpointer
from subnet.core.compress import ChunkingTransformer, ErrorFeedback, Quantizer, TopKCompressor
from subnet.core.outer_opt import ReplicatedOuterStep
from subnet.core.window_runner import (
    DENSE_SUFFIX,
    RunState,
    SingleNodeComm,
    WindowOutcome,
    WindowRunner,
)

__all__ = [
    "LocalLoopbackHarness",
    "LoopbackClock",
    "MemoryStorage",
    "ScriptedChain",
    "local_manifest",
    "make_compressor",
    "make_outer_step",
]


# --------------------------------------------------------------------------- #
# MemoryStorage — the StorageClient surface, filesystem-backed
# --------------------------------------------------------------------------- #


class MemoryStorage:
    """Filesystem-backed ``StorageClient`` stand-in (async API, same error types).

    One instance plays every bucket: objects live under
    ``root/<bucket_name>/<key>``, writes go to the instance's own bucket
    (mirroring ``StorageClient.put_bytes`` semantics), reads accept any
    ``BucketCreds`` whose ``bucket_name`` exists under ``root``. Timestamps
    are recorded through the injectable ``clock`` (falling back to file
    mtime), so upload-gate tests can script time.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        creds: BucketCreds,
        *,
        cfg: StorageConfig | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self._creds = creds
        self._cfg = cfg if cfg is not None else StorageConfig()
        self._clock = clock
        self._timestamps: dict[tuple[str, str], float] = {}
        (self.root / creds.bucket_name).mkdir(parents=True, exist_ok=True)

    # -- lifecycle parity with StorageClient ---------------------------------
    async def __aenter__(self) -> MemoryStorage:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:  # noqa: B027 — nothing to release, parity only
        pass

    # -- helpers -------------------------------------------------------------
    def _path(self, bucket: BucketCreds, key: str) -> Path:
        if key.startswith("/") or any(part in ("", ".", "..") for part in key.split("/")):
            raise StorageError(f"malformed key {key!r}")
        return self.root / bucket.bucket_name / key

    def _require(self, bucket: BucketCreds, key: str) -> Path:
        path = self._path(bucket, key)
        if not path.is_file():
            raise ObjectMissingError(f"{bucket.bucket_name}/{key}")
        return path

    # -- writes (own bucket) -------------------------------------------------
    async def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(self._creds, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        self._timestamps[(self._creds.bucket_name, key)] = self._clock()

    async def upload_file(self, key: str, path: str | os.PathLike[str]) -> None:
        await self.put_bytes(key, Path(path).read_bytes())

    # -- reads (any bucket) --------------------------------------------------
    async def get_bytes(
        self,
        bucket: BucketCreds,
        key: str,
        *,
        expected_hash: str | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        path = self._require(bucket, key)
        limit = max_bytes if max_bytes is not None else self._cfg.max_payload_bytes
        size = path.stat().st_size
        if size > limit:
            raise ObjectTooLargeError(f"{bucket.bucket_name}/{key}: {size} bytes > limit {limit}")
        data = path.read_bytes()
        if expected_hash is not None:
            actual = hash_bytes(data)
            if actual != expected_hash.lower():
                raise IntegrityError(
                    f"{bucket.bucket_name}/{key}: blake2b {actual} != expected {expected_hash.lower()}"
                )
        return data

    async def download_file(
        self,
        bucket: BucketCreds,
        key: str,
        path: str | os.PathLike[str],
        *,
        expected_hash: str | None = None,
        max_bytes: int | None = None,
    ) -> None:
        data = await self.get_bytes(bucket, key, expected_hash=expected_hash, max_bytes=max_bytes)
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dest)

    async def object_exists(self, bucket: BucketCreds, key: str) -> bool:
        return self._path(bucket, key).is_file()

    async def object_timestamp(self, bucket: BucketCreds, key: str) -> float:
        path = self._require(bucket, key)
        recorded = self._timestamps.get((bucket.bucket_name, key))
        return recorded if recorded is not None else path.stat().st_mtime

    async def list_keys(self, bucket: BucketCreds, prefix: str) -> list[str]:
        base = self.root / bucket.bucket_name
        if not base.is_dir():
            return []
        keys = [
            p.relative_to(base).as_posix()
            for p in base.rglob("*")
            if p.is_file() and not p.name.endswith((".tmp", ".part"))
        ]
        return sorted(k for k in keys if k.startswith(prefix))

    async def gather_bytes(
        self,
        peers: Mapping[int, BucketCreds],
        key_fn: Callable[[int], str],
        *,
        expected_hashes: Mapping[int, str],
        deadline_s: float,
        max_bytes: int | None = None,
    ) -> GatherResult:
        """uid-ascending gather with StorageClient's failure-reason prefixes.

        ``deadline_s`` is accepted for signature parity; local filesystem reads
        cannot time out.
        """
        del deadline_s
        ok: OrderedDict[int, bytes] = OrderedDict()
        failed: dict[int, str] = {}
        for uid in sorted(peers):
            try:
                ok[uid] = await self.get_bytes(
                    peers[uid],
                    key_fn(uid),
                    expected_hash=expected_hashes.get(uid),
                    max_bytes=max_bytes,
                )
            except ObjectMissingError as e:
                failed[uid] = f"missing: {e}"
            except IntegrityError as e:
                failed[uid] = f"integrity: {e}"
            except ObjectTooLargeError as e:
                failed[uid] = f"too_large: {e}"
            except StorageError as e:
                failed[uid] = f"error: {type(e).__name__}: {e}"
        return GatherResult(ok=ok, failed=failed)


# --------------------------------------------------------------------------- #
# ScriptedChain + LoopbackClock
# --------------------------------------------------------------------------- #


class ScriptedChain:
    """Recording chain double for loopback runs (the runner's chain surface).

    ``commit_window`` records; ``get_window_commits`` replays what was
    committed (keyed by ``uid``); ``block_hash`` is a deterministic blake2b of
    the block number so attestation/audit scheduling stays reproducible.
    """

    def __init__(self, uid: int = 0) -> None:
        self.uid = uid
        self.commits: list[WindowCommit] = []

    def commit_window(self, commit: WindowCommit) -> None:
        self.commits.append(commit)

    def get_window_commits(
        self, window: int, uids: Any | None = None
    ) -> dict[int, WindowCommit]:
        del uids
        out: dict[int, WindowCommit] = {}
        for commit in self.commits:
            if commit.window == window:
                out[self.uid] = commit  # single-uid chain: the latest commit wins
        return out

    def commit_manifest_hash(self, manifest_hash: str) -> None:  # pragma: no cover — parity hook
        del manifest_hash

    def block_hash(self, block: int) -> bytes:
        return hashlib.blake2b(
            int(block).to_bytes(8, "little") + b"loopback", digest_size=32
        ).digest()

    def current_block(self) -> int:
        return 0

    def my_uid(self) -> int:
        return self.uid

    def sign(self, data: bytes) -> bytes:
        del data
        return b""


class LoopbackClock:
    """Stepped window clock: ``boundary_ts(w) = w * seconds_per_window``;
    ``enter_gate(w)`` moves "now" just inside window ``w``'s upload gate."""

    def __init__(self, seconds_per_window: float = 1000.0, *, gate_offset_s: float = 10.0) -> None:
        if gate_offset_s < 0:
            raise ValueError("gate_offset_s must be >= 0")
        self.seconds_per_window = float(seconds_per_window)
        self.gate_offset_s = float(gate_offset_s)
        self.now_ts = 0.0

    def boundary_ts(self, window: int) -> float:
        return window * self.seconds_per_window

    def now(self) -> float:
        return self.now_ts

    def enter_gate(self, window: int) -> None:
        self.now_ts = self.boundary_ts(window + 1) + self.gate_offset_s


# --------------------------------------------------------------------------- #
# Builders shared by harness users
# --------------------------------------------------------------------------- #


def make_compressor(model: MoKTransformer, cfg: RunConfig) -> TopKCompressor:
    """The run's compressor over every non-dense master shape."""
    shapes = {n: s for n, s in model.param_shapes().items() if not n.endswith(DENSE_SUFFIX)}
    return TopKCompressor(
        ChunkingTransformer(shapes, target_chunk=cfg.compression.target_chunk),
        Quantizer(bins=cfg.compression.quant_bins, range_sigmas=cfg.compression.quant_range_sigmas),
        topk=cfg.compression.topk,
    )


def make_outer_step(model: MoKTransformer, cfg: RunConfig) -> ReplicatedOuterStep:
    return ReplicatedOuterStep(cfg.outer, {n: torch.Size(s) for n, s in model.param_shapes().items()})


def local_manifest(
    cfg: RunConfig,
    index: DatasetShardIndex,
    *,
    shard_path: Callable[[int], Path],
    run_seed: bytes,
    run_id: str = "local-calibration",
) -> RunManifest:
    """A minimal manifest over a LOCAL shard tree (calibration/rehearsal runs).

    ``shard_bytes``/``tokens_total`` are derived from the actual files; the
    consensus fields that only matter on-chain get fixed local placeholders.
    """
    if index.num_shards == 0:
        raise ValueError("index has no shards")
    shard_bytes = shard_path(0).stat().st_size
    ref = DatasetManifestRef(
        name=index.name,
        merkle_root=index.merkle().root.hex(),
        num_shards=index.num_shards,
        shard_bytes=shard_bytes,
        seq_len=index.seq_len,
        tokens_total=index.num_shards * (shard_bytes // 2),
        tokenizer_hash="00" * 32,
    )
    return RunManifest(
        spec_version=1,
        run_id=run_id,
        netuid=cfg.chain.netuid,
        network=cfg.chain.network,
        config_hash=config_hash(cfg),
        container_digest="local",
        mok_commit="local",
        tk_commit="local",
        attention_backend="cudnn_det",
        start_block=0,
        blocks_per_window=cfg.window.blocks_per_window,
        prf=PRFSpec(run_seed_hex=run_seed.hex()),
        datasets=(ref,),
        init_checkpoint_hash="00" * 32,
    )


# --------------------------------------------------------------------------- #
# The harness
# --------------------------------------------------------------------------- #


class LocalLoopbackHarness:
    """A complete single-node WindowRunner rig: self as the only peer and the
    leader, MemoryStorage for both roles, ScriptedChain, stepped clock.

    ``run_window`` enters the window's upload gate before delegating, so a
    plain loop over windows is a full dress rehearsal of the live protocol
    (two-phase commit, certificate, certified gather, outer step, sync check,
    periodic checkpoint).
    """

    def __init__(
        self,
        model: MoKTransformer,
        cfg: RunConfig,
        manifest: RunManifest,
        index: DatasetShardIndex,
        *,
        shard_path: Callable[[int], Path],
        work_dir: str | os.PathLike[str],
        uid: int = 0,
        device: str | torch.device = "cpu",
        clock: LoopbackClock | None = None,
        metrics: Any | None = None,
        checkpoint: bool = True,
        cert_timeout_s: float = 10.0,
    ) -> None:
        work = Path(work_dir)
        self.model = model
        self.cfg = cfg
        self.manifest = manifest
        self.uid = uid
        self.creds = BucketCreds(
            account_id="local",
            bucket_name=f"loopback-uid{uid:05d}",
            access_key_id="local",
            secret_access_key="local",
        )
        self.clock = clock if clock is not None else LoopbackClock()
        if self.clock.gate_offset_s >= cfg.window.upload_grace_s:
            raise ValueError(
                f"clock gate_offset_s {self.clock.gate_offset_s} does not fit inside the "
                f"{cfg.window.upload_grace_s}s upload gate"
            )
        self.storage = MemoryStorage(
            work / "storage", self.creds, cfg=cfg.storage, clock=self.clock.now
        )
        self.chain = ScriptedChain(uid)
        self.checkpointer = Checkpointer(None, work / "ckpt") if checkpoint else None

        async def fetch_fn(shard_idx: int) -> bytes:
            return shard_path(shard_idx).read_bytes()

        self.runner = WindowRunner(
            model,
            cfg,
            manifest,
            uid=uid,
            rank=0,
            world_size=1,
            comm=SingleNodeComm(),
            storage=self.storage,
            chain=self.chain,
            shard_cache=ShardCache(work / "cache", 1 << 34, index),
            fetch_fn=fetch_fn,
            compressor=make_compressor(model, cfg),
            error_feedback=ErrorFeedback(beta=cfg.compression.ef_beta),
            outer_step=make_outer_step(model, cfg),
            checkpointer=self.checkpointer,
            metrics=metrics,
            clock=self.clock,
            peer_buckets=lambda _w: {uid: self.creds},
            leader_bucket=lambda _w: self.creds,
            device=device,
            self_leader=True,
            cert_poll_s=0.01,
            cert_timeout_s=cert_timeout_s,
        )

    async def run_window(self, window: int, state: RunState) -> WindowOutcome:
        self.clock.enter_gate(window)
        return await self.runner.run_window(window, state)
