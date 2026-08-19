"""Shared node bootstrap for the three role applications (miner/validator/auditor).

This module owns everything the three apps have in common:

  * `NodeContext` — the bundle of clients, caches and identity every app runs on.
  * `bootstrap(role, argv)` — argument parsing, determinism enforcement, config
    loading, manifest fetch + verification, client construction, uid resolution.
  * The local-harness stack (`ScriptedChain`, `MemoryStorage`, `LoopbackClock`,
    `_FallbackHarness`) used by `--local-harness` runs (step-B calibration, CPU
    tests) when `B.calibration.local_harness` has not been installed yet.
  * Replica materialization (`materialize_replica`): backend selection
    (mok on CUDA when importable, reference otherwise — logged loudly),
    checkpoint restore or seed-42 init, outer-optimizer state.
  * `catch_up_replica` — the thin app-side wrapper over `C.core.checkpoint.catch_up`.

Trust-model conventions established here (documented, consensus-adjacent):

  * OWNER_UID (0) publishes the run: `manifest.json` in its bucket, hash
    committed on-chain via `ManifestCommit`. Nodes verify the canonical hash.
  * Dataset shard indexes live in the owner bucket under
    `datasets/<name>/shard_index.json`; the index content is verified against
    the manifest's `DatasetManifestRef` (`verify_index_matches_ref`), so the
    key layout itself is not consensus-bearing.
  * Auditors identify themselves with the plain chain commitment string
    `AUDITOR_COMMITMENT` ("auditor.v1"). Auditors never publish WindowCommits,
    so the tag persists in their single commitment slot. Audit-report
    authenticity comes from the hotkey signature over the report's canonical
    unsigned fields; availability comes from validators polling auditor buckets.

Multi-rank note: torch.distributed wiring (RANK env → `TorchDistRunnerComm`)
is in place, but the CPU-verified paths run world_size == 1; EP-sharded
catch-up/state-root handling across ranks is a GPU-milestone item (see
docs/ENGINEERING_NOTES.md on globally-unique EP shard names).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from C.core.checkpoint import CatchUpReport, Checkpointer, catch_up
from C.core.compress import ChunkingTransformer, Quantizer, TopKCompressor
from C.core.outer_opt import ReplicatedOuterStep
from C.core.window_runner import (
    DENSE_SUFFIX,
    RunnerComm,
    SingleNodeComm,
    TorchDistRunnerComm,
    WindowClock,
)
from mok_core.chain.schemas import (
    Commitment,
    ManifestCommit,
    VoteCommit,
    WindowCommit,
    decode_commitment,
)
from mok_core.chain.windows import boundary_block, window_of_block
from mok_core.config import RunConfig, config_hash, load_run_config
from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest
from mok_core.config.schemas import BucketCreds
from mok_core.data import DatasetShardIndex, ShardCache, verify_index_matches_ref
from mok_core.data.shards import shard_filename
from mok_core.determinism import (
    assert_container_digest,
    enforce_determinism,
    hash_bytes,
    hash_named_tensors,
)
from mok_core.model import MoKTransformer, build_reference_model, init_model
from mok_core.storage import (
    GatherResult,
    IntegrityError,
    ObjectMissingError,
    ObjectTooLargeError,
    StorageClient,
    keys,
)
from mok_core.telemetry import Metrics, bind, get_logger, setup_logging

__all__ = [
    "AUDITOR_COMMITMENT",
    "INIT_SEED",
    "OWNER_UID",
    "BootstrapError",
    "ChainWindowClock",
    "LocalHarness",
    "LocalSigner",
    "LoopbackClock",
    "MemoryStorage",
    "NodeContext",
    "ScriptedChain",
    "Signer",
    "auditor_uids_from_chain",
    "bootstrap",
    "build_arg_parser",
    "build_compressor",
    "build_node_model",
    "build_outer_step",
    "catch_up_replica",
    "choose_backend",
    "DATASET_BUCKET_KEY",
    "DATASET_BUCKET_UID",
    "dataset_index_key",
    "dataset_shard_key",
    "load_master_state",
    "load_static_buckets",
    "materialize_replica",
    "resolve_leader_uid",
    "storage_fetch_fn",
]

log = get_logger("app.bootstrap")

#: The subnet owner's uid: publishes manifest.json + shard indexes + init checkpoint.
OWNER_UID = 0

#: Seed of the published initialization (state_root pinned in the manifest).
INIT_SEED = 42

#: Chain-commitment tag identifying an auditor node (see module docstring).
AUDITOR_COMMITMENT = "auditor.v1"

FetchFn = Callable[[int], Awaitable[bytes]]


class BootstrapError(RuntimeError):
    """Bootstrap cannot produce a coherent NodeContext."""


# --------------------------------------------------------------------------- #
# Signing
# --------------------------------------------------------------------------- #


@runtime_checkable
class Signer(Protocol):
    """The signing identity of this node (sr25519 hotkey in production)."""

    hotkey: str

    def sign(self, data: bytes) -> bytes: ...  # pragma: no cover — protocol

    def verify(self, hotkey: str, data: bytes, signature: bytes) -> bool: ...  # pragma: no cover


_LOCAL_SIGN_DOMAIN = b"mok.local-sign.v1"


def _local_signature(hotkey: str, data: bytes) -> bytes:
    h = hashlib.blake2b(digest_size=32)
    h.update(_LOCAL_SIGN_DOMAIN)
    h.update(hotkey.encode("utf-8"))
    h.update(data)
    return h.digest()


@dataclass(frozen=True)
class LocalSigner:
    """Deterministic keyless signer for the local harness: any party can verify
    any other by recomputation. NOT cryptographic — local/loopback runs only."""

    hotkey: str

    def sign(self, data: bytes) -> bytes:
        return _local_signature(self.hotkey, data)

    def verify(self, hotkey: str, data: bytes, signature: bytes) -> bool:
        return _local_signature(hotkey, data) == signature


@dataclass(frozen=True)
class ChainSigner:
    """Signer backed by a real `ChainClient` wallet hotkey."""

    chain: Any
    hotkey: str

    def sign(self, data: bytes) -> bytes:
        return bytes(self.chain.sign(data))

    def verify(self, hotkey: str, data: bytes, signature: bytes) -> bool:
        return bool(self.chain.verify(hotkey, data, signature))


# --------------------------------------------------------------------------- #
# Clocks
# --------------------------------------------------------------------------- #


class ChainWindowClock:
    """`WindowClock` over real chain time: boundary timestamps come from the
    boundary block's on-chain `Timestamp.Now` (cached forever once seen);
    future boundaries are extrapolated from the newest block at `block_time_s`
    per block (never cached — they firm up when the block exists)."""

    def __init__(self, chain: Any, manifest: RunManifest, *, block_time_s: float = 12.0) -> None:
        self.chain = chain
        self.manifest = manifest
        self.block_time_s = float(block_time_s)
        self._cache: dict[int, float] = {}

    def boundary_ts(self, window: int) -> float:
        cached = self._cache.get(window)
        if cached is not None:
            return cached
        block = boundary_block(window, self.manifest.start_block, self.manifest.blocks_per_window)
        try:
            current = int(self.chain.current_block())
            if block <= current:
                ts = float(self.chain.block_timestamp(block))
                self._cache[window] = ts
                return ts
            anchor_ts = float(self.chain.block_timestamp(current))
            return anchor_ts + (block - current) * self.block_time_s
        except Exception as e:  # noqa: BLE001 — chain hiccups must not crash gate math
            log.warning("boundary timestamp fallback to wall clock", window=window, error=str(e))
            return time.time()

    def now(self) -> float:
        return time.time()


class LoopbackClock:
    """Settable fake `WindowClock`: `boundary_ts(w) = genesis + w * window_s`."""

    def __init__(self, *, genesis: float = 0.0, window_s: float = 1000.0, now_ts: float = 0.0) -> None:
        self.genesis = float(genesis)
        self.window_s = float(window_s)
        self.now_ts = float(now_ts)

    def boundary_ts(self, window: int) -> float:
        return self.genesis + self.window_s * window

    def now(self) -> float:
        return self.now_ts

    def set(self, now_ts: float) -> None:
        self.now_ts = float(now_ts)


# --------------------------------------------------------------------------- #
# Local harness: in-memory storage
# --------------------------------------------------------------------------- #


class MemoryStorage:
    """In-memory drop-in for `mok_core.storage.StorageClient` (fallback harness).

    One shared `store` dict (bucket_name, key) -> (bytes, epoch_ts) plays the
    role of the object store; each instance is bound to its own bucket like the
    real client. Implements exactly the surface the apps/engine use.
    """

    def __init__(
        self,
        creds: BucketCreds,
        *,
        store: dict[tuple[str, str], tuple[bytes, float]] | None = None,
        max_payload_bytes: int = 4 * 1024**3,
    ) -> None:
        self.creds = creds
        self.store = store if store is not None else {}
        self.max_payload_bytes = int(max_payload_bytes)

    # -- lifecycle ----------------------------------------------------- #

    async def __aenter__(self) -> MemoryStorage:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def aclose(self) -> None:
        return None

    # -- writes (own bucket) ------------------------------------------- #

    async def put_bytes(self, key: str, data: bytes) -> None:
        self.store[(self.creds.bucket_name, key)] = (bytes(data), time.time())

    async def upload_file(self, key: str, path: str | os.PathLike[str]) -> None:
        await self.put_bytes(key, Path(path).read_bytes())

    # -- reads (any bucket) -------------------------------------------- #

    def _get(self, bucket: BucketCreds, key: str) -> tuple[bytes, float]:
        entry = self.store.get((bucket.bucket_name, key))
        if entry is None:
            raise ObjectMissingError(f"{bucket.bucket_name}/{key}: no such object")
        return entry

    async def get_bytes(
        self,
        bucket: BucketCreds,
        key: str,
        *,
        expected_hash: str | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        data, _ts = self._get(bucket, key)
        limit = max_bytes if max_bytes is not None else self.max_payload_bytes
        if len(data) > limit:
            raise ObjectTooLargeError(f"{key}: {len(data)} bytes > limit {limit}")
        if expected_hash is not None and hash_bytes(data) != expected_hash.lower():
            raise IntegrityError(f"{key}: content hash mismatch")
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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    async def object_timestamp(self, bucket: BucketCreds, key: str) -> float:
        return self._get(bucket, key)[1]

    async def object_exists(self, bucket: BucketCreds, key: str) -> bool:
        return (bucket.bucket_name, key) in self.store

    async def list_keys(self, bucket: BucketCreds, prefix: str) -> list[str]:
        return sorted(
            k for (b, k) in self.store if b == bucket.bucket_name and k.startswith(prefix)
        )

    async def gather_bytes(
        self,
        peers: Mapping[int, BucketCreds],
        key_fn: Callable[[int], str],
        *,
        expected_hashes: Mapping[int, str],
        deadline_s: float,
        max_bytes: int | None = None,
    ) -> GatherResult:
        del deadline_s  # in-memory fetches cannot time out
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
        return GatherResult(ok=ok, failed=failed)


# --------------------------------------------------------------------------- #
# Local harness: scripted chain
# --------------------------------------------------------------------------- #


class ScriptedChain:
    """In-memory stand-in for `mok_core.chain.ChainClient` (fallback harness + tests).

    Block time is derived from the injected clock; commitments follow the real
    single-slot-per-uid model, but window commits and votes are ALSO retained
    in per-window history dicts so validators/tests can read past windows
    without racing the slot (a convenience the real chain provides implicitly
    because validators read during the gate)."""

    def __init__(
        self,
        *,
        clock: WindowClock,
        start_block: int,
        blocks_per_window: int,
        block_time_s: float | None = None,
        my_uid: int | None = 0,
        buckets: dict[int, BucketCreds] | None = None,
        stakes: dict[int, float] | None = None,
        manifest_hashes: dict[int, str] | None = None,
    ) -> None:
        self.clock = clock
        self.start_block = int(start_block)
        self.blocks_per_window = int(blocks_per_window)
        window_s = clock.boundary_ts(1) - clock.boundary_ts(0)
        self.block_time_s = (
            float(block_time_s) if block_time_s is not None else window_s / blocks_per_window
        )
        self._my_uid = my_uid
        self.buckets: dict[int, BucketCreds] = dict(buckets or {})
        self._stakes: dict[int, float] = dict(stakes or {})
        self.manifest_hashes: dict[int, str] = dict(manifest_hashes or {})
        self.commitments: dict[int, str] = {}
        self.window_commits: dict[int, dict[int, WindowCommit]] = {}
        self.votes: dict[int, VoteCommit] = {}
        self.weights_calls: list[dict[int, float]] = []

    # -- identity ------------------------------------------------------ #

    def sync_metagraph(self) -> None:
        return None

    def uids(self) -> list[int]:
        known = set(self.buckets) | set(self._stakes) | set(self.commitments)
        if self._my_uid is not None:
            known.add(self._my_uid)
        return sorted(known)

    def hotkeys(self) -> list[str]:
        return [f"local-{uid}" for uid in self.uids()]

    def hotkey_of(self, uid: int) -> str | None:
        return f"local-{uid}" if uid in self.uids() else None

    def uid_of_hotkey(self, hotkey: str) -> int | None:
        if not hotkey.startswith("local-"):
            return None
        try:
            uid = int(hotkey.removeprefix("local-"))
        except ValueError:
            return None
        return uid if uid in self.uids() else None

    def my_uid(self) -> int | None:
        return self._my_uid

    def stakes(self) -> dict[int, float]:
        return dict(self._stakes)

    # -- blocks & time ------------------------------------------------- #

    def current_block(self, *, force: bool = False) -> int:
        del force
        elapsed = self.clock.now() - self.clock.boundary_ts(0)
        return self.start_block + max(0, int(elapsed / self.block_time_s))

    def block_hash(self, block: int) -> bytes:
        h = hashlib.blake2b(digest_size=32)
        h.update(b"scripted-block.v1")
        h.update(int(block).to_bytes(8, "little", signed=False))
        return h.digest()

    def block_timestamp(self, block: int) -> float:
        return self.clock.boundary_ts(0) + (block - self.start_block) * self.block_time_s

    def current_window(self, schedule: Any) -> int:
        return window_of_block(
            max(self.current_block(), schedule.start_block),
            schedule.start_block,
            schedule.blocks_per_window,
        )

    async def wait_for_window(self, window: int, schedule: Any, poll_s: float = 0.05) -> int:
        while True:
            current = self.current_window(schedule)
            if current >= window:
                return current
            await asyncio.sleep(poll_s)

    # -- commitments --------------------------------------------------- #

    def commit(self, data: str) -> None:
        if self._my_uid is None:
            raise RuntimeError("scripted chain has no uid to commit as")
        self.commitments[self._my_uid] = data
        try:
            decoded: Commitment = decode_commitment(data, hotkey_ss58=self.hotkey_of(self._my_uid))
        except ValueError:
            return
        if isinstance(decoded, WindowCommit):
            self.window_commits.setdefault(decoded.window, {})[self._my_uid] = decoded
        elif isinstance(decoded, VoteCommit):
            self.votes[self._my_uid] = decoded
        elif isinstance(decoded, ManifestCommit):
            self.manifest_hashes[self._my_uid] = decoded.manifest_hash

    def get_commitment(self, uid: int) -> str | None:
        return self.commitments.get(uid)

    def get_all_commitments(self, block: int | None = None) -> dict[int, str]:
        del block
        return dict(self.commitments)

    def commit_window(self, commit: WindowCommit) -> None:
        self.commit(commit.encode())

    def get_window_commits(
        self, window: int, uids: Any | None = None
    ) -> dict[int, WindowCommit]:
        commits = dict(self.window_commits.get(window, {}))
        if uids is not None:
            wanted = set(uids)
            commits = {u: c for u, c in commits.items() if u in wanted}
        return commits

    def commit_manifest_hash(self, manifest_hash: str) -> None:
        self.commit(ManifestCommit(manifest_hash=manifest_hash).encode())

    def get_manifest_hash(self, owner_uid: int) -> str | None:
        return self.manifest_hashes.get(owner_uid)

    def commit_vote(self, vote: VoteCommit) -> None:
        self.commit(vote.encode())

    def get_votes(
        self,
        kind: str | None = None,
        target: int | None = None,
        uids: Any | None = None,
    ) -> dict[int, VoteCommit]:
        wanted = None if uids is None else set(uids)
        return {
            uid: v
            for uid, v in self.votes.items()
            if (kind is None or v.kind == kind)
            and (target is None or v.target == target)
            and (wanted is None or uid in wanted)
        }

    # -- buckets & weights --------------------------------------------- #

    def commit_bucket(self, creds: BucketCreds) -> None:
        if self._my_uid is None:
            raise RuntimeError("scripted chain has no uid to commit as")
        self.buckets[self._my_uid] = creds

    def get_bucket(self, uid: int) -> BucketCreds | None:
        return self.buckets.get(uid)

    def get_all_buckets(self) -> dict[int, BucketCreds]:
        return dict(self.buckets)

    def ensure_bucket_committed(self, creds: BucketCreds) -> bool:
        if self.buckets.get(self._my_uid) == creds:
            return False
        self.commit_bucket(creds)
        return True

    def set_weights(self, weights: dict[int, float], *, wait_for_inclusion: bool = False) -> bool:
        del wait_for_inclusion
        self.weights_calls.append(dict(weights))
        return True

    # -- signing ------------------------------------------------------- #

    def sign(self, data: bytes) -> bytes:
        if self._my_uid is None:
            raise RuntimeError("scripted chain has no uid to sign as")
        return _local_signature(f"local-{self._my_uid}", data)

    def verify(self, hotkey_ss58: str, data: bytes, signature: bytes) -> bool:
        return _local_signature(hotkey_ss58, data) == signature


# --------------------------------------------------------------------------- #
# Local harness: composition
# --------------------------------------------------------------------------- #


@runtime_checkable
class LocalHarness(Protocol):
    """What `--local-harness` must supply. `B.calibration.local_harness.
    LocalLoopbackHarness` (when present) and `_FallbackHarness` both satisfy it."""

    cfg: RunConfig
    manifest: RunManifest
    creds: BucketCreds
    chain: Any
    storage: Any
    clock: WindowClock
    data_dir: Path

    def bucket_for(self, uid: int) -> BucketCreds: ...  # pragma: no cover — protocol


def dataset_index_key(name: str) -> str:
    """Owner-bucket key of a dataset's `DatasetShardIndex` JSON."""
    if not name or "/" in name:
        raise ValueError(f"invalid dataset name {name!r}")
    return f"datasets/{name}/shard_index.json"


def dataset_shard_key(name: str, leaf_hex: str) -> str:
    """Owner-bucket key of one shard — the layout `mok-data upload` publishes
    (`datasets/<name>/shard-<hash16>.bin`, mirroring step A's local filenames).
    Content is hash-verified against the manifest, so the key layout is not
    consensus-bearing."""
    if not name or "/" in name:
        raise ValueError(f"invalid dataset name {name!r}")
    return f"datasets/{name}/{shard_filename(bytes.fromhex(leaf_hex))}"


@dataclass
class _FallbackHarness:
    """Minimal in-memory `LocalHarness`: synthetic verified dataset, scripted
    chain, memory object store, loopback clock. Used when step B's
    `local_harness` module is not installed."""

    cfg: RunConfig
    manifest: RunManifest
    creds: BucketCreds
    chain: ScriptedChain
    storage: MemoryStorage
    clock: LoopbackClock
    data_dir: Path
    store: dict[tuple[str, str], tuple[bytes, float]] = field(default_factory=dict)

    def bucket_for(self, uid: int) -> BucketCreds:
        return BucketCreds(
            account_id="local",
            bucket_name=f"mok-local-{uid:05d}",
            access_key_id="local",
            secret_access_key="local",
        )

    @classmethod
    def create(
        cls,
        cfg: RunConfig,
        *,
        root: Path,
        uid: int = OWNER_UID,
        num_shards: int = 8,
        seqs_per_shard: int = 8,
        run_seed: bytes = bytes(range(32)),
        now_ts: float | None = None,
    ) -> _FallbackHarness:
        seq_len = cfg.model.seq_len
        vocab = cfg.model.vocab_size
        data_dir = root / "dataset"
        data_dir.mkdir(parents=True, exist_ok=True)
        cycle = (3, 5, 7, 11)
        hashes: list[str] = []
        shard_paths: list[Path] = []
        for s in range(num_shards):
            rows = []
            for r in range(seqs_per_shard):
                if r == seqs_per_shard - 1:
                    rows.append(np.full(seq_len, (100 + s) % vocab, dtype="<u2"))
                else:
                    phase = (s + r) % len(cycle)
                    rows.append(
                        np.array(
                            [cycle[(phase + j) % len(cycle)] % vocab for j in range(seq_len)],
                            dtype="<u2",
                        )
                    )
            blob = np.stack(rows).tobytes()
            digest = hashlib.blake2b(blob, digest_size=32).hexdigest()
            path = data_dir / f"shard-{digest[:16]}.bin"
            path.write_bytes(blob)
            hashes.append(digest)
            shard_paths.append(path)
        index = DatasetShardIndex(name="bulk", seq_len=seq_len, shard_hashes=hashes)
        ref = DatasetManifestRef(
            name="bulk",
            merkle_root=index.merkle().root.hex(),
            num_shards=num_shards,
            shard_bytes=2 * seq_len * seqs_per_shard,
            seq_len=seq_len,
            tokens_total=num_shards * seqs_per_shard * seq_len,
            tokenizer_hash="00" * 32,
        )
        manifest = RunManifest(
            spec_version=1,
            run_id="local-harness",
            netuid=cfg.chain.netuid,
            network=cfg.chain.network,
            config_hash=config_hash(cfg),
            container_digest="sha256:" + "00" * 32,
            mok_commit="local",
            tk_commit="local",
            attention_backend="cudnn_det",
            start_block=100,
            blocks_per_window=cfg.window.blocks_per_window,
            prf=PRFSpec(run_seed_hex=run_seed.hex()),
            datasets=(ref,),
            init_checkpoint_hash="00" * 32,  # local runs skip the init-root gate
        )
        genesis = time.time() if now_ts is None else now_ts
        clock = LoopbackClock(
            genesis=genesis,
            window_s=cfg.window.blocks_per_window * cfg.chain.block_time_s,
            now_ts=genesis + 1.0,
        )
        store: dict[tuple[str, str], tuple[bytes, float]] = {}
        harness = cls(
            cfg=cfg,
            manifest=manifest,
            creds=BucketCreds(
                account_id="local",
                bucket_name=f"mok-local-{uid:05d}",
                access_key_id="local",
                secret_access_key="local",
            ),
            chain=ScriptedChain(
                clock=clock,
                start_block=manifest.start_block,
                blocks_per_window=manifest.blocks_per_window,
                block_time_s=cfg.chain.block_time_s,
                my_uid=uid,
                stakes={uid: 1.0},
            ),
            storage=MemoryStorage(
                BucketCreds(
                    account_id="local",
                    bucket_name=f"mok-local-{uid:05d}",
                    access_key_id="local",
                    secret_access_key="local",
                ),
                store=store,
            ),
            clock=clock,
            data_dir=data_dir,
            store=store,
        )
        harness.chain.buckets[uid] = harness.creds
        owner_bucket = harness.bucket_for(OWNER_UID)
        harness.chain.buckets.setdefault(OWNER_UID, owner_bucket)
        harness.chain.manifest_hashes[OWNER_UID] = manifest.manifest_hash()
        # Seed the "owner bucket" objects: manifest, shard index, shard bytes.
        # The manifest bytes MUST be canonical: fetchers verify them against the
        # on-chain manifest hash (canonical_bytes -> blake2b).
        from mok_core.config.canonical import canonical_bytes  # noqa: PLC0415

        now = time.time()
        harness.store[(owner_bucket.bucket_name, keys.MANIFEST_KEY)] = (
            canonical_bytes(manifest),
            now,
        )
        harness.store[(owner_bucket.bucket_name, dataset_index_key("bulk"))] = (
            canonical_bytes(index),
            now,
        )
        for digest, path in zip(hashes, shard_paths, strict=True):
            harness.store[(owner_bucket.bucket_name, dataset_shard_key("bulk", digest))] = (
                path.read_bytes(),
                now,
            )
        return harness


def _load_local_harness(cfg: RunConfig, *, root: Path, uid: int) -> LocalHarness:
    """Prefer step B's harness when it exposes the bootstrap contract.

    B's `LocalLoopbackHarness` may exist with a different (runner-rig) shape;
    only a `create(cfg, *, root, uid)` classmethod producing a `LocalHarness`
    is usable here — anything else falls back to the in-memory harness.
    """
    try:
        from B.calibration.local_harness import LocalLoopbackHarness  # noqa: PLC0415
    except ImportError:
        log.info("B.calibration.local_harness not installed — using in-memory fallback harness")
        return _FallbackHarness.create(cfg, root=root, uid=uid)
    create = getattr(LocalLoopbackHarness, "create", None)
    if create is None:
        log.info("B harness lacks the bootstrap contract — using in-memory fallback harness")
        return _FallbackHarness.create(cfg, root=root, uid=uid)
    return create(cfg, root=root, uid=uid)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# NodeContext
# --------------------------------------------------------------------------- #


@dataclass
class NodeContext:
    """Everything a role application needs, fully constructed and verified."""

    role: str
    cfg: RunConfig
    manifest: RunManifest
    uid: int
    signer: Signer
    chain: Any
    storage: Any                              # StorageClient or MemoryStorage
    own_bucket: BucketCreds
    shard_caches: dict[str, ShardCache]
    shard_indexes: dict[str, DatasetShardIndex]
    fetch_fns: dict[str, FetchFn]
    metrics: Any
    comm: RunnerComm
    clock: WindowClock
    rank: int
    world_size: int
    protocol_world_size: int                  # the miner-node rank count (token accounting)
    device: str
    state_dir: Path
    local: bool = False
    dev_insecure: bool = False
    static_buckets: dict[int, BucketCreds] = field(default_factory=dict)

    @property
    def run_seed(self) -> bytes:
        return bytes.fromhex(self.manifest.prf.run_seed_hex)

    def owner_bucket(self) -> BucketCreds:
        bucket = self.static_buckets.get(OWNER_UID) or self.chain.get_bucket(OWNER_UID)
        return bucket if bucket is not None else self.own_bucket

    def dataset_bucket(self) -> BucketCreds:
        """Where shard trees are read from: the static map's ``"dataset"`` entry
        when present, else the owner bucket (the default single-bucket layout)."""
        return self.static_buckets.get(DATASET_BUCKET_UID) or self.owner_bucket()

    def peer_buckets(self) -> dict[int, BucketCreds]:
        # get_all_buckets is history-aware (onboarding BucketCommits survive later
        # per-window commits); MOK_STATIC_BUCKETS entries override per uid.
        buckets = dict(self.chain.get_all_buckets())
        buckets.update({uid: b for uid, b in self.static_buckets.items() if uid >= 0})
        buckets.setdefault(self.uid, self.own_bucket)
        return buckets

    def leader_uid(self) -> int:
        return resolve_leader_uid(self.chain, fallback=self.uid)

    def leader_bucket(self) -> BucketCreds:
        leader = self.leader_uid()
        bucket = self.static_buckets.get(leader) or self.chain.get_bucket(leader)
        return bucket if bucket is not None else self.own_bucket

    async def aclose(self) -> None:
        import contextlib  # noqa: PLC0415 — teardown-only helper

        with contextlib.suppress(Exception):  # telemetry teardown is best-effort
            self.metrics.close()
        aclose = getattr(self.storage, "aclose", None)
        if aclose is not None:
            await aclose()


def resolve_leader_uid(chain: Any, *, fallback: int) -> int:
    """The leader validator: highest stake, ties to the LOWEST uid (total order)."""
    stakes: dict[int, float] = dict(chain.stakes())
    positive = {uid: s for uid, s in stakes.items() if s > 0.0}
    if not positive:
        return fallback
    return min(positive, key=lambda uid: (-positive[uid], uid))


def auditor_uids_from_chain(chain: Any) -> list[int]:
    """Uids whose latest chain commitment is the auditor tag (see module docstring)."""
    return sorted(
        uid for uid, wire in chain.get_all_commitments().items() if wire == AUDITOR_COMMITMENT
    )


def storage_fetch_fn(storage: Any, bucket: BucketCreds, index: DatasetShardIndex) -> FetchFn:
    """Shard downloader over content-addressed storage keys, hash-verified."""

    async def fetch(shard_idx: int) -> bytes:
        leaf = index.shard_hashes[shard_idx]
        return await storage.get_bytes(
            bucket, dataset_shard_key(index.name, leaf), expected_hash=leaf
        )

    return fetch


# --------------------------------------------------------------------------- #
# Model / engine builders
# --------------------------------------------------------------------------- #


def choose_backend(device: str | torch.device) -> str:
    """'mok' on CUDA when the kernel wheel is importable, else 'reference' — loudly."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if importlib.util.find_spec("mok") is not None:
            log.info("backend selected: mok (SM103 megakernel)", device=str(dev))
            return "mok"
        log.warning(
            "backend FALLBACK: CUDA device but the 'mok' wheel is not importable — "
            "running the pure-PyTorch reference backend (SLOW, not for mining rewards)",
            device=str(dev),
        )
        return "reference"
    log.info("backend selected: reference (CPU device)", device=str(dev))
    return "reference"


def build_node_model(cfg: RunConfig, device: str | torch.device) -> MoKTransformer:
    """The seed-42 initialized replica on the chosen backend."""
    backend = choose_backend(device)
    if backend == "mok":
        return init_model(cfg.model, INIT_SEED, device=device, backend="mok", mok_runtime=cfg.mok)
    return build_reference_model(cfg.model, INIT_SEED, device=device)


def build_compressor(model: MoKTransformer, cfg: RunConfig) -> TopKCompressor:
    """SparseLoCo compressor over every non-dense master parameter."""
    shapes = {n: s for n, s in model.param_shapes().items() if not n.endswith(DENSE_SUFFIX)}
    return TopKCompressor(
        ChunkingTransformer(shapes, target_chunk=cfg.compression.target_chunk),
        Quantizer(bins=cfg.compression.quant_bins, range_sigmas=cfg.compression.quant_range_sigmas),
        topk=cfg.compression.topk,
    )


def build_outer_step(model: MoKTransformer, cfg: RunConfig) -> ReplicatedOuterStep:
    return ReplicatedOuterStep(
        cfg.outer, {n: torch.Size(s) for n, s in model.param_shapes().items()}
    )


def load_master_state(model: MoKTransformer, state: Mapping[str, torch.Tensor]) -> None:
    """Copy a checkpointed master state dict into the model, bitwise, strict names."""
    master = dict(model.iter_master_params())
    missing = sorted(set(master) - set(state))
    extra = sorted(set(state) - set(master))
    if missing or extra:
        raise BootstrapError(f"checkpoint state mismatch: missing={missing[:4]} extra={extra[:4]}")
    with torch.no_grad():
        for name, param in master.items():
            t = state[name]
            if tuple(t.shape) != tuple(param.shape) or t.dtype != param.dtype:
                raise BootstrapError(
                    f"checkpoint tensor {name!r}: {tuple(t.shape)}/{t.dtype} != "
                    f"{tuple(param.shape)}/{param.dtype}"
                )
            param.data.copy_(t)


async def materialize_replica(
    ctx: NodeContext, checkpointer: Checkpointer | None
) -> tuple[MoKTransformer, ReplicatedOuterStep, int]:
    """Model + outer optimizer at the newest known θ; returns `from_window`.

    Order of preference: the newest regular checkpoint (`Checkpointer.
    load_latest` — local, then the leader's bucket), else the owner-published
    init via `B.onboarding.fetch_and_verify_init` (lazy import; bitwise-checked
    against `manifest.init_checkpoint_hash`), else the fresh seed-42 init whose
    state_root must equal `manifest.init_checkpoint_hash` (advisory in
    local/dev-insecure runs). `from_window` follows the checkpoint convention:
    the returned state is θ_start(from_window + 1); -1 means θ_init. B's init
    checkpoint is stored AS window 0 but holds θ_start(0) — any restored state
    whose root equals the manifest init hash therefore maps to -1.
    """
    model = build_node_model(ctx.cfg, ctx.device)
    outer = build_outer_step(model, ctx.cfg)
    init_root = ctx.manifest.init_checkpoint_hash

    if checkpointer is not None:
        loaded = await checkpointer.load_latest(bucket=ctx.leader_bucket())
        if loaded is not None:
            state, outer_state, meta = loaded
            if meta.manifest_hash != ctx.manifest.manifest_hash():
                log.warning(
                    "checkpoint manifest hash differs from the governing manifest",
                    checkpoint=meta.manifest_hash,
                    manifest=ctx.manifest.manifest_hash(),
                )
            load_master_state(model, state)
            outer.load_state_dict(outer_state)
            from_window = -1 if meta.state_root == init_root else meta.window
            log.info("replica restored from checkpoint", window=from_window, root=meta.state_root)
            return model, outer, from_window

    try:
        from B.onboarding import fetch_and_verify_init  # noqa: PLC0415 — optional step-B hook
    except ImportError:
        fetch_and_verify_init = None
    if fetch_and_verify_init is not None:
        try:
            state, outer_state, _meta = await fetch_and_verify_init(
                ctx.storage,
                None,  # the run-manifest chain slot is NOT the init-root slot here
                init_root,
                local_dir=ctx.state_dir / "init",
                bucket=ctx.owner_bucket(),
            )
            load_master_state(model, state)
            outer.load_state_dict(outer_state)
            log.info("replica restored from the published init", root=init_root)
            return model, outer, -1
        except Exception as e:  # noqa: BLE001 — init publication is optional; fall through
            log.info("fetch_and_verify_init unavailable; building the init locally", error=str(e))

    root = hash_named_tensors(model.iter_master_params())
    if root != ctx.manifest.init_checkpoint_hash:
        if ctx.local or ctx.dev_insecure:
            log.warning(
                "init state_root differs from manifest (allowed in local/dev-insecure runs)",
                init_root=root,
                manifest_root=ctx.manifest.init_checkpoint_hash,
            )
        else:
            raise BootstrapError(
                f"seed-{INIT_SEED} init root {root} != manifest init_checkpoint_hash "
                f"{ctx.manifest.init_checkpoint_hash} — wrong container/torch build?"
            )
    log.info("replica initialized fresh", seed=INIT_SEED, root=root)
    return model, outer, -1


async def catch_up_replica(
    ctx: NodeContext,
    model: MoKTransformer,
    outer_step: ReplicatedOuterStep,
    *,
    from_window: int,
    to_window: int,
) -> CatchUpReport:
    """Bitwise catch-up of windows (from_window, to_window] via the leader mirror."""
    report = await catch_up(
        dict(model.iter_master_params()),
        outer_step,
        None,
        ctx.storage,
        ctx.chain,
        ctx.manifest,
        ctx.cfg,
        from_window,
        to_window,
        leader_bucket=ctx.leader_bucket(),
    )
    log.info(
        "catch-up complete",
        applied=list(report.applied_windows),
        skipped_void=list(report.skipped_void),
        unverified=list(report.unverified_windows),
        final_root=report.final_root,
    )
    return report


# --------------------------------------------------------------------------- #
# bootstrap()
# --------------------------------------------------------------------------- #


def build_arg_parser(role: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=f"mok-{role}", description=f"MOK subnet {role} application")
    p.add_argument("--config", default="C/configs/base.yaml", help="base RunConfig YAML")
    p.add_argument(
        "--overlay", action="append", default=[], help="overlay YAML (repeatable, deep-merged)"
    )
    p.add_argument("--netuid", type=int, default=None, help="override cfg.chain.netuid")
    p.add_argument("--network", default=None, help="override cfg.chain.network")
    p.add_argument(
        "--local-harness",
        action="store_true",
        help="run against the in-process loopback harness (calibration/tests)",
    )
    p.add_argument("--uid", type=int, default=None, help="uid override (local harness only)")
    p.add_argument(
        "--dev-insecure",
        action="store_true",
        help="skip the container-digest assertion (development only)",
    )
    p.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    p.add_argument("--state-dir", default=None, help="root for checkpoints/state/telemetry")
    return p


def _apply_chain_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    updates: dict[str, Any] = {}
    if args.netuid is not None:
        updates["netuid"] = args.netuid
    if args.network is not None:
        updates["network"] = args.network
    if not updates:
        return cfg
    return cfg.model_copy(update={"chain": cfg.chain.model_copy(update=updates)})


def _init_comm() -> tuple[RunnerComm, int, int]:
    """torch.distributed comm when launched under torchrun, else single-process."""
    if os.environ.get("RANK") is None:
        return SingleNodeComm(), 0, 1
    import torch.distributed as dist  # noqa: PLC0415 — only under torchrun

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return TorchDistRunnerComm(), dist.get_rank(), dist.get_world_size()


def _resolve_device(args: argparse.Namespace, rank: int) -> str:
    if args.device is not None:
        return str(args.device)
    if torch.cuda.is_available():
        return f"cuda:{rank % max(1, torch.cuda.device_count())}"
    return "cpu"


async def _fetch_manifest(storage: Any, owner_bucket: BucketCreds, expected_hash: str) -> RunManifest:
    data = await storage.get_bytes(
        owner_bucket, keys.MANIFEST_KEY, expected_hash=expected_hash, max_bytes=1 << 20
    )
    manifest = RunManifest.model_validate_json(data)
    if manifest.manifest_hash() != expected_hash:
        raise BootstrapError(
            f"manifest canonical hash {manifest.manifest_hash()} != on-chain {expected_hash}"
        )
    return manifest


async def _build_shard_caches(
    cfg: RunConfig,
    manifest: RunManifest,
    storage: Any,
    owner_bucket: BucketCreds,
    *,
    cache_dir: Path,
) -> tuple[dict[str, ShardCache], dict[str, DatasetShardIndex], dict[str, FetchFn]]:
    caches: dict[str, ShardCache] = {}
    indexes: dict[str, DatasetShardIndex] = {}
    fetchers: dict[str, FetchFn] = {}
    for ref in manifest.datasets:
        raw = await storage.get_bytes(
            owner_bucket, dataset_index_key(ref.name), max_bytes=64 << 20
        )
        index = DatasetShardIndex.model_validate_json(raw)
        verify_index_matches_ref(index, ref)
        caches[ref.name] = ShardCache(cache_dir, cfg.data.shard_cache_max_bytes, index)
        indexes[ref.name] = index
        fetchers[ref.name] = storage_fetch_fn(storage, owner_bucket, index)
    return caches, indexes, fetchers


async def bootstrap(
    role: str,
    argv: list[str] | None = None,
    *,
    harness: LocalHarness | None = None,
) -> NodeContext:
    """Parse args, enforce determinism, verify the run, build the NodeContext.

    `harness` injects a pre-built local harness (tests); otherwise
    `--local-harness` loads `B.calibration.local_harness` or the in-memory
    fallback. The returned context owns its storage client — call
    `ctx.aclose()` when done.
    """
    if role not in ("miner", "validator", "auditor"):
        raise ValueError(f"unknown role {role!r}")
    args = build_arg_parser(role).parse_args(argv)

    enforce_determinism()  # FIRST — before any CUDA context exists

    cfg = _apply_chain_overrides(load_run_config(args.config, *args.overlay), args)
    setup_logging(cfg.telemetry.log_level)

    comm, rank, world_size = _init_comm()
    device = _resolve_device(args, rank)
    local = bool(args.local_harness or harness is not None)
    if args.uid is not None and not local:
        raise BootstrapError("--uid is only valid with --local-harness")

    state_dir = Path(
        args.state_dir if args.state_dir is not None else f"mok-state/{role}"
    ).expanduser()
    state_dir.mkdir(parents=True, exist_ok=True)
    static_buckets = load_static_buckets()

    if local:
        h = harness if harness is not None else _load_local_harness(cfg, root=state_dir, uid=args.uid or 0)
        chain: Any = h.chain
        storage: Any = h.storage
        clock: WindowClock = h.clock
        own_bucket = h.creds
        uid = args.uid if args.uid is not None else (chain.my_uid() or 0)
        manifest_hash = chain.get_manifest_hash(OWNER_UID)
        if manifest_hash is None:
            raise BootstrapError("local harness has no committed manifest hash")
        owner_bucket = chain.get_bucket(OWNER_UID) or own_bucket
        manifest = await _fetch_manifest(storage, owner_bucket, manifest_hash)
        signer: Signer = LocalSigner(hotkey=f"local-{uid}")
        protocol_world_size = max(1, world_size)
    else:
        from mok_core.chain import ChainClient  # noqa: PLC0415 — bittensor stays lazy

        chain = ChainClient(cfg.chain)
        uid_maybe = chain.my_uid()
        if uid_maybe is None:
            raise BootstrapError(
                f"hotkey {cfg.chain.wallet_hotkey!r} is not registered on netuid {cfg.chain.netuid}"
            )
        uid = int(uid_maybe)
        own_bucket = _bucket_from_env(chain.hotkey_of(uid) or "")
        storage = StorageClient(own_bucket, cfg.storage)
        await storage.__aenter__()
        manifest_hash = chain.get_manifest_hash(OWNER_UID)
        if manifest_hash is None:
            raise BootstrapError(f"no manifest committed by owner uid {OWNER_UID}")
        # ChainClient.get_bucket is history-aware: the owner committed its bucket
        # at onboarding BEFORE the ManifestCommit, so it is recoverable from
        # earlier blocks. MOK_STATIC_BUCKETS remains an optional local override.
        owner_bucket = static_buckets.get(OWNER_UID) or chain.get_bucket(OWNER_UID)
        if owner_bucket is None:
            raise BootstrapError(
                f"owner uid {OWNER_UID} has no BucketCommit in commitment history "
                f"(did the owner run mok-onboard before publishing?) and no "
                f"{STATIC_BUCKETS_ENV} override"
            )
        manifest = await _fetch_manifest(storage, owner_bucket, manifest_hash)
        signer = ChainSigner(chain=chain, hotkey=chain.hotkey_of(uid) or "")
        protocol_world_size = cfg.model.ep_size

    if config_hash(cfg) != manifest.config_hash:
        message = (
            f"RunConfig canonical hash {config_hash(cfg)} != manifest config_hash "
            f"{manifest.config_hash} — this node would desync"
        )
        if local or args.dev_insecure:
            log.warning(message)
        else:
            raise BootstrapError(message)
    if not (args.dev_insecure or local):
        assert_container_digest(manifest.container_digest)

    bind(role=role, uid=uid, rank=rank)
    metrics = Metrics(
        cfg.telemetry,
        run_name=f"{manifest.run_id}-{role}-uid{uid:05d}",
        out_dir=state_dir / "telemetry",
    )
    caches, indexes, fetchers = await _build_shard_caches(
        cfg,
        manifest,
        storage,
        static_buckets.get(DATASET_BUCKET_UID) or owner_bucket,
        cache_dir=Path(cfg.data.shard_cache_dir).expanduser()
        if not local
        else state_dir / "shard-cache",
    )
    clock = clock if local else ChainWindowClock(chain, manifest, block_time_s=cfg.chain.block_time_s)

    log.info(
        "bootstrap complete",
        role=role,
        uid=uid,
        network=cfg.chain.network,
        netuid=cfg.chain.netuid,
        world_size=world_size,
        device=device,
        run_id=manifest.run_id,
        manifest_hash=manifest.manifest_hash(),
        start_block=manifest.start_block,
        blocks_per_window=manifest.blocks_per_window,
        current_window=(0 if local else chain.current_window(manifest)),
        own_bucket=own_bucket.bucket_name,
        owner_bucket=owner_bucket.bucket_name,
        dataset_bucket=(static_buckets.get(DATASET_BUCKET_UID) or owner_bucket).bucket_name,
        datasets=[d.name for d in manifest.datasets],
        local=local,
        dev_insecure=bool(args.dev_insecure),
    )
    return NodeContext(
        role=role,
        cfg=cfg,
        manifest=manifest,
        uid=uid,
        signer=signer,
        chain=chain,
        storage=storage,
        own_bucket=own_bucket,
        shard_caches=caches,
        shard_indexes=indexes,
        fetch_fns=fetchers,
        metrics=metrics,
        comm=comm,
        clock=clock,
        rank=rank,
        world_size=world_size,
        protocol_world_size=protocol_world_size,
        device=device,
        state_dir=state_dir,
        local=local,
        dev_insecure=bool(args.dev_insecure),
        static_buckets=static_buckets,
    )


#: Optional JSON file of {uid: BucketCreds fields} — LOCAL OVERRIDE of chain-derived
#: bucket discovery (e.g. offline rigs, or pointing at a mirror). Normal operation
#: needs none: ChainClient.get_bucket recovers onboarding BucketCommits from history.
STATIC_BUCKETS_ENV = "MOK_STATIC_BUCKETS"


#: Reserved key in the MOK_STATIC_BUCKETS file: the bucket holding the frozen
#: step-A dataset trees when they live apart from the owner's operational bucket.
DATASET_BUCKET_KEY = "dataset"
#: Sentinel uid under which the dataset bucket is stored in the static map.
DATASET_BUCKET_UID = -1


def load_static_buckets(env: Mapping[str, str] | None = None) -> dict[int, BucketCreds]:
    """Parse the MOK_STATIC_BUCKETS JSON file into {uid: BucketCreds}; {} if unset.

    Keys are uids; the reserved key ``"dataset"`` (stored as DATASET_BUCKET_UID)
    names the bucket the shard trees are read from when it differs from uid 0's.
    """
    path = (env if env is not None else os.environ).get(STATIC_BUCKETS_ENV, "")
    if not path:
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        out: dict[int, BucketCreds] = {}
        for key, creds in raw.items():
            uid = DATASET_BUCKET_UID if key == DATASET_BUCKET_KEY else int(key)
            out[uid] = BucketCreds(**creds)
        return out
    except (OSError, ValueError, TypeError) as e:
        raise BootstrapError(f"unreadable {STATIC_BUCKETS_ENV} file {path!r}: {e}") from e


def _bucket_from_env(hotkey_ss58: str) -> BucketCreds:
    """This node's own R2 bucket (WRITE pair) from the step-B onboarding env.
    Wire v2: the bucket is named after the hotkey; R2_BUCKET_NAME, if set, must match."""
    from B.onboarding.wallet_setup import OnboardingError, write_creds_from_env  # noqa: PLC0415

    try:
        return write_creds_from_env(hotkey_ss58)
    except OnboardingError as e:
        raise BootstrapError(str(e)) from e
