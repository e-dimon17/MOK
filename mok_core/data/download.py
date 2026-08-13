"""ShardCache — verified local NVMe cache of dataset shards.

Storage-agnostic: callers inject an async `fetch_fn(shard_index) -> bytes`
(the R2 client arrives in mok_core/storage); this module owns naming,
hash-verification against the dataset index BEFORE install, atomic installs
(tmp + rename), and LRU eviction to the configured byte budget. Shards being
prefetched are pinned and never evicted mid-window.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path

from mok_core.config.schemas import DataConfig

from .shards import DatasetShardIndex, shard_filename, shard_leaf_hash

FetchFn = Callable[[int], Awaitable[bytes]]


class ShardCacheError(RuntimeError):
    pass


class ShardVerificationError(ShardCacheError):
    """Fetched bytes did not hash to the manifest leaf — never installed."""


class ShardCache:
    """LRU byte-budgeted cache of one dataset's shards under `cache_dir/<name>/`."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        max_bytes: int,
        index: DatasetShardIndex,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        self.index = index
        self.max_bytes = max_bytes
        self.dir = Path(cache_dir).expanduser() / index.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._pinned: set[int] = set()
        # shard_idx -> file size; ordered least- to most-recently used.
        self._resident: OrderedDict[int, int] = OrderedDict()
        self._names = {shard_filename(index.leaf(i)): i for i in range(index.num_shards)}
        self._scan()

    @classmethod
    def from_config(cls, cfg: DataConfig, index: DatasetShardIndex) -> ShardCache:
        return cls(cfg.shard_cache_dir, cfg.shard_cache_max_bytes, index)

    # ------------------------------------------------------------------ #
    # lookup
    # ------------------------------------------------------------------ #

    def path_for(self, shard_idx: int) -> Path:
        """Local path of an installed shard (touches LRU). FileNotFoundError if absent."""
        path = self._path(shard_idx)
        if shard_idx not in self._resident:
            raise FileNotFoundError(f"shard {shard_idx} not in cache (expected {path})")
        self._resident.move_to_end(shard_idx)
        return path

    def has(self, shard_idx: int) -> bool:
        return shard_idx in self._resident

    @property
    def resident_bytes(self) -> int:
        return sum(self._resident.values())

    # ------------------------------------------------------------------ #
    # prefetch
    # ------------------------------------------------------------------ #

    async def prefetch(
        self,
        shard_indices: Iterable[int],
        fetch_fn: FetchFn,
        *,
        concurrency: int = 4,
    ) -> dict[int, Path]:
        """Ensure every shard is installed and verified; returns idx -> path.

        Missing shards are fetched concurrently, hash-verified against the
        dataset index leaf, installed atomically, then LRU-evicted down to the
        byte budget. The requested set is pinned for the duration of the call.
        """
        wanted = list(dict.fromkeys(int(i) for i in shard_indices))
        for i in wanted:
            if not 0 <= i < self.index.num_shards:
                raise IndexError(f"shard {i} out of range [0, {self.index.num_shards})")
        sem = asyncio.Semaphore(concurrency)

        async def _ensure(idx: int) -> Path:
            async with sem:
                async with self._lock:
                    if idx in self._resident:
                        self._resident.move_to_end(idx)
                        return self._path(idx)
                data = await fetch_fn(idx)
                expected = self.index.leaf(idx)
                actual = await asyncio.to_thread(_blake2b_256, data)
                if actual != expected:
                    raise ShardVerificationError(
                        f"shard {idx}: fetched bytes hash {actual.hex()[:16]}…, "
                        f"manifest leaf {expected.hex()[:16]}…"
                    )
                path = self._path(idx)
                await asyncio.to_thread(_atomic_write, path, data)
                async with self._lock:
                    self._resident[idx] = len(data)
                    self._resident.move_to_end(idx)
                    self._evict_locked()
                return path

        self._pinned.update(wanted)
        tasks = [asyncio.ensure_future(_ensure(i)) for i in wanted]
        try:
            paths = await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._pinned.difference_update(wanted)
        return dict(zip(wanted, paths, strict=True))

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _path(self, shard_idx: int) -> Path:
        return self.dir / shard_filename(self.index.leaf(shard_idx))

    def _scan(self) -> None:
        """Adopt shards installed by earlier runs (LRU-seeded by mtime); drop
        stale tmp files and re-verify adopted content before trusting it."""
        entries: list[tuple[float, int, int]] = []
        for p in self.dir.iterdir():
            if p.suffix == ".tmp":
                p.unlink(missing_ok=True)
                continue
            idx = self._names.get(p.name)
            if idx is None:
                continue
            if shard_leaf_hash(p) != self.index.leaf(idx):
                p.unlink(missing_ok=True)  # truncated/corrupt leftover
                continue
            st = p.stat()
            entries.append((st.st_mtime, idx, st.st_size))
        for _, idx, size in sorted(entries):
            self._resident[idx] = size
        self._evict_locked()

    def _evict_locked(self) -> None:
        """Drop least-recently-used unpinned shards until within budget."""
        while self.resident_bytes > self.max_bytes:
            victim = next((i for i in self._resident if i not in self._pinned), None)
            if victim is None:
                raise ShardCacheError(
                    f"pinned shards need {self.resident_bytes} bytes but the cache "
                    f"budget is {self.max_bytes} — raise DataConfig.shard_cache_max_bytes"
                )
            self._resident.pop(victim)
            self._path(victim).unlink(missing_ok=True)


def _blake2b_256(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
