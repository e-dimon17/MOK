"""ShardCache: verified installs, cache hits, LRU eviction, adoption, corruption."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import numpy as np
import pytest

from mok_core.config.schemas import DataConfig
from mok_core.data.download import ShardCache, ShardCacheError, ShardVerificationError
from mok_core.data.shards import DatasetShardIndex, ShardReader, shard_filename

SEQ_LEN = 8
NUM_SEQ = 2
SHARD_BYTES = 2 * SEQ_LEN * NUM_SEQ  # 32 bytes per shard
NUM_SHARDS = 4


def _shard_bytes(i: int) -> bytes:
    return ((np.arange(NUM_SEQ * SEQ_LEN, dtype=np.int64) + i * 100) % 65536).astype("<u2").tobytes()


SHARDS = {i: _shard_bytes(i) for i in range(NUM_SHARDS)}
INDEX = DatasetShardIndex(
    name="bulk",
    seq_len=SEQ_LEN,
    shard_hashes=[hashlib.blake2b(SHARDS[i], digest_size=32).hexdigest() for i in range(NUM_SHARDS)],
)


class Fetcher:
    """Counts calls; can serve corrupted bytes for chosen indices."""

    def __init__(self, corrupt: set[int] | None = None) -> None:
        self.calls: list[int] = []
        self.corrupt = corrupt or set()

    async def __call__(self, shard_index: int) -> bytes:
        self.calls.append(shard_index)
        await asyncio.sleep(0)
        data = SHARDS[shard_index]
        if shard_index in self.corrupt:
            data = b"\xff" + data[1:]
        return data


def _cache(tmp_path: Path, max_bytes: int = 10 * SHARD_BYTES) -> ShardCache:
    return ShardCache(tmp_path, max_bytes, INDEX)


async def test_prefetch_installs_and_verifies(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    fetch = Fetcher()
    paths = await cache.prefetch([0, 1], fetch)
    assert sorted(fetch.calls) == [0, 1]
    assert set(paths) == {0, 1}
    for i, path in paths.items():
        assert path == tmp_path / "bulk" / shard_filename(INDEX.leaf(i))
        assert path.read_bytes() == SHARDS[i]
        with ShardReader(path, SEQ_LEN) as reader:
            assert reader.verify(INDEX.leaf(i))
    assert cache.resident_bytes == 2 * SHARD_BYTES
    assert cache.path_for(0) == paths[0]


async def test_cache_hit_skips_fetch(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    first = Fetcher()
    await cache.prefetch([0, 1], first)
    second = Fetcher()
    paths = await cache.prefetch([0, 1, 2], second)
    assert second.calls == [2]
    assert set(paths) == {0, 1, 2}


async def test_duplicate_indices_fetch_once(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    fetch = Fetcher()
    paths = await cache.prefetch([3, 3, 3], fetch)
    assert fetch.calls == [3]
    assert set(paths) == {3}


async def test_corrupt_fetch_rejected(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    fetch = Fetcher(corrupt={1})
    with pytest.raises(ShardVerificationError, match="shard 1"):
        await cache.prefetch([1], fetch)
    assert not cache.has(1)
    assert not (tmp_path / "bulk" / shard_filename(INDEX.leaf(1))).exists()
    assert not list((tmp_path / "bulk").glob("*.tmp"))  # nothing half-installed


async def test_lru_eviction(tmp_path: Path) -> None:
    cache = _cache(tmp_path, max_bytes=2 * SHARD_BYTES)
    fetch = Fetcher()
    await cache.prefetch([0], fetch)
    await cache.prefetch([1], fetch)
    cache.path_for(0)  # touch: 1 becomes least recently used
    await cache.prefetch([2], fetch)
    assert cache.has(0) and cache.has(2) and not cache.has(1)
    assert not (tmp_path / "bulk" / shard_filename(INDEX.leaf(1))).exists()
    assert cache.resident_bytes == 2 * SHARD_BYTES
    with pytest.raises(FileNotFoundError):
        cache.path_for(1)


async def test_pinned_set_larger_than_budget_raises(tmp_path: Path) -> None:
    cache = _cache(tmp_path, max_bytes=SHARD_BYTES)  # room for one shard only
    with pytest.raises(ShardCacheError, match="budget"):
        await cache.prefetch([0, 1], Fetcher(), concurrency=1)


async def test_scan_adopts_verified_files(tmp_path: Path) -> None:
    await _cache(tmp_path).prefetch([0, 2], Fetcher())
    reopened = _cache(tmp_path)
    assert reopened.has(0) and reopened.has(2) and not reopened.has(1)
    fetch = Fetcher()
    await reopened.prefetch([0, 2], fetch)
    assert fetch.calls == []


async def test_scan_drops_corrupt_and_tmp_files(tmp_path: Path) -> None:
    shard_dir = tmp_path / "bulk"
    shard_dir.mkdir(parents=True)
    bad = shard_dir / shard_filename(INDEX.leaf(1))
    bad.write_bytes(b"garbage")
    stale = shard_dir / "shard-0011223344556677.bin.tmp"
    stale.write_bytes(b"partial")
    cache = _cache(tmp_path)
    assert not cache.has(1)
    assert not bad.exists()
    assert not stale.exists()


async def test_prefetch_bounds(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    with pytest.raises(IndexError, match="out of range"):
        await cache.prefetch([NUM_SHARDS], Fetcher())


def test_from_config(tmp_path: Path) -> None:
    cfg = DataConfig(shard_cache_dir=str(tmp_path / "cache"), shard_cache_max_bytes=1024)
    cache = ShardCache.from_config(cfg, INDEX)
    assert cache.dir == tmp_path / "cache" / "bulk"
    assert cache.max_bytes == 1024
    assert cache.dir.is_dir()


def test_bad_budget(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        ShardCache(tmp_path, 0, INDEX)
