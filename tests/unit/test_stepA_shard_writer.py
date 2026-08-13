"""Shard writer round-trip and boundary math (A/pipeline/shard_writer.py)."""

from __future__ import annotations

import numpy as np
import pytest

from A.pipeline.shard_writer import (
    FULL_SHARD_SEQUENCES,
    ShardMeta,
    load_shard_metas,
    save_shard_metas,
    write_shards,
)
from mok_core.data.shards import ShardReader, shard_filename, shard_leaf_hash


def seqs(n: int, seq_len: int = 16) -> list[np.ndarray]:
    return [np.arange(i * seq_len, (i + 1) * seq_len, dtype=np.uint16) for i in range(n)]


def test_full_shard_is_exactly_512_mib():
    # consensus constant — change requires SPEC_VERSION bump
    assert FULL_SHARD_SEQUENCES * 4096 * 2 == 512 * 1024 * 1024 == 536_870_912


def test_write_shards_sizes_and_content_addressing(tmp_path):
    metas = write_shards(seqs(5), tmp_path, shard_sequences=2, seq_len=16)
    assert [m.num_sequences for m in metas] == [2, 2, 1]
    for m in metas:
        assert m.path.stat().st_size == m.num_sequences * 16 * 2
        assert m.path.name == shard_filename(bytes.fromhex(m.hash_hex))
        assert shard_leaf_hash(m.path).hex() == m.hash_hex
    assert not list(tmp_path.glob("*.tmp"))


def test_round_trip_via_shard_reader(tmp_path):
    original = seqs(5)
    metas = write_shards(original, tmp_path, shard_sequences=2, seq_len=16)
    read_back: list[np.ndarray] = []
    for m in metas:
        with ShardReader(m.path, seq_len=16) as reader:
            assert reader.num_sequences == m.num_sequences
            assert reader.verify(bytes.fromhex(m.hash_hex))
            read_back.extend(reader.sequence(i) for i in range(reader.num_sequences))
    for x, y in zip(original, read_back, strict=True):
        assert np.array_equal(x, y)


def test_single_partial_shard(tmp_path):
    metas = write_shards(seqs(1), tmp_path, shard_sequences=4, seq_len=16)
    assert len(metas) == 1
    assert metas[0].num_sequences == 1
    assert metas[0].path.stat().st_size == 16 * 2


def test_empty_iterator_writes_nothing(tmp_path):
    assert write_shards([], tmp_path, shard_sequences=4, seq_len=16) == []
    assert list(tmp_path.iterdir()) == []


def test_exact_multiple_leaves_no_partial(tmp_path):
    metas = write_shards(seqs(4), tmp_path, shard_sequences=2, seq_len=16)
    assert [m.num_sequences for m in metas] == [2, 2]


def test_rejects_wrong_dtype_and_shape(tmp_path):
    with pytest.raises(ValueError, match="dtype"):
        write_shards([np.zeros(16, dtype=np.int32)], tmp_path, shard_sequences=2, seq_len=16)
    with pytest.raises(ValueError, match="shape"):
        write_shards([np.zeros(15, dtype=np.uint16)], tmp_path, shard_sequences=2, seq_len=16)
    assert not list(tmp_path.glob("*.tmp"))  # in-flight temp cleaned up on error


def test_deterministic_hashes(tmp_path):
    a = write_shards(seqs(5), tmp_path / "a", shard_sequences=2, seq_len=16)
    b = write_shards(seqs(5), tmp_path / "b", shard_sequences=2, seq_len=16)
    assert [m.hash_hex for m in a] == [m.hash_hex for m in b]


def test_shard_metas_json_round_trip(tmp_path):
    metas = write_shards(seqs(3), tmp_path, shard_sequences=2, seq_len=16)
    path = tmp_path / "shards.json"
    save_shard_metas(metas, path)
    loaded = load_shard_metas(path)
    assert loaded == [ShardMeta(path=m.path, hash_hex=m.hash_hex, num_sequences=m.num_sequences) for m in metas]
