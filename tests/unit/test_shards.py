"""Shard files: content addressing, mmap reads, verification, dataset index."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from mok_core.config.manifest import DatasetManifestRef
from mok_core.data.shards import (
    DatasetShardIndex,
    ShardReader,
    shard_filename,
    shard_leaf_hash,
    verify_index_matches_ref,
)

SEQ_LEN = 16
NUM_SEQ = 8


def _tokens(shard_no: int) -> np.ndarray:
    base = np.arange(NUM_SEQ * SEQ_LEN, dtype=np.int64)
    return ((base + shard_no * 1000) % 65536).astype("<u2")


def _write_shard(dirpath: Path, shard_no: int) -> Path:
    data = _tokens(shard_no).tobytes()
    digest = hashlib.blake2b(data, digest_size=32).digest()
    path = dirpath / shard_filename(digest)
    path.write_bytes(data)
    return path


def test_shard_filename_format() -> None:
    digest = hashlib.blake2b(b"x", digest_size=32).digest()
    name = shard_filename(digest)
    assert re.fullmatch(r"shard-[0-9a-f]{16}\.bin", name)
    assert name == f"shard-{digest.hex()[:16]}.bin"
    with pytest.raises(ValueError, match="32 bytes"):
        shard_filename(b"short")


def test_shard_leaf_hash_matches_hashlib(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, 0)
    expected = hashlib.blake2b(path.read_bytes(), digest_size=32).digest()
    assert shard_leaf_hash(path) == expected
    assert path.name == shard_filename(expected)


def test_reader_sequences(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, 3)
    tokens = _tokens(3).reshape(NUM_SEQ, SEQ_LEN)
    with ShardReader(path, SEQ_LEN) as reader:
        assert reader.num_sequences == NUM_SEQ
        for i in range(NUM_SEQ):
            seq = reader.sequence(i)
            assert seq.dtype == np.uint16
            assert seq.shape == (SEQ_LEN,)
            np.testing.assert_array_equal(seq, tokens[i])


def test_sequence_is_owned_copy(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, 1)
    reader = ShardReader(path, SEQ_LEN)
    seq = reader.sequence(2)
    reader.close()
    np.testing.assert_array_equal(seq, _tokens(1).reshape(NUM_SEQ, SEQ_LEN)[2])  # alive after close


def test_reader_verify(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, 5)
    reader = ShardReader(path, SEQ_LEN)
    assert reader.verify(shard_leaf_hash(path))
    assert not reader.verify(bytes(32))
    reader.close()


def test_reader_bounds_and_close(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, 0)
    reader = ShardReader(path, SEQ_LEN)
    with pytest.raises(IndexError):
        reader.sequence(NUM_SEQ)
    with pytest.raises(IndexError):
        reader.sequence(-1)
    reader.close()
    reader.close()  # idempotent
    with pytest.raises(ValueError, match="closed"):
        reader.sequence(0)


def test_reader_rejects_bad_geometry(tmp_path: Path) -> None:
    ragged = tmp_path / "shard-deadbeefdeadbeef.bin"
    ragged.write_bytes(b"\x00" * 33)  # not a multiple of seq_len * 2
    with pytest.raises(ValueError, match="multiple"):
        ShardReader(ragged, SEQ_LEN)
    empty = tmp_path / "shard-0000000000000000.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="multiple"):
        ShardReader(empty, SEQ_LEN)
    with pytest.raises(ValueError, match="positive"):
        ShardReader(ragged, 0)


# --------------------------------------------------------------------------- #
# DatasetShardIndex
# --------------------------------------------------------------------------- #


def _index(tmp_path: Path, n: int = 4) -> DatasetShardIndex:
    hashes = [shard_leaf_hash(_write_shard(tmp_path, i)).hex() for i in range(n)]
    return DatasetShardIndex(name="bulk", seq_len=SEQ_LEN, shard_hashes=hashes)


def _ref(index: DatasetShardIndex) -> DatasetManifestRef:
    return DatasetManifestRef(
        name=index.name,
        merkle_root=index.merkle().root.hex(),
        num_shards=index.num_shards,
        shard_bytes=2 * SEQ_LEN * NUM_SEQ,
        seq_len=index.seq_len,
        tokens_total=index.num_shards * SEQ_LEN * NUM_SEQ,
        tokenizer_hash="ab" * 32,
    )


def test_index_leaf_and_merkle(tmp_path: Path) -> None:
    index = _index(tmp_path)
    assert index.num_shards == 4
    for i in range(4):
        assert index.leaf(i) == bytes.fromhex(index.shard_hashes[i])
    tree = index.merkle()
    assert tree.num_leaves == 4
    assert index.merkle().root == tree.root  # rebuild is deterministic


def test_index_rejects_bad_hex() -> None:
    with pytest.raises(ValueError, match="64-char hex"):
        DatasetShardIndex(name="bulk", seq_len=16, shard_hashes=["zz" * 32])
    with pytest.raises(ValueError, match="64-char hex"):
        DatasetShardIndex(name="bulk", seq_len=16, shard_hashes=["ab" * 31])


def test_verify_index_matches_ref(tmp_path: Path) -> None:
    index = _index(tmp_path)
    verify_index_matches_ref(index, _ref(index))  # no raise


def test_verify_index_mismatches(tmp_path: Path) -> None:
    index = _index(tmp_path)
    ref = _ref(index)
    with pytest.raises(ValueError, match="merkle root"):
        verify_index_matches_ref(index, ref.model_copy(update={"merkle_root": "00" * 32}))
    with pytest.raises(ValueError, match="dataset"):
        verify_index_matches_ref(index, ref.model_copy(update={"name": "anneal"}))
    with pytest.raises(ValueError, match="seq_len"):
        verify_index_matches_ref(index, ref.model_copy(update={"seq_len": 32}))
    with pytest.raises(ValueError, match="count"):
        verify_index_matches_ref(index, ref.model_copy(update={"num_shards": 5}))
    tampered = index.model_copy(update={"shard_hashes": [index.shard_hashes[1], *index.shard_hashes[1:]]})
    with pytest.raises(ValueError, match="merkle root"):
        verify_index_matches_ref(tampered, ref)


def test_shard_files_verify_against_index(tmp_path: Path) -> None:
    index = _index(tmp_path)
    for i in range(index.num_shards):
        path = tmp_path / shard_filename(index.leaf(i))
        with ShardReader(path, SEQ_LEN) as reader:
            assert reader.verify(index.leaf(i))
