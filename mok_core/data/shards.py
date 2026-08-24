"""Content-addressed shard files: raw little-endian uint16 tokens, mmap reads.

A shard is exactly `num_sequences * seq_len` tokens named
`shard-<first-16-hex-of-blake2b-256>.bin`; its full-file blake2b-256 digest is
the Merkle leaf committed in the dataset manifest. Dataprep writes shards; this
module reads and verifies them.

Memory-mapped shard lifecycle, narrowed to flat uint16 token files with
content addressing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
from pydantic import field_validator

from mok_core.config.manifest import DatasetManifestRef
from mok_core.config.schemas import FrozenModel

from .merkle import MerkleTree

_DIGEST_SIZE = 32
_TOKEN_DTYPE = np.dtype("<u2")
SHARD_NAME_HEX_CHARS = 16


def shard_leaf_hash(path: str | os.PathLike[str]) -> bytes:
    """blake2b-256 over the raw shard file bytes — the Merkle leaf."""
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.digest()


def shard_filename(leaf_hash: bytes) -> str:
    """Canonical content-addressed filename for a shard with this leaf hash."""
    if len(leaf_hash) != _DIGEST_SIZE:
        raise ValueError(f"leaf hash must be {_DIGEST_SIZE} bytes, got {len(leaf_hash)}")
    return f"shard-{leaf_hash.hex()[:SHARD_NAME_HEX_CHARS]}.bin"


class ShardReader:
    """Zero-copy mmap view over one shard; `.sequence(i)` returns an owned copy."""

    def __init__(self, path: str | os.PathLike[str], seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        self.path = Path(path)
        self.seq_len = seq_len
        size = self.path.stat().st_size
        seq_bytes = seq_len * _TOKEN_DTYPE.itemsize
        if size == 0 or size % seq_bytes != 0:
            raise ValueError(
                f"{self.path.name}: {size} bytes is not a positive multiple of seq_len * 2 ({seq_bytes})"
            )
        self.num_sequences = size // seq_bytes
        self._mmap: np.memmap | None = np.memmap(self.path, dtype=_TOKEN_DTYPE, mode="r").reshape(
            self.num_sequences, seq_len
        )

    def sequence(self, i: int) -> np.ndarray:
        """Tokens of sequence `i` as an owned uint16 array (safe after close)."""
        if self._mmap is None:
            raise ValueError(f"{self.path.name}: reader is closed")
        if not 0 <= i < self.num_sequences:
            raise IndexError(f"sequence {i} out of range [0, {self.num_sequences})")
        return np.array(self._mmap[i], dtype=np.uint16, copy=True)

    def verify(self, expected_leaf_hash: bytes) -> bool:
        """Hash the file bytes and compare with the manifest leaf."""
        return shard_leaf_hash(self.path) == expected_leaf_hash

    def close(self) -> None:
        if self._mmap is not None:
            mm = self._mmap
            self._mmap = None
            del mm  # drop the last reference; numpy releases the mapping

    def __enter__(self) -> ShardReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DatasetShardIndex(FrozenModel):
    """Sidecar emitted by dataprep next to the dataset manifest: the ordered leaf
    hashes behind `DatasetManifestRef.merkle_root` (the ref stores the root only)."""

    name: str
    seq_len: int
    shard_hashes: list[str]  # hex blake2b-256, one per shard, in shard-index order

    @field_validator("shard_hashes")
    @classmethod
    def _hex_leaves(cls, v: list[str]) -> list[str]:
        for i, s in enumerate(v):
            if len(s) != 2 * _DIGEST_SIZE or any(c not in "0123456789abcdef" for c in s):
                raise ValueError(f"shard_hashes[{i}] must be lowercase 64-char hex")
        return v

    @property
    def num_shards(self) -> int:
        return len(self.shard_hashes)

    def leaf(self, i: int) -> bytes:
        return bytes.fromhex(self.shard_hashes[i])

    def merkle(self) -> MerkleTree:
        return MerkleTree([self.leaf(i) for i in range(self.num_shards)])


def verify_index_matches_ref(index: DatasetShardIndex, ref: DatasetManifestRef) -> None:
    """Raise ValueError unless `index` is exactly the tree committed in `ref`."""
    if index.name != ref.name:
        raise ValueError(f"index is for dataset {index.name!r}, ref is {ref.name!r}")
    if index.seq_len != ref.seq_len:
        raise ValueError(f"seq_len mismatch: index {index.seq_len}, ref {ref.seq_len}")
    if index.num_shards != ref.num_shards:
        raise ValueError(f"shard count mismatch: index {index.num_shards}, ref {ref.num_shards}")
    root = index.merkle().root.hex()
    if root != ref.merkle_root:
        raise ValueError(
            f"merkle root mismatch for dataset {ref.name!r}: index {root}, ref {ref.merkle_root}"
        )
