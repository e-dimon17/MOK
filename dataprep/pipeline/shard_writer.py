"""Content-addressed 512 MiB shard emission — the tokenize/pack output stage.

A full shard is exactly 65,536 sequences x 4,096 tokens x 2 bytes = 512 MiB
of raw little-endian uint16, hashed incrementally while writing and renamed
to `shard-<first-16-hex>.bin` (the full blake2b-256 digest is the Merkle
leaf). The final shard may be partial; its true sequence count is recorded.

Writes raw little-endian uint16 token shards with content addressing.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

from mok_core.data.shards import shard_filename

FULL_SHARD_SEQUENCES = 65536  # x 4096 tokens x 2 bytes == 512 MiB exactly
SHARD_METAS_FILENAME = "shards.json"
_DIGEST_SIZE = 32
_TOKEN_DTYPE = np.dtype("<u2")


@dataclass(frozen=True)
class ShardMeta:
    """One written shard: content-addressed path, full leaf hash, true length."""

    path: Path
    hash_hex: str
    num_sequences: int


def write_shards(
    seq_iter: Iterable[np.ndarray],
    out_dir: str | PathLike[str],
    *,
    shard_sequences: int = FULL_SHARD_SEQUENCES,
    seq_len: int = 4096,
) -> list[ShardMeta]:
    """Buffer exact-length uint16 sequences into content-addressed shard files.

    Each shard holds `shard_sequences` sequences (the last may hold fewer);
    bytes are hashed while written, then the temp file is renamed to its
    content-addressed name. Returns metas in write order — that order is the
    Merkle leaf order.
    """
    if shard_sequences <= 0 or seq_len <= 0:
        raise ValueError("shard_sequences and seq_len must be positive")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metas: list[ShardMeta] = []
    fh = None
    hasher = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    count = 0
    tmp = out / ".shard-inflight.tmp"

    def finalize() -> None:
        nonlocal fh, hasher, count
        assert fh is not None
        fh.close()
        fh = None
        digest = hasher.digest()
        final = out / shard_filename(digest)
        os.replace(tmp, final)
        metas.append(ShardMeta(path=final, hash_hex=digest.hex(), num_sequences=count))
        hasher = hashlib.blake2b(digest_size=_DIGEST_SIZE)
        count = 0

    try:
        for seq in seq_iter:
            arr = np.asarray(seq)
            if arr.dtype != np.uint16:
                raise ValueError(f"sequence dtype must be uint16, got {arr.dtype}")
            if arr.shape != (seq_len,):
                raise ValueError(f"sequence shape must be ({seq_len},), got {arr.shape}")
            data = arr.astype(_TOKEN_DTYPE, copy=False).tobytes()
            if fh is None:
                fh = open(tmp, "wb")  # noqa: SIM115 — lifetime spans loop iterations; closed in finalize/finally
            fh.write(data)
            hasher.update(data)
            count += 1
            if count == shard_sequences:
                finalize()
        if fh is not None and count > 0:
            finalize()
    finally:
        if fh is not None:
            fh.close()
        tmp.unlink(missing_ok=True)
    return metas


def save_shard_metas(metas: list[ShardMeta], path: str | PathLike[str]) -> None:
    """Persist write-order shard metas (filenames relative to the json's dir)."""
    payload = [
        {"file": m.path.name, "hash_hex": m.hash_hex, "num_sequences": m.num_sequences} for m in metas
    ]
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    os.replace(tmp, p)


def load_shard_metas(path: str | PathLike[str]) -> list[ShardMeta]:
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        payload = json.load(f)
    return [
        ShardMeta(path=p.parent / e["file"], hash_hex=e["hash_hex"], num_sequences=e["num_sequences"])
        for e in payload
    ]
