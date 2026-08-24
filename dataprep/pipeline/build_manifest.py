"""Dataset commitment: shard index, Merkle root, manifest ref.

Emits the `DatasetShardIndex` sidecar (ordered leaf hashes), computes the
Merkle root via mok_core.data.merkle, and writes the `DatasetManifestRef`
that the subnet owner commits into the on-chain run manifest. After this,
changing a single shard byte breaks the root.
"""

from __future__ import annotations

import json
import os
from os import PathLike
from pathlib import Path

from mok_core.config.manifest import DatasetManifestRef
from mok_core.data.shards import DatasetShardIndex

from .shard_writer import FULL_SHARD_SEQUENCES, ShardMeta

SHARD_INDEX_FILENAME = "shard_index.json"
MANIFEST_FILENAME = "manifest.json"
TOKENIZER_FILENAME = "tokenizer.json"


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(tmp, path)


def build_dataset_manifest(
    metas: list[ShardMeta],
    *,
    name: str,
    seq_len: int,
    tokenizer_hash: str,
    out_dir: str | PathLike[str],
    shard_sequences: int = FULL_SHARD_SEQUENCES,
) -> tuple[DatasetShardIndex, DatasetManifestRef]:
    """Build and write `shard_index.json` + `manifest.json` for written shards.

    `metas` must be in write order (Merkle leaf order); every shard except the
    last must be full. Returns the validated index and manifest ref.
    """
    if not metas:
        raise ValueError("cannot build a manifest for zero shards")
    for i, m in enumerate(metas[:-1]):
        if m.num_sequences != shard_sequences:
            raise ValueError(
                f"shard {i} has {m.num_sequences} sequences; only the final shard may be partial"
            )
    last = metas[-1]
    if not 0 < last.num_sequences <= shard_sequences:
        raise ValueError(f"final shard has {last.num_sequences} sequences (max {shard_sequences})")

    index = DatasetShardIndex(name=name, seq_len=seq_len, shard_hashes=[m.hash_hex for m in metas])
    ref = DatasetManifestRef(
        name=name,
        merkle_root=index.merkle().root.hex(),
        num_shards=len(metas),
        shard_bytes=shard_sequences * seq_len * 2,
        seq_len=seq_len,
        tokens_total=sum(m.num_sequences for m in metas) * seq_len,
        tokenizer_hash=tokenizer_hash,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / SHARD_INDEX_FILENAME, index.model_dump(mode="json"))
    _write_json(out / MANIFEST_FILENAME, ref.model_dump(mode="json"))
    return index, ref


def load_shard_index(path: str | PathLike[str]) -> DatasetShardIndex:
    with open(path, encoding="utf-8") as f:
        return DatasetShardIndex.model_validate(json.load(f))


def load_manifest_ref(path: str | PathLike[str]) -> DatasetManifestRef:
    with open(path, encoding="utf-8") as f:
        return DatasetManifestRef.model_validate(json.load(f))
