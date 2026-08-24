"""Manifest building, Merkle golden vector, and local verification
(dataprep/pipeline/build_manifest.py, dataprep/pipeline/verify.py)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dataprep.pipeline.build_manifest import (
    MANIFEST_FILENAME,
    SHARD_INDEX_FILENAME,
    build_dataset_manifest,
    load_manifest_ref,
    load_shard_index,
)
from dataprep.pipeline.shard_writer import write_shards
from dataprep.pipeline.verify import verify_local, verify_sample
from mok_core.data.shards import verify_index_matches_ref

_TOKENIZER_HASH = "00" * 32


def golden_seqs() -> list[np.ndarray]:
    return [np.arange(i * 16, (i + 1) * 16, dtype=np.uint16) for i in range(5)]


def build_golden(tmp_path):
    metas = write_shards(golden_seqs(), tmp_path, shard_sequences=2, seq_len=16)
    index, ref = build_dataset_manifest(
        metas,
        name="golden",
        seq_len=16,
        tokenizer_hash=_TOKENIZER_HASH,
        out_dir=tmp_path,
        shard_sequences=2,
    )
    return metas, index, ref


def test_merkle_root_golden(tmp_path):
    """Fixed fixture -> fixed shard leaves -> fixed root: pins leaf-hash wire
    format, shard byte layout, and the Merkle pairing rule end to end."""
    metas, _, ref = build_golden(tmp_path)
    # consensus constant — change requires SPEC_VERSION bump
    assert metas[0].hash_hex == "aeb002c6a5f6cae8fd527ea8a0a7ff34ba747147ac618fe6a6867ff0318c5554"
    # consensus constant — change requires SPEC_VERSION bump
    assert ref.merkle_root == "d1d9efe6a50fe7b3008979943269ef096dc794e6154c2e28bf8cf2d56d183006"


def test_manifest_fields_and_files(tmp_path):
    metas, index, ref = build_golden(tmp_path)
    assert ref.num_shards == 3
    assert ref.shard_bytes == 2 * 16 * 2
    assert ref.tokens_total == 5 * 16
    assert ref.seq_len == 16
    assert ref.tokenizer_hash == _TOKENIZER_HASH
    assert index.shard_hashes == [m.hash_hex for m in metas]
    # written files round-trip and satisfy the shared consistency check
    loaded_index = load_shard_index(tmp_path / SHARD_INDEX_FILENAME)
    loaded_ref = load_manifest_ref(tmp_path / MANIFEST_FILENAME)
    assert loaded_index == index
    assert loaded_ref == ref
    verify_index_matches_ref(loaded_index, loaded_ref)


def test_build_rejects_empty_and_non_full_intermediate(tmp_path):
    with pytest.raises(ValueError, match="zero shards"):
        build_dataset_manifest(
            [], name="x", seq_len=16, tokenizer_hash=_TOKENIZER_HASH, out_dir=tmp_path
        )
    metas = write_shards(golden_seqs(), tmp_path, shard_sequences=2, seq_len=16)
    reordered = [metas[2], metas[0], metas[1]]  # partial shard is no longer last
    with pytest.raises(ValueError, match="partial"):
        build_dataset_manifest(
            reordered,
            name="x",
            seq_len=16,
            tokenizer_hash=_TOKENIZER_HASH,
            out_dir=tmp_path,
            shard_sequences=2,
        )


def test_verify_local_passes_on_clean_dir(tmp_path):
    _, _, ref = build_golden(tmp_path)
    report = verify_local(tmp_path)
    assert report.ok
    assert report.merkle_root == ref.merkle_root
    assert report.shards_hashed == report.num_shards == 3
    assert report.tokens_total == 80


def test_verify_detects_shard_tamper(tmp_path):
    metas, _, _ = build_golden(tmp_path)
    data = bytearray(metas[1].path.read_bytes())
    data[7] ^= 0xFF  # same size, different content
    metas[1].path.write_bytes(bytes(data))
    report = verify_local(tmp_path)
    assert not report.ok
    assert any("shard 1" in f and "hash" in f for f in report.failures)


def test_verify_detects_missing_shard(tmp_path):
    metas, _, _ = build_golden(tmp_path)
    metas[0].path.unlink()
    report = verify_local(tmp_path)
    assert not report.ok
    assert any("missing" in f for f in report.failures)


def test_verify_detects_manifest_tamper(tmp_path):
    build_golden(tmp_path)
    path = tmp_path / MANIFEST_FILENAME
    doc = json.loads(path.read_text())
    doc["tokens_total"] += 16
    path.write_text(json.dumps(doc))
    report = verify_local(tmp_path)
    assert not report.ok


def test_verify_sample_mode(tmp_path):
    build_golden(tmp_path)
    report = verify_sample(tmp_path, 1, seed=0)
    assert report.ok
    assert report.shards_hashed == 1
    # deterministic sample: same seed picks the same shard set
    again = verify_sample(tmp_path, 1, seed=0)
    assert again.shards_hashed == 1 and again.ok
