"""Tests for fleet/calibration/local_harness.py — MemoryStorage semantics, the
scripted chain, the stepped clock, and local_manifest.

Reuses the tiny shard rig from tests/unit/test_window_runner.py (its plain
builder functions, per that module's sharing convention).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import test_window_runner as twr

from fleet.calibration.local_harness import (
    LoopbackClock,
    MemoryStorage,
    ScriptedChain,
    local_manifest,
)
from mok_core.chain.schemas import WindowCommit
from mok_core.config.schemas import BucketCreds, StorageConfig
from mok_core.determinism import hash_bytes
from mok_core.storage import IntegrityError, ObjectMissingError, ObjectTooLargeError


def creds(name: str) -> BucketCreds:
    return BucketCreds(
        account_id="local", bucket_name=name, access_key_id="k", secret_access_key="s"
    )


@pytest.fixture
def storage(tmp_path: Path) -> MemoryStorage:
    return MemoryStorage(tmp_path / "root", creds("own"), clock=lambda: 1234.5)


# --------------------------------------------------------------------------- #
# MemoryStorage — the StorageClient surface
# --------------------------------------------------------------------------- #


async def test_put_get_roundtrip_and_exists(storage: MemoryStorage) -> None:
    await storage.put_bytes("payloads/w00000001/uid00003-v1.zst", b"hello")
    assert await storage.object_exists(creds("own"), "payloads/w00000001/uid00003-v1.zst")
    assert not await storage.object_exists(creds("own"), "payloads/w00000001/uid00004-v1.zst")
    got = await storage.get_bytes(creds("own"), "payloads/w00000001/uid00003-v1.zst")
    assert got == b"hello"


async def test_get_missing_raises_object_missing(storage: MemoryStorage) -> None:
    with pytest.raises(ObjectMissingError, match="own/nope.json"):
        await storage.get_bytes(creds("own"), "nope.json")


async def test_get_verifies_expected_hash(storage: MemoryStorage) -> None:
    await storage.put_bytes("a.bin", b"data")
    ok = await storage.get_bytes(creds("own"), "a.bin", expected_hash=hash_bytes(b"data"))
    assert ok == b"data"
    with pytest.raises(IntegrityError):
        await storage.get_bytes(creds("own"), "a.bin", expected_hash="00" * 32)


async def test_get_enforces_max_bytes(storage: MemoryStorage) -> None:
    await storage.put_bytes("big.bin", b"x" * 100)
    with pytest.raises(ObjectTooLargeError):
        await storage.get_bytes(creds("own"), "big.bin", max_bytes=99)
    small = MemoryStorage(storage.root, creds("own"), cfg=StorageConfig(max_payload_bytes=10))
    with pytest.raises(ObjectTooLargeError):  # default limit comes from cfg
        await small.get_bytes(creds("own"), "big.bin")


async def test_object_timestamp_uses_injected_clock(storage: MemoryStorage) -> None:
    await storage.put_bytes("t.json", b"{}")
    assert await storage.object_timestamp(creds("own"), "t.json") == 1234.5
    with pytest.raises(ObjectMissingError):
        await storage.object_timestamp(creds("own"), "absent.json")


async def test_list_keys_sorted_and_prefixed(storage: MemoryStorage) -> None:
    for key in ("certificates/w00000002.json", "certificates/w00000001.json", "manifest.json"):
        await storage.put_bytes(key, b"{}")
    assert await storage.list_keys(creds("own"), "certificates/") == [
        "certificates/w00000001.json",
        "certificates/w00000002.json",
    ]
    assert await storage.list_keys(creds("own"), "") == [
        "certificates/w00000001.json",
        "certificates/w00000002.json",
        "manifest.json",
    ]
    assert await storage.list_keys(creds("elsewhere"), "") == []


async def test_upload_download_file(storage: MemoryStorage, tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"file-bytes")
    await storage.upload_file("files/f.bin", src)
    dest = tmp_path / "deep" / "dest.bin"
    await storage.download_file(
        creds("own"), "files/f.bin", dest, expected_hash=hash_bytes(b"file-bytes")
    )
    assert dest.read_bytes() == b"file-bytes"


async def test_cross_bucket_reads(tmp_path: Path) -> None:
    alice = MemoryStorage(tmp_path / "root", creds("alice"))
    bob = MemoryStorage(tmp_path / "root", creds("bob"))
    await alice.put_bytes("shared.json", b"from-alice")
    assert await bob.get_bytes(creds("alice"), "shared.json") == b"from-alice"


async def test_gather_bytes_uid_ascending_with_reason_prefixes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    s3 = MemoryStorage(root, creds("uid3"))
    MemoryStorage(root, creds("uid5"))  # creates the bucket, no object
    s9 = MemoryStorage(root, creds("uid9"))
    await s3.put_bytes("payloads/w00000001/uid00003-v1.zst", b"three")
    await s9.put_bytes("payloads/w00000001/uid00009-v1.zst", b"nine")

    peers = {9: creds("uid9"), 5: creds("uid5"), 3: creds("uid3")}
    result = await s3.gather_bytes(
        peers,
        lambda uid: f"payloads/w00000001/uid{uid:05d}-v1.zst",
        expected_hashes={3: hash_bytes(b"three"), 9: hash_bytes(b"WRONG")},
        deadline_s=5.0,
    )
    assert list(result.ok) == [3]  # ascending, failures excluded
    assert result.ok[3] == b"three"
    assert result.failed[5].startswith("missing: ")
    assert result.failed[9].startswith("integrity: ")


async def test_malformed_keys_rejected(storage: MemoryStorage) -> None:
    from mok_core.storage import StorageError

    with pytest.raises(StorageError, match="malformed key"):
        await storage.put_bytes("../escape.json", b"x")
    with pytest.raises(StorageError, match="malformed key"):
        await storage.get_bytes(creds("own"), "/abs.json")


# --------------------------------------------------------------------------- #
# ScriptedChain + LoopbackClock + local_manifest
# --------------------------------------------------------------------------- #


def test_scripted_chain_records_and_replays_commits() -> None:
    chain = ScriptedChain(uid=3)
    commit = WindowCommit(window=4, payload_hash="aa" * 32, state_root="bb" * 32,
                          theta_end_hash="cc" * 32)
    chain.commit_window(commit)
    assert chain.get_window_commits(4) == {3: commit}
    assert chain.get_window_commits(5) == {}
    assert chain.my_uid() == 3
    assert chain.block_hash(1) != chain.block_hash(2)
    assert len(chain.block_hash(1)) == 32
    assert chain.sign(b"anything") == b""


def test_loopback_clock_gate_arithmetic() -> None:
    clock = LoopbackClock(seconds_per_window=1000.0, gate_offset_s=10.0)
    assert clock.boundary_ts(4) == 4000.0
    clock.enter_gate(4)
    assert clock.now() == 5010.0  # inside window 4's gate (boundary(5) + 10 < +90)
    with pytest.raises(ValueError):
        LoopbackClock(gate_offset_s=-1.0)


def test_local_manifest_derives_ref_from_files(tmp_path: Path) -> None:
    twr.write_shard_files(tmp_path)
    index = twr.build_index(tmp_path)
    cfg = twr.make_run_cfg()
    manifest = local_manifest(
        cfg, index, shard_path=lambda i: tmp_path / f"shard-{i}.bin", run_seed=bytes(32)
    )
    ref = manifest.datasets[0]
    assert ref.name == index.name == "bulk"
    assert ref.num_shards == twr.NUM_SHARDS
    assert ref.shard_bytes == (tmp_path / "shard-0.bin").stat().st_size
    assert ref.tokens_total == twr.NUM_SHARDS * ref.shard_bytes // 2
    assert manifest.prf.run_seed_hex == "00" * 32
    # deterministic: building twice gives the same manifest hash
    again = local_manifest(
        cfg, index, shard_path=lambda i: tmp_path / f"shard-{i}.bin", run_seed=bytes(32)
    )
    assert again.manifest_hash() == manifest.manifest_hash()
