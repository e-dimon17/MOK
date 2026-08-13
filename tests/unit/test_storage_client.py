"""StorageClient integration tests against a local ThreadedMotoServer.

Every test builds its own uniquely-named buckets on a module-scoped moto
server, so tests are independent and order-free.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

import pytest

from mok_core.config.schemas import BucketCreds, StorageConfig
from mok_core.determinism.hashing import hash_bytes
from mok_core.storage import (
    GatherResult,
    IntegrityError,
    ObjectMissingError,
    ObjectTooLargeError,
    StorageClient,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def moto_endpoint():
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture(scope="module")
def admin(moto_endpoint: str):
    """Sync boto3 client for bucket creation and out-of-band priming/inspection."""
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=moto_endpoint,
        region_name="us-east-1",
        aws_access_key_id="admin",
        aws_secret_access_key="admin",
    )


def make_creds(bucket_name: str, access_key: str = "test-key") -> BucketCreds:
    return BucketCreds(
        account_id="testaccount",
        bucket_name=bucket_name,
        access_key_id=access_key,
        secret_access_key="test-secret",
    )


def fresh_bucket(admin: Any, tag: str) -> str:
    name = f"mok-{tag}-{uuid.uuid4().hex[:8]}"
    admin.create_bucket(Bucket=name)
    return name


def make_client(
    creds: BucketCreds,
    endpoint: str,
    **cfg_overrides: Any,
) -> StorageClient:
    return StorageClient(
        creds,
        StorageConfig(**cfg_overrides),
        endpoint_override=endpoint,
        retry_base_delay_s=0.01,
    )


# --------------------------------------------------------------------------- #
# put/get
# --------------------------------------------------------------------------- #


async def test_put_get_roundtrip(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "rt"))
    data = os.urandom(50_000)
    async with make_client(creds, moto_endpoint) as sc:
        await sc.put_bytes("payloads/blob.zst", data)
        got = await sc.get_bytes(
            creds, "payloads/blob.zst", expected_hash=hash_bytes(data), max_bytes=1 << 20
        )
    assert got == data


async def test_get_missing_raises(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "miss"))
    async with make_client(creds, moto_endpoint) as sc:
        with pytest.raises(ObjectMissingError):
            await sc.get_bytes(creds, "does/not/exist", max_bytes=100)


async def test_hash_mismatch_raises_integrity_error(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "bad"))
    async with make_client(creds, moto_endpoint) as sc:
        await sc.put_bytes("k", b"tampered payload")
        with pytest.raises(IntegrityError):
            await sc.get_bytes(creds, "k", expected_hash="00" * 32, max_bytes=1 << 20)


async def test_oversize_rejected_by_head_before_any_download(admin: Any, moto_endpoint: str) -> None:
    """The size gate runs on HEAD alone: get_object must never be called."""
    creds = make_creds(fresh_bucket(admin, "big"))
    async with make_client(creds, moto_endpoint) as sc:
        await sc.put_bytes("big.bin", b"x" * 1000)
        s3 = await sc._client(creds)
        get_called = False
        original = s3.get_object

        async def spy(**kwargs: Any) -> Any:
            nonlocal get_called
            get_called = True
            return await original(**kwargs)

        s3.get_object = spy
        with pytest.raises(ObjectTooLargeError):
            await sc.get_bytes(creds, "big.bin", max_bytes=10)
        assert not get_called


# --------------------------------------------------------------------------- #
# upload_file / download_file
# --------------------------------------------------------------------------- #


async def test_upload_file_small_takes_single_put(admin: Any, moto_endpoint: str, tmp_path) -> None:
    creds = make_creds(fresh_bucket(admin, "up"))
    data = os.urandom(10_000)
    src = tmp_path / "small.bin"
    src.write_bytes(data)
    async with make_client(creds, moto_endpoint) as sc:
        await sc.upload_file("small.bin", src)
    head = admin.head_object(Bucket=creds.bucket_name, Key="small.bin")
    assert head["ContentLength"] == len(data)
    assert "-" not in head["ETag"]  # single-part upload => plain md5 ETag


async def test_upload_file_multipart_above_threshold(admin: Any, moto_endpoint: str, tmp_path) -> None:
    creds = make_creds(fresh_bucket(admin, "mpu"))
    data = os.urandom(11 * 1024 * 1024)  # 11 MiB -> three 5 MiB-min parts
    src = tmp_path / "large.bin"
    src.write_bytes(data)
    # Tiny threshold forces the multipart path; part size is clamped to the S3 5 MiB minimum.
    async with make_client(creds, moto_endpoint, multipart_threshold_bytes=1024) as sc:
        await sc.upload_file("large.bin", src)
        head = admin.head_object(Bucket=creds.bucket_name, Key="large.bin")
        assert head["ContentLength"] == len(data)
        assert head["ETag"].strip('"').endswith("-3")  # multipart ETag records 3 parts
        got = await sc.get_bytes(creds, "large.bin", expected_hash=hash_bytes(data), max_bytes=32 << 20)
    assert got == data


async def test_download_file_ranged_atomic(admin: Any, moto_endpoint: str, tmp_path) -> None:
    creds = make_creds(fresh_bucket(admin, "dl"))
    data = os.urandom(200_000)
    dest = tmp_path / "out" / "blob.bin"
    async with make_client(creds, moto_endpoint, download_chunk_bytes=16_384) as sc:
        await sc.put_bytes("blob.bin", data)
        await sc.download_file(creds, "blob.bin", dest, expected_hash=hash_bytes(data))
    assert dest.read_bytes() == data
    assert not os.path.exists(str(dest) + ".part")


async def test_download_file_resumes_truncated_part(admin: Any, moto_endpoint: str, tmp_path) -> None:
    """A truncated .part is treated as a valid prefix: only the tail is re-fetched."""
    creds = make_creds(fresh_bucket(admin, "resume"))
    data = os.urandom(200_000)
    dest = tmp_path / "blob.bin"
    part = tmp_path / "blob.bin.part"
    part.write_bytes(data[:100_000])  # simulate an interrupted download, truncated at 100 kB
    async with make_client(creds, moto_endpoint, download_chunk_bytes=16_384) as sc:
        await sc.put_bytes("blob.bin", data)
        s3 = await sc._client(creds)
        ranges: list[str] = []
        original = s3.get_object

        async def spy(**kwargs: Any) -> Any:
            if "Range" in kwargs:
                ranges.append(kwargs["Range"])
            return await original(**kwargs)

        s3.get_object = spy
        await sc.download_file(creds, "blob.bin", dest, expected_hash=hash_bytes(data))
    assert dest.read_bytes() == data
    assert not part.exists()
    assert ranges[0] == "bytes=100000-116383"  # resumed at the .part boundary, not byte 0


async def test_download_file_restarts_oversized_stale_part(admin: Any, moto_endpoint: str, tmp_path) -> None:
    creds = make_creds(fresh_bucket(admin, "stale"))
    data = os.urandom(50_000)
    dest = tmp_path / "blob.bin"
    (tmp_path / "blob.bin.part").write_bytes(b"z" * 60_000)  # larger than the object: restart
    async with make_client(creds, moto_endpoint, download_chunk_bytes=16_384) as sc:
        await sc.put_bytes("blob.bin", data)
        await sc.download_file(creds, "blob.bin", dest, expected_hash=hash_bytes(data))
    assert dest.read_bytes() == data


async def test_download_file_integrity_mismatch_discards_part(
    admin: Any, moto_endpoint: str, tmp_path
) -> None:
    creds = make_creds(fresh_bucket(admin, "poison"))
    dest = tmp_path / "blob.bin"
    async with make_client(creds, moto_endpoint) as sc:
        await sc.put_bytes("blob.bin", b"not what was committed")
        with pytest.raises(IntegrityError):
            await sc.download_file(creds, "blob.bin", dest, expected_hash="ff" * 32)
    assert not dest.exists()
    assert not os.path.exists(str(dest) + ".part")  # poisoned partial must not be resumable


# --------------------------------------------------------------------------- #
# HEAD metadata + listing
# --------------------------------------------------------------------------- #


async def test_object_timestamp_and_exists(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "ts"))
    async with make_client(creds, moto_endpoint) as sc:
        before = time.time() - 300
        await sc.put_bytes("gate/obj", b"payload")
        after = time.time() + 300
        ts = await sc.object_timestamp(creds, "gate/obj")
        assert before <= ts <= after
        assert await sc.object_exists(creds, "gate/obj")
        assert not await sc.object_exists(creds, "gate/absent")
        with pytest.raises(ObjectMissingError):
            await sc.object_timestamp(creds, "gate/absent")


async def test_list_keys_sorted_under_prefix(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "ls"))
    async with make_client(creds, moto_endpoint) as sc:
        for key in ("p/w2/b", "p/w2/a", "p/w1/c", "other/x"):
            await sc.put_bytes(key, b"1")
        assert await sc.list_keys(creds, "p/") == ["p/w1/c", "p/w2/a", "p/w2/b"]
        assert await sc.list_keys(creds, "nope/") == []


# --------------------------------------------------------------------------- #
# gather
# --------------------------------------------------------------------------- #


async def test_gather_uid_sorted_with_failures_recorded(admin: Any, moto_endpoint: str) -> None:
    """3 good peers (one unverified), one bad hash, one missing object. ok is
    uid-ascending; failures are recorded per uid, never dropped."""
    payloads = {3: b"peer-three", 7: b"peer-seven", 9: b"peer-nine"}
    peers: dict[int, BucketCreds] = {}
    for uid in (3, 5, 7, 9, 11):
        creds = make_creds(fresh_bucket(admin, f"peer{uid}"), access_key=f"key-{uid}")
        peers[uid] = creds
    for uid, blob in payloads.items():
        admin.put_object(Bucket=peers[uid].bucket_name, Key=f"w/uid{uid}.zst", Body=blob)
    admin.put_object(Bucket=peers[5].bucket_name, Key="w/uid5.zst", Body=b"tampered")
    # uid 11: bucket exists but object never uploaded

    expected_hashes = {uid: hash_bytes(blob) for uid, blob in payloads.items() if uid != 9}
    expected_hashes[5] = "aa" * 32  # wrong on purpose
    own = make_creds(fresh_bucket(admin, "self"), access_key="key-self")
    async with make_client(own, moto_endpoint) as sc:
        result = await sc.gather_bytes(
            peers,
            lambda uid: f"w/uid{uid}.zst",
            expected_hashes=expected_hashes,
            deadline_s=30.0,
            max_bytes=1 << 20,
        )
        # one cached client per distinct (endpoint, access key)
        assert len(sc._clients) == len(peers)

    assert isinstance(result, GatherResult)
    assert list(result.ok) == [3, 7, 9]  # deterministic uid-ascending order
    assert result.uids == [3, 7, 9]
    assert dict(result.ok) == payloads
    assert sorted(result.failed) == [5, 11]
    assert result.failed[5].startswith("integrity:")
    assert result.failed[11].startswith("missing:")


async def test_gather_per_fetch_timeout(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "slow"))
    admin.put_object(Bucket=creds.bucket_name, Key="w/uid1.zst", Body=b"data")
    own = make_creds(fresh_bucket(admin, "self2"))
    async with make_client(own, moto_endpoint) as sc:
        result = await sc.gather_bytes(
            {1: creds},
            lambda uid: f"w/uid{uid}.zst",
            expected_hashes={},
            deadline_s=0.0,  # expires before any fetch can complete
            max_bytes=1 << 20,
        )
    assert result.ok == {}
    assert result.failed == {1: "timeout"}


# --------------------------------------------------------------------------- #
# retry wrapper
# --------------------------------------------------------------------------- #


async def test_retry_recovers_from_transient_faults() -> None:
    from botocore.exceptions import EndpointConnectionError

    sc = StorageClient(make_creds("unused"), StorageConfig(), retry_base_delay_s=0.001)
    calls = 0

    async def flaky() -> int:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise EndpointConnectionError(endpoint_url="http://unreachable")
        return 42

    assert await sc._retry("flaky", flaky) == 42
    assert calls == 3


async def test_retry_gives_up_after_attempts() -> None:
    from botocore.exceptions import EndpointConnectionError

    sc = StorageClient(make_creds("unused"), StorageConfig(), retry_base_delay_s=0.001)
    calls = 0

    async def always_down() -> None:
        nonlocal calls
        calls += 1
        raise EndpointConnectionError(endpoint_url="http://unreachable")

    with pytest.raises(EndpointConnectionError):
        await sc._retry("down", always_down)
    assert calls == 3


async def test_retry_never_retries_protocol_errors() -> None:
    sc = StorageClient(make_creds("unused"), StorageConfig(), retry_base_delay_s=0.001)
    calls = 0

    async def corrupt() -> None:
        nonlocal calls
        calls += 1
        raise IntegrityError("hash mismatch")

    with pytest.raises(IntegrityError):
        await sc._retry("corrupt", corrupt)
    assert calls == 1


async def test_gather_ok_is_ordered_dict_even_when_completion_order_differs() -> None:
    """Structural determinism check: results are assembled from sorted uids, so
    even artificially reordered completions cannot change iteration order."""
    sc = StorageClient(make_creds("unused"), StorageConfig())
    delays = {2: 0.03, 4: 0.0, 8: 0.02}

    async def fake_get_bytes(bucket: BucketCreds, key: str, **_: Any) -> bytes:
        uid = int(key)
        await asyncio.sleep(delays[uid])
        return f"blob-{uid}".encode()

    sc.get_bytes = fake_get_bytes  # type: ignore[method-assign]
    peers = {uid: make_creds(f"bucket-{uid}") for uid in (8, 2, 4)}
    result = await sc.gather_bytes(
        peers, lambda uid: str(uid), expected_hashes={}, deadline_s=5.0, max_bytes=100
    )
    assert list(result.ok) == [2, 4, 8]
    assert result.failed == {}
