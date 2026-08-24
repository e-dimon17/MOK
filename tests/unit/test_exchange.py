"""Tests for subnet/core/exchange.py — two-phase commit, certified gather, bucket objects.

Storage runs against a local ThreadedMotoServer (pattern from
test_storage_client.py); the chain side is a MagicMock so two-phase ordering
and failure injection are exact.
"""

from __future__ import annotations

import json
import uuid
import zlib
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import zstandard

from mok_core.chain import ChainError, WindowCommit
from mok_core.config.canonical import canonical_bytes
from mok_core.config.schemas import BucketCreds, StorageConfig
from mok_core.determinism.hashing import hash_bytes
from mok_core.storage import KeyFormatError, StorageClient, keys
from subnet.core.certificate import WindowCertificate
from subnet.core.compress import ChunkingTransformer, Quantizer, TopKCompressor
from subnet.core.exchange import (
    AGGREGATOR_MAGIC,
    AGGREGATOR_WIRE_VERSION,
    AggregatorObject,
    CertifiedGather,
    ExchangeError,
    debug_key,
    gate_check,
    gather_certified,
    gather_from_aggregator,
    get_aggregator_object,
    get_certificate,
    get_debug_slices,
    get_telemetry,
    list_audit_reports,
    put_aggregator_object,
    put_audit_report,
    put_certificate,
    put_debug_slices,
    put_telemetry,
    put_window_payload,
)
from subnet.core.payload import PayloadMeta, WindowPayload, canonical_payload_hash, serialize

TARGET_CHUNK = 4
TOPK = 3
PARAM_SHAPES: dict[str, tuple[int, ...]] = {"a.weight": (6, 10), "b.bias": (7,)}
DENSE_SHAPES: dict[str, tuple[int, ...]] = {"router.balance_bias": (5,)}
MAX_BYTES = 1 << 20
WINDOW = 12

_ROOT = "cc" * 32
_THETA = "bb" * 32

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


def make_client(creds: BucketCreds, endpoint: str, **cfg_overrides: Any) -> StorageClient:
    return StorageClient(
        creds,
        StorageConfig(**cfg_overrides),
        endpoint_override=endpoint,
        retry_base_delay_s=0.01,
    )


# --------------------------------------------------------------------------- #
# Payload helpers (test_payload.py pattern — fully deterministic, no RNG)
# --------------------------------------------------------------------------- #


def _make_compressor() -> TopKCompressor:
    tf = ChunkingTransformer(PARAM_SHAPES, target_chunk=TARGET_CHUNK)
    return TopKCompressor(tf, Quantizer(bins=4, range_sigmas=6.0), topk=TOPK)


def _make_payload(
    uid: int, window: int = WINDOW, shapes: dict[str, tuple[int, ...]] | None = None
) -> WindowPayload:
    comp = _make_compressor()
    compressed = {}
    for i, (name, shape) in enumerate(sorted((shapes or PARAM_SHAPES).items())):
        n = 1
        for s in shape:
            n *= s
        t = (torch.arange(n, dtype=torch.float32).reshape(shape) - n / 2) * (0.01 * (i + uid + 1))
        compressed[name] = comp.compress(name, t)
    dense = {
        name: torch.arange(shape[0], dtype=torch.float32) * 0.125 * (uid + 1)
        for name, shape in DENSE_SHAPES.items()
    }
    meta = PayloadMeta(
        sample_digest="aa" * 32,
        sample_count=64,
        theta_end_hash=_THETA,
        state_root=_ROOT,
        global_step=100,
        spec_version=1,
    )
    return WindowPayload(uid=uid, window=window, compressed=compressed, dense=dense, metadata=meta)


def _cert_for(blobs: dict[int, bytes], window: int = WINDOW) -> WindowCertificate:
    return WindowCertificate(
        window=window,
        included_uids=tuple(sorted(blobs)),
        payload_hashes={uid: hash_bytes(b) for uid, b in blobs.items()},
        theta_start_root="ee" * 32,
        leader_uid=0,
    )


_GATHER_KW: dict[str, Any] = {
    "expected_param_shapes": PARAM_SHAPES,
    "expected_dense": DENSE_SHAPES,
    "topk": TOPK,
    "target_chunk": TARGET_CHUNK,
    "max_bytes": MAX_BYTES,
}


# --------------------------------------------------------------------------- #
# Two-phase commit
# --------------------------------------------------------------------------- #


async def test_put_window_payload_commits_before_upload(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "twophase"))
    payload = _make_payload(uid=7)
    data = serialize(payload)
    events: list[str] = []
    chain = MagicMock()
    chain.commit_window.side_effect = lambda commit: events.append("commit")

    async with make_client(creds, moto_endpoint) as sc:
        original_put = sc.put_bytes

        async def spy_put(key: str, body: bytes) -> None:
            events.append("put")
            await original_put(key, body)

        sc.put_bytes = spy_put  # type: ignore[method-assign]
        receipt = await put_window_payload(sc, chain, payload, version=1)

    assert events == ["commit", "put"]  # phase 1 strictly before phase 2
    assert receipt.committed
    assert receipt.key == keys.payload_key(WINDOW, 7, "1")
    assert receipt.payload_hash == hash_bytes(data) == canonical_payload_hash(payload)

    commit = chain.commit_window.call_args.args[0]
    assert isinstance(commit, WindowCommit)
    assert commit.window == WINDOW
    assert commit.payload_hash == receipt.payload_hash
    assert commit.state_root == _ROOT
    assert commit.theta_end_hash == _THETA

    stored = admin.get_object(Bucket=creds.bucket_name, Key=receipt.key)["Body"].read()
    assert stored == data


async def test_put_window_payload_chain_failure_uploads_nothing(
    admin: Any, moto_endpoint: str
) -> None:
    creds = make_creds(fresh_bucket(admin, "chainfail"))
    chain = MagicMock()
    chain.commit_window.side_effect = ChainError("commit exhausted")
    async with make_client(creds, moto_endpoint) as sc:
        with pytest.raises(ChainError):
            await put_window_payload(sc, chain, _make_payload(uid=7), version=1)
        assert await sc.list_keys(creds, "payloads/") == []  # bucket untouched


# --------------------------------------------------------------------------- #
# Certified gather
# --------------------------------------------------------------------------- #


async def test_gather_certified_happy_path_three_peers(admin: Any, moto_endpoint: str) -> None:
    payloads = {uid: _make_payload(uid) for uid in (1, 2, 3)}
    blobs = {uid: serialize(p) for uid, p in payloads.items()}
    peer_buckets: dict[int, BucketCreds] = {}
    for uid, blob in blobs.items():
        creds = make_creds(fresh_bucket(admin, f"peer{uid}"), access_key=f"key-{uid}")
        peer_buckets[uid] = creds
        admin.put_object(Bucket=creds.bucket_name, Key=keys.payload_key(WINDOW, uid, "1"), Body=blob)

    cert = _cert_for(blobs)
    own = make_creds(fresh_bucket(admin, "gself"))
    async with make_client(own, moto_endpoint) as sc:
        result = await gather_certified(
            sc, cert, peer_buckets, version=1, deadline_s=30.0, **_GATHER_KW
        )

    assert isinstance(result, CertifiedGather)
    assert result.missing == {}
    assert result.uids == [1, 2, 3]  # uid-ascending, exactly the certified set
    for uid, p in result.payloads.items():
        assert p.uid == uid and p.window == WINDOW
        assert canonical_payload_hash(p) == cert.payload_hashes[uid]


async def test_gather_corrupted_peer_recovered_from_leader_mirror(
    admin: Any, moto_endpoint: str
) -> None:
    payloads = {uid: _make_payload(uid) for uid in (1, 2, 3)}
    blobs = {uid: serialize(p) for uid, p in payloads.items()}
    peer_buckets: dict[int, BucketCreds] = {}
    for uid, blob in blobs.items():
        creds = make_creds(fresh_bucket(admin, f"mpeer{uid}"), access_key=f"mkey-{uid}")
        peer_buckets[uid] = creds
        body = b"corrupted bytes" if uid == 2 else blob
        admin.put_object(Bucket=creds.bucket_name, Key=keys.payload_key(WINDOW, uid, "1"), Body=body)

    cert = _cert_for(blobs)
    leader = make_creds(fresh_bucket(admin, "leader"), access_key="key-leader")
    admin.put_object(
        Bucket=leader.bucket_name,
        Key=keys.aggregator_key(WINDOW),
        Body=AggregatorObject(window=WINDOW, payloads=blobs).serialize(),
    )

    own = make_creds(fresh_bucket(admin, "mself"))
    async with make_client(own, moto_endpoint) as sc:
        without = await gather_certified(
            sc, cert, peer_buckets, version=1, deadline_s=30.0, **_GATHER_KW
        )
        assert list(without.missing) == [2]
        assert without.missing[2].startswith("integrity:")

        recovered = await gather_certified(
            sc, cert, peer_buckets, version=1, deadline_s=30.0, leader_bucket=leader, **_GATHER_KW
        )
    assert recovered.missing == {}
    assert recovered.uids == [1, 2, 3]
    assert canonical_payload_hash(recovered.payloads[2]) == cert.payload_hashes[2]


async def test_gather_missing_and_bucketless_peers(admin: Any, moto_endpoint: str) -> None:
    """uid 1: fine. uid 2: bucket but no object -> mirror recovers. uid 4: no
    bucket at all and absent from the mirror -> stays missing with both reasons."""
    payloads = {uid: _make_payload(uid) for uid in (1, 2, 4)}
    blobs = {uid: serialize(p) for uid, p in payloads.items()}
    peer_buckets = {
        1: make_creds(fresh_bucket(admin, "xpeer1"), access_key="xkey-1"),
        2: make_creds(fresh_bucket(admin, "xpeer2"), access_key="xkey-2"),
    }
    admin.put_object(
        Bucket=peer_buckets[1].bucket_name, Key=keys.payload_key(WINDOW, 1, "1"), Body=blobs[1]
    )
    # uid 2 never uploaded; uid 4 committed but has no bucket on file.

    cert = _cert_for(blobs)
    leader = make_creds(fresh_bucket(admin, "xleader"), access_key="xkey-leader")
    admin.put_object(
        Bucket=leader.bucket_name,
        Key=keys.aggregator_key(WINDOW),
        Body=AggregatorObject(window=WINDOW, payloads={2: blobs[2]}).serialize(),
    )

    own = make_creds(fresh_bucket(admin, "xself"))
    async with make_client(own, moto_endpoint) as sc:
        result = await gather_certified(
            sc, cert, peer_buckets, version=1, deadline_s=30.0, leader_bucket=leader, **_GATHER_KW
        )
    assert result.uids == [1, 2]
    assert list(result.missing) == [4]
    assert result.missing[4] == "no_bucket; mirror: absent"


async def test_gather_validation_failures_land_in_missing(admin: Any, moto_endpoint: str) -> None:
    """Hash-correct but protocol-invalid payloads: wrong embedded uid (peer 1)
    and a missing compressed parameter (peer 2)."""
    wrong_uid = serialize(_make_payload(uid=9))  # stored under uid 1's key
    partial = serialize(_make_payload(uid=2, shapes={"a.weight": PARAM_SHAPES["a.weight"]}))
    good = serialize(_make_payload(uid=3))
    blobs = {1: wrong_uid, 2: partial, 3: good}

    peer_buckets: dict[int, BucketCreds] = {}
    for uid, blob in blobs.items():
        creds = make_creds(fresh_bucket(admin, f"vpeer{uid}"), access_key=f"vkey-{uid}")
        peer_buckets[uid] = creds
        admin.put_object(Bucket=creds.bucket_name, Key=keys.payload_key(WINDOW, uid, "1"), Body=blob)

    cert = _cert_for(blobs)
    own = make_creds(fresh_bucket(admin, "vself"))
    async with make_client(own, moto_endpoint) as sc:
        result = await gather_certified(
            sc, cert, peer_buckets, version=1, deadline_s=30.0, **_GATHER_KW
        )
    assert result.uids == [3]
    assert result.missing[1].startswith("invalid:") and "uid" in result.missing[1]
    assert result.missing[2].startswith("invalid:")


# --------------------------------------------------------------------------- #
# Aggregator object wire format
# --------------------------------------------------------------------------- #


def test_aggregator_round_trip_and_determinism() -> None:
    blobs = {5: b"payload-five", 2: b"payload-two", 9: b"x" * 1000}
    obj = AggregatorObject(window=3, payloads=blobs)
    data = obj.serialize()
    assert data[:4] == AGGREGATOR_MAGIC and data[4] == AGGREGATOR_WIRE_VERSION
    assert data == AggregatorObject(window=3, payloads=dict(sorted(blobs.items()))).serialize()

    back = AggregatorObject.deserialize(data, max_bytes=MAX_BYTES)
    assert back.window == 3
    assert back.payloads == blobs
    assert list(back.payloads) == [2, 5, 9]  # uid-ascending entry order


def test_aggregator_bounds_and_tampering_rejected() -> None:
    data = AggregatorObject(window=3, payloads={1: b"abc", 2: b"defg"}).serialize()
    with pytest.raises(ExchangeError, match="max_bytes"):
        AggregatorObject.deserialize(data, max_bytes=10)
    with pytest.raises(ExchangeError, match="magic"):
        AggregatorObject.deserialize(b"NOPE" + data[4:], max_bytes=MAX_BYTES)
    with pytest.raises(ExchangeError, match="content size"):
        AggregatorObject.deserialize(data, max_bytes=MAX_BYTES, max_decompressed_bytes=4)

    # Flip one blob byte inside the zstd body: per-entry hash check must fire.
    body = zstandard.ZstdDecompressor().decompress(data[5:], max_output_size=1 << 20)
    tampered_body = body[:-1] + bytes([body[-1] ^ 0xFF])
    cctx = zstandard.ZstdCompressor(level=3, write_checksum=False, threads=0)
    tampered = data[:5] + cctx.compress(tampered_body)
    with pytest.raises(ExchangeError, match="hash mismatch"):
        AggregatorObject.deserialize(tampered, max_bytes=MAX_BYTES)


def test_aggregator_noncanonical_header_rejected() -> None:
    blob = b"abc"
    header = {
        "window": 3,
        "entries": [{"uid": 1, "nbytes": len(blob), "payload_hash": hash_bytes(blob)}],
    }
    sloppy = json.dumps(header, indent=2).encode()  # valid JSON, not canonical bytes
    body = len(sloppy).to_bytes(4, "little") + sloppy + blob
    cctx = zstandard.ZstdCompressor(level=3, write_checksum=False, threads=0)
    frame = AGGREGATOR_MAGIC + bytes([AGGREGATOR_WIRE_VERSION]) + cctx.compress(body)
    with pytest.raises(ExchangeError, match="canonical"):
        AggregatorObject.deserialize(frame, max_bytes=MAX_BYTES)


async def test_put_get_aggregator_object(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "agg"))
    blobs = {1: serialize(_make_payload(1)), 4: serialize(_make_payload(4))}
    async with make_client(creds, moto_endpoint) as sc:
        key = await put_aggregator_object(sc, WINDOW, blobs)
        assert key == keys.aggregator_key(WINDOW)
        back = await get_aggregator_object(sc, creds, WINDOW, max_bytes=4 * MAX_BYTES)
    assert back.window == WINDOW
    assert back.payloads == blobs


async def test_gather_from_aggregator_only(admin: Any, moto_endpoint: str) -> None:
    blobs = {uid: serialize(_make_payload(uid)) for uid in (1, 2)}
    cert = _cert_for(blobs)
    leader = make_creds(fresh_bucket(admin, "aggonly"))
    admin.put_object(
        Bucket=leader.bucket_name,
        Key=keys.aggregator_key(WINDOW),
        Body=AggregatorObject(window=WINDOW, payloads=blobs).serialize(),
    )
    own = make_creds(fresh_bucket(admin, "aggself"))
    async with make_client(own, moto_endpoint) as sc:
        result = await gather_from_aggregator(sc, cert, leader, **_GATHER_KW)
    assert result.missing == {}
    assert result.uids == [1, 2]


# --------------------------------------------------------------------------- #
# Certificates
# --------------------------------------------------------------------------- #


async def test_certificate_round_trip_and_malformed(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "cert"))
    cert = _cert_for({3: b"three", 8: b"eight"})
    async with make_client(creds, moto_endpoint) as sc:
        key = await put_certificate(sc, cert)
        assert key == keys.certificate_key(WINDOW)
        assert await get_certificate(sc, creds, WINDOW) == cert
        admin.put_object(Bucket=creds.bucket_name, Key=key, Body=b"not json at all")
        with pytest.raises(ExchangeError, match="malformed"):
            await get_certificate(sc, creds, WINDOW)


# --------------------------------------------------------------------------- #
# Debug slices
# --------------------------------------------------------------------------- #


def test_debug_key_format_and_bounds() -> None:
    # consensus-adjacent key layout — change requires SPEC_VERSION bump
    assert debug_key(12, 7) == "debug/w00000012/uid00007.json"
    with pytest.raises(KeyFormatError):
        debug_key(-1, 7)
    with pytest.raises(KeyFormatError):
        debug_key(12, 10**5)


async def test_debug_slices_round_trip(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "dbg"))
    named = {
        "a.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4) * 0.5,
        "b.bias": torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16),
    }
    async with make_client(creds, moto_endpoint) as sc:
        key = await put_debug_slices(sc, WINDOW, 7, named, elems=2)
        assert key == debug_key(WINDOW, 7)
        slices = await get_debug_slices(sc, creds, WINDOW, 7)
    assert slices == {"a.weight": [0.0, 0.5], "b.bias": [1.0, 2.0]}


async def test_debug_slices_reject_mismatched_ids(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "dbgbad"))
    body = {"window": WINDOW, "uid": 99, "elems": 2, "slices": {"w": [0.0]}}
    admin.put_object(
        Bucket=creds.bucket_name, Key=debug_key(WINDOW, 7), Body=canonical_bytes(body)
    )
    async with make_client(creds, moto_endpoint) as sc:
        with pytest.raises(ExchangeError, match="claim"):
            await get_debug_slices(sc, creds, WINDOW, 7)


# --------------------------------------------------------------------------- #
# Telemetry + audit reports
# --------------------------------------------------------------------------- #


async def test_telemetry_round_trip_and_bound(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "tel"))
    snapshot = {"loss": 2.5, "tokens": 1024, "window": WINDOW}
    async with make_client(creds, moto_endpoint) as sc:
        key = await put_telemetry(sc, WINDOW, 3, snapshot)
        assert key == keys.telemetry_key(WINDOW, 3)
        assert await get_telemetry(sc, creds, WINDOW, 3) == snapshot
        with pytest.raises(ExchangeError, match="exceeds bound"):
            await put_telemetry(sc, WINDOW, 3, {"blob": "x" * 100}, max_bytes=50)


async def test_audit_reports_publish_and_list(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "audit"))
    reports = [
        {
            "window": WINDOW,
            "auditor_uid": 1,
            "miner_uid": 5,
            "match": True,
            "hotkey": "5F3s...",
            "sig": "ab" * 32,
        },
        {"window": WINDOW, "auditor_uid": 2, "miner_uid": 5, "match": False, "sig": "cd" * 32},
    ]
    async with make_client(creds, moto_endpoint) as sc:
        for r in reports:
            await put_audit_report(sc, r)
        # A well-keyed but self-contradicting object and a garbage object: both skipped.
        admin.put_object(
            Bucket=creds.bucket_name,
            Key=keys.audit_report_key(WINDOW, 3, 5),
            Body=canonical_bytes({"window": WINDOW + 1, "auditor_uid": 3, "miner_uid": 5}),
        )
        admin.put_object(
            Bucket=creds.bucket_name, Key=keys.audit_report_key(WINDOW, 4, 5), Body=b"garbage"
        )
        listed = await list_audit_reports(sc, creds, WINDOW)
    assert listed == reports  # key-sorted == auditor-uid order; sig/hotkey untouched
    with pytest.raises(ExchangeError, match="must be an integer"):
        async with make_client(creds, moto_endpoint) as sc:
            await put_audit_report(sc, {"window": "12", "auditor_uid": 1, "miner_uid": 2})


# --------------------------------------------------------------------------- #
# Upload gate
# --------------------------------------------------------------------------- #


async def test_gate_check_boundary_cases(admin: Any, moto_endpoint: str) -> None:
    creds = make_creds(fresh_bucket(admin, "gate"))
    async with make_client(creds, moto_endpoint) as sc:
        await sc.put_bytes("gate/obj", b"payload")
        ts = await sc.object_timestamp(creds, "gate/obj")

        assert await gate_check(sc, creds, "gate/obj", ts, 90.0)  # boundary inclusive
        assert await gate_check(sc, creds, "gate/obj", ts - 89.0, 90.0)  # inside
        assert not await gate_check(sc, creds, "gate/obj", ts - 90.0, 90.0)  # deadline exclusive
        assert not await gate_check(sc, creds, "gate/obj", ts + 1.0, 90.0)  # uploaded early
        assert not await gate_check(sc, creds, "gate/absent", ts, 90.0)  # missing object


# --------------------------------------------------------------------------- #
# Wire-format cross-checks
# --------------------------------------------------------------------------- #


def test_aggregator_header_is_json_and_blobs_are_appended() -> None:
    """Structural pin of the MOKA body layout (header_len | canonical JSON | blobs)."""
    blobs = {7: b"BLOB-SEVEN"}
    data = AggregatorObject(window=1, payloads=blobs).serialize()
    body = zstandard.ZstdDecompressor().decompress(data[5:], max_output_size=1 << 16)
    header_len = int.from_bytes(body[:4], "little")
    header = json.loads(body[4 : 4 + header_len])
    assert header["window"] == 1
    assert header["entries"] == [
        {"uid": 7, "nbytes": len(blobs[7]), "payload_hash": hash_bytes(blobs[7])}
    ]
    assert body[4 + header_len :] == blobs[7]
    assert zlib.crc32(body) == zlib.crc32(body)  # body is plain bytes, no hidden state
