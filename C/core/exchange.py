"""Protocol I/O — the two-phase commit, certified gather, and every bucket object.

This is the only module that couples the payload wire format to the transport
layers (``mok_core.storage.StorageClient`` + ``mok_core.chain.ChainClient``,
both injected). Key properties: the enforced two-phase commit order,
the certificate-exact gather with a leader-mirror retry, and the bounded
canonical ``MOKA`` aggregator wire format.

Ordering rule (protocol decision #2): ``put_window_payload`` commits
``H(payload) ‖ state_root ‖ H(θ_end)`` on-chain FIRST and uploads the payload
bytes only after the commit lands. A payload whose bytes exist without a chain
commit is unusable (validators reconstruct expected hashes from chain state),
so a crash between the phases fails safe.

Aggregator wire format v1 (consensus surface — change requires SPEC_VERSION bump):

    frame  := b"MOKA" | u8 wire_version=1 | zstd(body)       (zstd level 3, no
                                                             checksum, 1 thread)
    body   := u32le header_len | header_json | blobs
    header := canonical JSON {"window": int, "entries": [{"uid", "nbytes",
              "payload_hash"}...]} with uids strictly increasing
    blobs  := each uid's serialized WindowPayload frame, in entry order

Key namespace owned by this module (not in mok_core.storage.keys):
``debug/w{window:08d}/uid{uid:05d}.json`` — per-parameter debug slices for
sync scoring; consensus-adjacent (validators reconstruct it), golden-pinned
in tests.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import zstandard

from mok_core.chain.schemas import WindowCommit
from mok_core.chain.windows import is_in_gate
from mok_core.config.canonical import canonical_bytes
from mok_core.config.schemas import BucketCreds
from mok_core.determinism.hashing import hash_bytes
from mok_core.storage import (
    KeyFormatError,
    ObjectMissingError,
    StorageClient,
    StorageError,
    keys,
)
from mok_core.telemetry import get_logger

from .certificate import WindowCertificate
from .payload import PayloadError, WindowPayload, deserialize, serialize, validate_structure

__all__ = [
    "AggregatorObject",
    "CertifiedGather",
    "ExchangeError",
    "UploadReceipt",
    "debug_key",
    "gate_check",
    "gather_certified",
    "gather_from_aggregator",
    "get_aggregator_object",
    "get_certificate",
    "get_debug_slices",
    "get_telemetry",
    "list_audit_reports",
    "put_aggregator_object",
    "put_audit_report",
    "put_certificate",
    "put_debug_slices",
    "put_telemetry",
    "put_window_payload",
]

log = get_logger("core.exchange")

AGGREGATOR_MAGIC = b"MOKA"
AGGREGATOR_WIRE_VERSION = 1
_ZSTD_LEVEL = 3

# Bound for every small JSON object (certificates, telemetry, audits, debug slices).
DEFAULT_JSON_MAX_BYTES = 1 << 20

_I64 = (1 << 63) - 1
_AGG_ENTRY_KEYS = frozenset({"uid", "nbytes", "payload_hash"})
_AGG_HEADER_KEYS = frozenset({"entries", "window"})


class ExchangeError(RuntimeError):
    """A protocol object failed exchange-level validation or bounds."""


# --------------------------------------------------------------------------- #
# Two-phase payload publication
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UploadReceipt:
    """Proof-of-work record of one two-phase payload publication."""

    payload_hash: str  # hex blake2b-256 of the uploaded frame == on-chain commitment
    key: str           # object key inside the miner's own bucket
    committed: bool    # phase 1 (chain commit) landed before phase 2 (upload)


async def put_window_payload(
    storage: StorageClient,
    chain: Any,
    payload: WindowPayload,
    *,
    version: int,
) -> UploadReceipt:
    """Publish one window payload with the TWO-PHASE ORDER ENFORCED.

    Phase 1: ``chain.commit_window`` pins ``H(payload) ‖ state_root ‖ H(θ_end)``
    on-chain. Phase 2: the serialized bytes go to this node's own bucket under
    ``keys.payload_key``. If the chain commit raises, NO bytes are uploaded —
    the exception propagates and the bucket stays clean for this window.
    """
    data = serialize(payload)
    payload_hash = hash_bytes(data)
    key = keys.payload_key(payload.window, payload.uid, str(version))
    commit = WindowCommit(
        window=payload.window,
        payload_hash=payload_hash,
        state_root=payload.metadata.state_root,
        theta_end_hash=payload.metadata.theta_end_hash,
    )
    # Phase 1 — blocking chain call off the event loop. Raises on failure: no upload.
    await asyncio.to_thread(chain.commit_window, commit)
    # Phase 2 — bytes only after the on-chain hash exists.
    await storage.put_bytes(key, data)
    return UploadReceipt(payload_hash=payload_hash, key=key, committed=True)


# --------------------------------------------------------------------------- #
# Aggregator object (leader's merged mirror for retries and catch-up)
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class AggregatorObject:
    """Leader-published mirror: every certified peer's payload bytes in one object."""

    window: int
    payloads: dict[int, bytes] = field(default_factory=dict)

    def serialize(self) -> bytes:
        """Deterministic MOKA frame (uid-sorted entries, canonical header)."""
        entries: list[dict[str, object]] = []
        blobs: list[bytes] = []
        for uid in sorted(self.payloads):
            if not 0 <= int(uid) < keys.MAX_UID:
                raise ExchangeError(f"aggregator uid {uid} out of range")
            blob = bytes(self.payloads[uid])
            if not blob:
                raise ExchangeError(f"aggregator entry for uid {uid} is empty")
            entries.append({"uid": int(uid), "nbytes": len(blob), "payload_hash": hash_bytes(blob)})
            blobs.append(blob)
        header = {"window": int(self.window), "entries": entries}
        header_raw = canonical_bytes(header)
        body = len(header_raw).to_bytes(4, "little") + header_raw + b"".join(blobs)
        cctx = zstandard.ZstdCompressor(level=_ZSTD_LEVEL, write_checksum=False, threads=0)
        return AGGREGATOR_MAGIC + bytes([AGGREGATOR_WIRE_VERSION]) + cctx.compress(body)

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        *,
        max_bytes: int,
        max_decompressed_bytes: int | None = None,
    ) -> AggregatorObject:
        """Bounds-checked parse: every limit verified before any blob is copied.

        Mirrors ``payload.deserialize`` hardening — frame size cap, declared
        zstd content size cap, byte-canonical header, exact blob accounting —
        plus a per-entry blake2b check so a corrupted mirror is rejected here
        rather than poisoning the outer step.
        """
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        data = bytes(data)
        if len(data) > max_bytes:
            raise ExchangeError(f"aggregator {len(data)} bytes exceeds max_bytes {max_bytes}")
        if len(data) < 6:
            raise ExchangeError("aggregator frame too short")
        if data[:4] != AGGREGATOR_MAGIC:
            raise ExchangeError(f"bad aggregator magic {data[:4]!r}")
        if data[4] != AGGREGATOR_WIRE_VERSION:
            raise ExchangeError(f"unsupported aggregator wire version {data[4]}")

        budget = 4 * max_bytes if max_decompressed_bytes is None else max_decompressed_bytes
        frame = data[5:]
        try:
            content_size = zstandard.frame_content_size(frame)
        except zstandard.ZstdError as e:
            raise ExchangeError(f"invalid zstd frame: {e}") from e
        if content_size < 0 or content_size > budget:
            raise ExchangeError(f"declared content size {content_size} outside (0, {budget}]")
        try:
            body = zstandard.ZstdDecompressor().decompress(frame, max_output_size=budget)
        except zstandard.ZstdError as e:
            raise ExchangeError(f"zstd decompression failed: {e}") from e

        if len(body) < 4:
            raise ExchangeError("aggregator body too short for header length")
        header_len = int.from_bytes(body[:4], "little")
        if not 2 <= header_len <= len(body) - 4:
            raise ExchangeError(f"header length {header_len} out of range for body {len(body)}")
        header_raw = body[4 : 4 + header_len]
        try:
            header = json.loads(header_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ExchangeError(f"aggregator header is not valid JSON: {e}") from e
        if not isinstance(header, dict) or set(header) != _AGG_HEADER_KEYS:
            raise ExchangeError("aggregator header must be {'window', 'entries'}")
        if canonical_bytes(header) != header_raw:
            raise ExchangeError("aggregator header bytes are not in canonical form")

        window = header["window"]
        if type(window) is not int or not 0 <= window < keys.MAX_WINDOW:
            raise ExchangeError(f"aggregator window {window!r} out of range")
        entries = header["entries"]
        if not isinstance(entries, list):
            raise ExchangeError("aggregator entries must be a list")

        parsed: list[tuple[int, int, str]] = []
        total_blob = 0
        prev_uid = -1
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != _AGG_ENTRY_KEYS:
                raise ExchangeError("aggregator entry must be {'uid', 'nbytes', 'payload_hash'}")
            uid, nbytes, ph = entry["uid"], entry["nbytes"], entry["payload_hash"]
            if type(uid) is not int or not 0 <= uid < keys.MAX_UID:
                raise ExchangeError(f"aggregator entry uid {uid!r} out of range")
            if uid <= prev_uid:
                raise ExchangeError("aggregator entry uids must be strictly increasing")
            prev_uid = uid
            if type(nbytes) is not int or not 1 <= nbytes <= _I64:
                raise ExchangeError(f"aggregator entry nbytes {nbytes!r} out of range")
            if type(ph) is not str or len(ph) != 64:
                raise ExchangeError("aggregator entry payload_hash must be a 64-char hex string")
            parsed.append((uid, nbytes, ph))
            total_blob += nbytes

        if len(body) != 4 + header_len + total_blob:
            raise ExchangeError(
                f"aggregator body {len(body)} != header ({4 + header_len}) + declared blobs ({total_blob})"
            )

        payloads: dict[int, bytes] = {}
        off = 4 + header_len
        for uid, nbytes, ph in parsed:
            blob = body[off : off + nbytes]
            off += nbytes
            if hash_bytes(blob) != ph:
                raise ExchangeError(f"aggregator entry uid {uid}: blob hash mismatch")
            payloads[uid] = blob
        return cls(window=window, payloads=payloads)


async def put_aggregator_object(
    storage: StorageClient,
    window: int,
    payloads: Mapping[int, bytes],
) -> str:
    """Leader duty: publish the merged uid -> payload-bytes mirror for `window`."""
    key = keys.aggregator_key(window)
    await storage.put_bytes(key, AggregatorObject(window=window, payloads=dict(payloads)).serialize())
    return key


async def get_aggregator_object(
    storage: StorageClient,
    bucket: BucketCreds,
    window: int,
    *,
    max_bytes: int,
) -> AggregatorObject:
    """Fetch + bounds-checked parse of a peer's aggregator mirror for `window`."""
    data = await storage.get_bytes(bucket, keys.aggregator_key(window), max_bytes=max_bytes)
    obj = AggregatorObject.deserialize(data, max_bytes=max_bytes)
    if obj.window != window:
        raise ExchangeError(f"aggregator object window {obj.window} != requested {window}")
    return obj


# --------------------------------------------------------------------------- #
# Certified gather
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CertifiedGather:
    """The exact certificate peer set, fetched and validated. ``payloads`` is
    uid-ascending; every certified uid appears in exactly one of the two dicts.
    The certificate SHOULD make ``missing`` empty — window_runner treats any
    entry as fatal for lockstep and enters catch-up."""

    payloads: OrderedDict[int, WindowPayload]
    missing: dict[int, str] = field(default_factory=dict)

    @property
    def uids(self) -> list[int]:
        return list(self.payloads)


def _aggregator_budget(max_bytes: int, n_peers: int) -> int:
    return max_bytes * max(1, n_peers) + (1 << 16)


def _decode_certified(
    cert: WindowCertificate,
    raw: Mapping[int, bytes],
    failures: Mapping[int, str],
    *,
    expected_param_shapes: Mapping[str, tuple[int, ...]],
    expected_dense: Mapping[str, tuple[int, ...]] | set[str] | frozenset[str],
    topk: int,
    target_chunk: int,
    max_bytes: int,
) -> CertifiedGather:
    """Deserialize + structurally validate fetched bytes for every certified uid."""
    payloads: OrderedDict[int, WindowPayload] = OrderedDict()
    missing: dict[int, str] = {}
    for uid in sorted(set(cert.included_uids)):
        reason = failures.get(uid)
        if reason is not None:
            missing[uid] = reason
            continue
        try:
            p = deserialize(raw[uid], max_bytes=max_bytes)
            if p.uid != uid:
                raise PayloadError(f"payload uid {p.uid} != certified uid {uid}")
            if p.window != cert.window:
                raise PayloadError(f"payload window {p.window} != certificate window {cert.window}")
            validate_structure(
                p, expected_param_shapes, expected_dense, topk, target_chunk=target_chunk
            )
        except PayloadError as e:
            missing[uid] = f"invalid: {e}"
            continue
        payloads[uid] = p
    return CertifiedGather(payloads=payloads, missing=missing)


async def _mirror_retry(
    storage: StorageClient,
    cert: WindowCertificate,
    leader_bucket: BucketCreds,
    failures: dict[int, str],
    raw: dict[int, bytes],
    *,
    max_bytes: int,
) -> None:
    """One retry pass from the leader's aggregator mirror for every failed uid."""
    try:
        agg = await get_aggregator_object(
            storage,
            leader_bucket,
            cert.window,
            max_bytes=_aggregator_budget(max_bytes, len(cert.included_uids)),
        )
    except (TimeoutError, StorageError, ExchangeError) as e:
        log.warning("aggregator mirror unavailable", window=cert.window, error=str(e))
        for uid in list(failures):
            failures[uid] += f"; mirror: {type(e).__name__}: {e}"
        return
    for uid in sorted(failures):
        blob = agg.payloads.get(uid)
        if blob is None:
            failures[uid] += "; mirror: absent"
            continue
        if len(blob) > max_bytes:
            failures[uid] += f"; mirror: too_large {len(blob)} > {max_bytes}"
            continue
        if hash_bytes(blob) != cert.payload_hashes[uid].lower():
            failures[uid] += "; mirror: integrity mismatch"
            continue
        raw[uid] = blob
        del failures[uid]


async def gather_certified(
    storage: StorageClient,
    cert: WindowCertificate,
    peer_buckets: Mapping[int, BucketCreds],
    *,
    expected_param_shapes: Mapping[str, tuple[int, ...]],
    expected_dense: Mapping[str, tuple[int, ...]] | set[str] | frozenset[str],
    topk: int,
    version: int,
    deadline_s: float,
    max_bytes: int,
    target_chunk: int = 64,
    leader_bucket: BucketCreds | None = None,
) -> CertifiedGather:
    """Fetch EXACTLY ``cert.included_uids``, each verified against the certified hash.

    Per-peer failures (missing/timeout/integrity/no bucket) are retried once
    from the leader's aggregator mirror when ``leader_bucket`` is given. Every
    recovered or fetched frame is deserialized and structurally validated;
    anything still failing lands in ``missing`` with a prefix-classified reason.
    """
    uids = sorted(set(cert.included_uids))
    with_bucket = {uid: peer_buckets[uid] for uid in uids if uid in peer_buckets}
    failures: dict[int, str] = {uid: "no_bucket" for uid in uids if uid not in peer_buckets}

    raw: dict[int, bytes] = {}
    if with_bucket:
        result = await storage.gather_bytes(
            with_bucket,
            lambda uid: keys.payload_key(cert.window, uid, str(version)),
            expected_hashes={uid: cert.payload_hashes[uid] for uid in with_bucket},
            deadline_s=deadline_s,
            max_bytes=max_bytes,
        )
        raw.update(result.ok)
        failures.update(result.failed)

    if failures and leader_bucket is not None:
        await _mirror_retry(storage, cert, leader_bucket, failures, raw, max_bytes=max_bytes)

    return _decode_certified(
        cert,
        raw,
        failures,
        expected_param_shapes=expected_param_shapes,
        expected_dense=expected_dense,
        topk=topk,
        target_chunk=target_chunk,
        max_bytes=max_bytes,
    )


async def gather_from_aggregator(
    storage: StorageClient,
    cert: WindowCertificate,
    leader_bucket: BucketCreds,
    *,
    expected_param_shapes: Mapping[str, tuple[int, ...]],
    expected_dense: Mapping[str, tuple[int, ...]] | set[str] | frozenset[str],
    topk: int,
    max_bytes: int,
    target_chunk: int = 64,
) -> CertifiedGather:
    """Certificate-exact gather sourced ONLY from the leader's aggregator mirror.

    The catch-up path: peers age their per-window objects out, but the leader's
    mirror persists. Hashes still verify against the certificate, so the mirror
    is untrusted storage, not a trusted party.
    """
    raw: dict[int, bytes] = {}
    failures: dict[int, str] = {}
    uids = sorted(set(cert.included_uids))
    try:
        agg = await get_aggregator_object(
            storage, leader_bucket, cert.window, max_bytes=_aggregator_budget(max_bytes, len(uids))
        )
    except (TimeoutError, StorageError, ExchangeError) as e:
        failures = {uid: f"mirror: {type(e).__name__}: {e}" for uid in uids}
    else:
        for uid in uids:
            blob = agg.payloads.get(uid)
            if blob is None:
                failures[uid] = "mirror: absent"
            elif len(blob) > max_bytes:
                failures[uid] = f"mirror: too_large {len(blob)} > {max_bytes}"
            elif hash_bytes(blob) != cert.payload_hashes[uid].lower():
                failures[uid] = "mirror: integrity mismatch"
            else:
                raw[uid] = blob
    return _decode_certified(
        cert,
        raw,
        failures,
        expected_param_shapes=expected_param_shapes,
        expected_dense=expected_dense,
        topk=topk,
        target_chunk=target_chunk,
        max_bytes=max_bytes,
    )


# --------------------------------------------------------------------------- #
# Certificates
# --------------------------------------------------------------------------- #


async def put_certificate(storage: StorageClient, cert: WindowCertificate) -> str:
    """Leader duty: publish the signed window certificate as canonical JSON."""
    key = keys.certificate_key(cert.window)
    await storage.put_bytes(key, canonical_bytes(cert))
    return key


async def get_certificate(
    storage: StorageClient,
    bucket: BucketCreds,
    window: int,
    *,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> WindowCertificate:
    """Fetch + parse a peer's published certificate. Signature/consensus checks
    are the caller's job (``certificate.verify_certificate``)."""
    data = await storage.get_bytes(bucket, keys.certificate_key(window), max_bytes=max_bytes)
    try:
        cert = WindowCertificate.model_validate_json(data)
    except ValueError as e:
        raise ExchangeError(f"certificate for window {window} is malformed: {e}") from e
    if cert.window != window:
        raise ExchangeError(f"certificate window {cert.window} != requested {window}")
    return cert


# --------------------------------------------------------------------------- #
# Debug slices (sync scoring evidence)
# --------------------------------------------------------------------------- #


def debug_key(window: int, uid: int) -> str:
    """``debug/w{window:08d}/uid{uid:05d}.json`` — validated like mok_core.storage.keys."""
    if not 0 <= window < keys.MAX_WINDOW:
        raise KeyFormatError(f"window must be in [0, {keys.MAX_WINDOW}), got {window}")
    if not 0 <= uid < keys.MAX_UID:
        raise KeyFormatError(f"uid must be in [0, {keys.MAX_UID}), got {uid}")
    return f"debug/w{window:08d}/uid{uid:05d}.json"


async def put_debug_slices(
    storage: StorageClient,
    window: int,
    uid: int,
    named_params: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    elems: int = 2,
) -> str:
    """Upload ``{name: first-`elems` fp32 values}`` for validator sync scoring."""
    if elems < 1:
        raise ValueError(f"elems must be >= 1, got {elems}")
    pairs = named_params.items() if isinstance(named_params, Mapping) else named_params
    slices: dict[str, list[float]] = {}
    for name, tensor in sorted(pairs, key=lambda kv: kv[0]):
        head = tensor.detach().reshape(-1)[:elems].to(device="cpu", dtype=torch.float32)
        values = [float(v) for v in head.tolist()]
        if not all(math.isfinite(v) for v in values):
            raise ExchangeError(f"debug slice for {name!r} has non-finite values")
        slices[name] = values
    body = {"window": int(window), "uid": int(uid), "elems": int(elems), "slices": slices}
    key = debug_key(window, uid)
    await storage.put_bytes(key, canonical_bytes(body))
    return key


async def get_debug_slices(
    storage: StorageClient,
    bucket: BucketCreds,
    window: int,
    uid: int,
    *,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> dict[str, list[float]]:
    """Fetch a peer's debug slices; returns ``{name: [floats]}`` after strict checks."""
    data = await storage.get_bytes(bucket, debug_key(window, uid), max_bytes=max_bytes)
    obj = _load_json_object(data, f"debug slices w{window}/uid{uid}")
    if set(obj) != {"window", "uid", "elems", "slices"}:
        raise ExchangeError("debug slices object has wrong field set")
    if obj["window"] != window or obj["uid"] != uid:
        raise ExchangeError(
            f"debug slices claim w{obj['window']}/uid{obj['uid']}, expected w{window}/uid{uid}"
        )
    slices = obj["slices"]
    if not isinstance(slices, dict):
        raise ExchangeError("debug slices 'slices' must be an object")
    out: dict[str, list[float]] = {}
    for name, values in slices.items():
        if not isinstance(name, str) or not isinstance(values, list):
            raise ExchangeError("debug slices entries must map names to lists")
        floats: list[float] = []
        for v in values:
            if type(v) not in (int, float) or not math.isfinite(float(v)):
                raise ExchangeError(f"debug slice {name!r} has a non-finite entry")
            floats.append(float(v))
        out[name] = floats
    return out


# --------------------------------------------------------------------------- #
# Telemetry + audit reports (bounded JSON; outside the deterministic path)
# --------------------------------------------------------------------------- #


def _load_json_object(data: bytes, what: str) -> dict[str, Any]:
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ExchangeError(f"{what} is not valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ExchangeError(f"{what} must be a JSON object")
    return obj


def _bounded_canonical(obj: Mapping[str, Any], what: str, max_bytes: int) -> bytes:
    try:
        raw = canonical_bytes(dict(obj))
    except (TypeError, ValueError) as e:
        raise ExchangeError(f"{what} is not canonically serializable: {e}") from e
    if len(raw) > max_bytes:
        raise ExchangeError(f"{what} is {len(raw)} bytes, exceeds bound {max_bytes}")
    return raw


async def put_telemetry(
    storage: StorageClient,
    window: int,
    uid: int,
    payload: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> str:
    """Publish this node's per-window telemetry snapshot (bounded canonical JSON)."""
    key = keys.telemetry_key(window, uid)
    await storage.put_bytes(key, _bounded_canonical(payload, "telemetry", max_bytes))
    return key


async def get_telemetry(
    storage: StorageClient,
    bucket: BucketCreds,
    window: int,
    uid: int,
    *,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> dict[str, Any]:
    """Fetch a peer's telemetry snapshot; size-bounded, must be a JSON object."""
    data = await storage.get_bytes(bucket, keys.telemetry_key(window, uid), max_bytes=max_bytes)
    return _load_json_object(data, f"telemetry w{window}/uid{uid}")


async def put_audit_report(
    storage: StorageClient,
    report: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> str:
    """Publish an AuditReport-shaped dict as canonical JSON.

    Requires integer ``window``, ``auditor_uid``, ``miner_uid`` fields (they
    address the key); every other field — including the auditor's hotkey and
    signature — passes through untouched so the signed bytes stay stable.
    """
    ids: list[int] = []
    for field_name in ("window", "auditor_uid", "miner_uid"):
        v = report.get(field_name)
        if type(v) is not int:
            raise ExchangeError(f"audit report field {field_name!r} must be an integer, got {v!r}")
        ids.append(v)
    key = keys.audit_report_key(ids[0], ids[1], ids[2])
    await storage.put_bytes(key, _bounded_canonical(report, "audit report", max_bytes))
    return key


async def list_audit_reports(
    storage: StorageClient,
    bucket: BucketCreds,
    window: int,
    *,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> list[dict[str, Any]]:
    """All well-formed audit reports in `bucket` for `window`, key-sorted.

    Malformed keys/objects and reports whose embedded ids contradict their key
    are skipped (logged), never raised — one bad object must not block ingest.
    """
    reports: list[dict[str, Any]] = []
    for key in await storage.list_keys(bucket, keys.audit_prefix(window)):
        try:
            ref = keys.parse_audit_report_key(key)
        except KeyFormatError:
            continue
        try:
            obj = _load_json_object(
                await storage.get_bytes(bucket, key, max_bytes=max_bytes), key
            )
        except (TimeoutError, StorageError, ExchangeError) as e:
            log.warning("audit report skipped", key=key, error=str(e))
            continue
        if (
            obj.get("window") != ref.window
            or obj.get("auditor_uid") != ref.auditor_uid
            or obj.get("miner_uid") != ref.miner_uid
        ):
            log.warning("audit report ids contradict key", key=key)
            continue
        reports.append(obj)
    return reports


# --------------------------------------------------------------------------- #
# Upload gate
# --------------------------------------------------------------------------- #


async def gate_check(
    storage: StorageClient,
    bucket: BucketCreds,
    key: str,
    boundary_ts: float,
    grace_s: float,
) -> bool:
    """True iff the object exists and its storage timestamp lies inside the
    two-phase-commit upload gate ``[boundary_ts, boundary_ts + grace_s)``."""
    try:
        ts = await storage.object_timestamp(bucket, key)
    except ObjectMissingError:
        return False
    return is_in_gate(ts, boundary_ts, grace_s)
