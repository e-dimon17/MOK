"""Object-key layout for every artifact in a participant's R2 bucket.

These strings are consensus wire formats: gather, certificates, audits, and
timestamp gates all address peers' buckets by reconstructing these keys from
on-chain data, so every builder validates its inputs and every format is
golden-vector pinned in tests. Changing any layout requires a SPEC_VERSION
bump.

Layout:
    payloads/w{window:08d}/uid{uid:05d}-v{version}.zst
    checkpoints/w{window:08d}/{kind}
    shards/{hash16}.bin
    telemetry/w{window:08d}/uid{uid:05d}.json
    certificates/w{window:08d}.json
    aggregators/w{window:08d}.zst
    audits/w{window:08d}/auditor{auditor_uid:05d}-miner{miner_uid:05d}.json
    manifest.json
    attest/uid{uid:05d}-{nonce}.json
"""

from __future__ import annotations

import re
from typing import NamedTuple

MAX_WINDOW = 10**8  # w{:08d} — fixed width keeps keys lexicographically window-ordered
MAX_UID = 10**5  # uid{:05d} — Bittensor uids are < 4096; width leaves headroom

_VERSION_RE = re.compile(r"^[0-9A-Za-z._-]+$")
_KIND_RE = re.compile(r"^[0-9A-Za-z._-]+$")
_HASH16_RE = re.compile(r"^[0-9a-f]{16}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{8,64}$")

_PAYLOAD_KEY_RE = re.compile(
    r"^payloads/w(?P<window>\d{8})/uid(?P<uid>\d{5})-v(?P<version>[0-9A-Za-z._-]+)\.zst$"
)
_CHECKPOINT_KEY_RE = re.compile(r"^checkpoints/w(?P<window>\d{8})/(?P<kind>[0-9A-Za-z._-]+)$")
_SHARD_KEY_RE = re.compile(r"^shards/(?P<hash16>[0-9a-f]{16})\.bin$")
_TELEMETRY_KEY_RE = re.compile(r"^telemetry/w(?P<window>\d{8})/uid(?P<uid>\d{5})\.json$")
_CERTIFICATE_KEY_RE = re.compile(r"^certificates/w(?P<window>\d{8})\.json$")
_AGGREGATOR_KEY_RE = re.compile(r"^aggregators/w(?P<window>\d{8})\.zst$")
_AUDIT_REPORT_KEY_RE = re.compile(
    r"^audits/w(?P<window>\d{8})/auditor(?P<auditor_uid>\d{5})-miner(?P<miner_uid>\d{5})\.json$"
)
_ATTEST_KEY_RE = re.compile(r"^attest/uid(?P<uid>\d{5})-(?P<nonce>[0-9a-f]{8,64})\.json$")

MANIFEST_KEY = "manifest.json"


class KeyFormatError(ValueError):
    """A key string does not match its expected layout, or a component is out of range."""


def _check_window(window: int) -> int:
    if not 0 <= window < MAX_WINDOW:
        raise KeyFormatError(f"window must be in [0, {MAX_WINDOW}), got {window}")
    return window


def _check_uid(uid: int, name: str = "uid") -> int:
    if not 0 <= uid < MAX_UID:
        raise KeyFormatError(f"{name} must be in [0, {MAX_UID}), got {uid}")
    return uid


# --------------------------------------------------------------------------- #
# Parsed-component records
# --------------------------------------------------------------------------- #


class PayloadRef(NamedTuple):
    window: int
    uid: int
    version: str


class CheckpointRef(NamedTuple):
    window: int
    kind: str


class ShardRef(NamedTuple):
    hash16: str


class TelemetryRef(NamedTuple):
    window: int
    uid: int


class CertificateRef(NamedTuple):
    window: int


class AggregatorRef(NamedTuple):
    window: int


class AuditReportRef(NamedTuple):
    window: int
    auditor_uid: int
    miner_uid: int


class AttestRef(NamedTuple):
    uid: int
    nonce: str


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def payload_key(window: int, uid: int, version: str) -> str:
    """Compressed WindowPayload a miner uploads during the two-phase-commit gate."""
    _check_window(window)
    _check_uid(uid)
    if not _VERSION_RE.match(version):
        raise KeyFormatError(f"invalid version string {version!r}")
    return f"payloads/w{window:08d}/uid{uid:05d}-v{version}.zst"


def checkpoint_key(window: int, kind: str) -> str:
    """One object of a checkpoint published at a window boundary (e.g. 'meta', 'shard-00003')."""
    _check_window(window)
    if not _KIND_RE.match(kind):
        raise KeyFormatError(f"invalid checkpoint kind {kind!r}")
    return f"checkpoints/w{window:08d}/{kind}"


def shard_key(hash16: str) -> str:
    """Content-addressed dataset shard; hash16 = first 16 hex chars of the shard's blake2b."""
    if not _HASH16_RE.match(hash16):
        raise KeyFormatError(f"hash16 must be 16 lowercase hex chars, got {hash16!r}")
    return f"shards/{hash16}.bin"


def telemetry_key(window: int, uid: int) -> str:
    """Per-window telemetry snapshot (outside the deterministic path)."""
    _check_window(window)
    _check_uid(uid)
    return f"telemetry/w{window:08d}/uid{uid:05d}.json"


def certificate_key(window: int) -> str:
    """Leader-published window certificate fixing the included-peer set."""
    _check_window(window)
    return f"certificates/w{window:08d}.json"


def aggregator_key(window: int) -> str:
    """Leader-published merged-payload object for slow-peer catch-up."""
    _check_window(window)
    return f"aggregators/w{window:08d}.zst"


def audit_report_key(window: int, auditor_uid: int, miner_uid: int) -> str:
    """Signed replay-audit report by auditor_uid about miner_uid's window."""
    _check_window(window)
    _check_uid(auditor_uid, "auditor_uid")
    _check_uid(miner_uid, "miner_uid")
    return f"audits/w{window:08d}/auditor{auditor_uid:05d}-miner{miner_uid:05d}.json"


def manifest_key() -> str:
    """The run manifest (dataset Merkle root, phase table, amendments)."""
    return MANIFEST_KEY


def attest_key(uid: int, nonce: str) -> str:
    """Hardware-attestation challenge response; nonce is the challenge's hex tag."""
    _check_uid(uid)
    if not _NONCE_RE.match(nonce):
        raise KeyFormatError(f"nonce must be 8..64 lowercase hex chars, got {nonce!r}")
    return f"attest/uid{uid:05d}-{nonce}.json"


# --------------------------------------------------------------------------- #
# Prefixes (for list_keys)
# --------------------------------------------------------------------------- #


def payload_prefix(window: int) -> str:
    _check_window(window)
    return f"payloads/w{window:08d}/"


def telemetry_prefix(window: int) -> str:
    _check_window(window)
    return f"telemetry/w{window:08d}/"


def audit_prefix(window: int) -> str:
    _check_window(window)
    return f"audits/w{window:08d}/"


# --------------------------------------------------------------------------- #
# Parsers (key -> components; KeyFormatError on mismatch)
# --------------------------------------------------------------------------- #


def _match(pattern: re.Pattern[str], key: str, what: str) -> re.Match[str]:
    m = pattern.match(key)
    if m is None:
        raise KeyFormatError(f"not a {what} key: {key!r}")
    return m


def parse_payload_key(key: str) -> PayloadRef:
    m = _match(_PAYLOAD_KEY_RE, key, "payload")
    return PayloadRef(window=int(m["window"]), uid=int(m["uid"]), version=m["version"])


def parse_checkpoint_key(key: str) -> CheckpointRef:
    m = _match(_CHECKPOINT_KEY_RE, key, "checkpoint")
    return CheckpointRef(window=int(m["window"]), kind=m["kind"])


def parse_shard_key(key: str) -> ShardRef:
    m = _match(_SHARD_KEY_RE, key, "shard")
    return ShardRef(hash16=m["hash16"])


def parse_telemetry_key(key: str) -> TelemetryRef:
    m = _match(_TELEMETRY_KEY_RE, key, "telemetry")
    return TelemetryRef(window=int(m["window"]), uid=int(m["uid"]))


def parse_certificate_key(key: str) -> CertificateRef:
    m = _match(_CERTIFICATE_KEY_RE, key, "certificate")
    return CertificateRef(window=int(m["window"]))


def parse_aggregator_key(key: str) -> AggregatorRef:
    m = _match(_AGGREGATOR_KEY_RE, key, "aggregator")
    return AggregatorRef(window=int(m["window"]))


def parse_audit_report_key(key: str) -> AuditReportRef:
    m = _match(_AUDIT_REPORT_KEY_RE, key, "audit report")
    return AuditReportRef(
        window=int(m["window"]),
        auditor_uid=int(m["auditor_uid"]),
        miner_uid=int(m["miner_uid"]),
    )


def parse_attest_key(key: str) -> AttestRef:
    m = _match(_ATTEST_KEY_RE, key, "attest")
    return AttestRef(uid=int(m["uid"]), nonce=m["nonce"])
