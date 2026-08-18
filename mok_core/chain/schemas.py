"""On-chain commitment wire objects and their string codecs.

Commitment concat pattern: fixed-width fields joined into one string, with a
4-char kind tag + 2-digit version prefix so one commitment slot can carry
bucket credentials, window commits, manifest hashes, or votes.

Wire strings are CONSENSUS CONSTANTS: every byte position is fixed per
(tag, version). Golden-vector tests pin exact strings; any change requires
a SPEC_VERSION bump. All encodings are pure ASCII, at most
MAX_COMMITMENT_BYTES bytes.

WIRE v2 (2026-08): the live subtensor commitment pallet accepts ONE field of
at most 128 bytes (``Raw0..Raw128``; the SDK sends ``Raw{len(data)}``). v1's
BucketCommit (197) and WindowCommit (210) could never be committed to a real
chain. v2 fits everything in <=128 bytes:
  * BucketCommit v2 — 92 B: prefix ‖ base64url(account_id ‖ access_key_id ‖
    secret_access_key as RAW bytes, 64 B). The bucket name is NOT on the wire:
    it is DERIVED as the committing hotkey's ss58 address lowercased
    (`bucket_name_for_hotkey`), which is a valid S3/R2 bucket name.
  * WindowCommit v2 — 120 B: prefix ‖ window(6 hex) ‖ base64url(payload_hash[:16])
    ‖ base64url(state_root) ‖ base64url(theta_end_hash). The on-chain payload
    hash binds its first 128 bits; the full 256-bit hash still travels in the
    leader-signed certificate and is verified in full on every fetch. state_root
    and theta_end_hash (the audit/catch-up bindings) stay full 256-bit.
v1 codecs are retained for decoding archived data only.
"""

from __future__ import annotations

import base64
import re
from typing import ClassVar, Literal

from pydantic import model_validator

from mok_core.config.schemas import BucketCreds, FrozenModel

WIRE_VERSION = 2
#: Hard limit of the live commitment pallet (one Raw0..Raw128 field).
MAX_COMMITMENT_BYTES = 128

TAG_BUCKET = "MOKB"
TAG_WINDOW = "MOKW"
TAG_MANIFEST = "MOKM"
TAG_VOTE = "MOKV"

_PAD = "~"                      # right-pad char; forbidden inside any padded field
_DIGITS = set("0123456789")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Fixed field widths (bytes) — never change within a wire version.
_ACCOUNT_W = 32
_BUCKET_W = 63
_ACCESS_W = 32
_SECRET_W = 64
_WINDOW_W = 12                  # zero-padded decimal, so window < 10**12

_VOTE_KINDS: dict[str, str] = {"rollback": "R", "amendment": "A"}
_VOTE_CHARS: dict[str, str] = {c: k for k, c in _VOTE_KINDS.items()}


# --------------------------------------------------------------------------- #
# Codec primitives
# --------------------------------------------------------------------------- #


def _check_version(version: int) -> None:
    if not 0 <= version <= 99:
        raise ValueError(f"wire version must be in [0, 99], got {version}")


def _check_hex64(value: str, name: str) -> None:
    if not _HEX64.fullmatch(value):
        raise ValueError(f"{name} must be 64 lowercase hex chars, got {value!r}")


def _pad(value: str, width: int, name: str) -> str:
    if not value.isascii():
        raise ValueError(f"{name} must be ASCII")
    if _PAD in value:
        raise ValueError(f"{name} must not contain the pad char {_PAD!r}")
    if not 0 < len(value) <= width:
        raise ValueError(f"{name} must be 1..{width} chars, got {len(value)}")
    return value.ljust(width, _PAD)


def _unpad(field: str, name: str) -> str:
    value = field.rstrip(_PAD)
    if not value or _PAD in value:
        raise ValueError(f"malformed padded field {name}")
    return value


def _parse_uint(field: str, name: str) -> int:
    if not field or any(c not in _DIGITS for c in field):
        raise ValueError(f"{name} must be decimal digits, got {field!r}")
    return int(field)


def _prefix(tag: str, version: int) -> str:
    _check_version(version)
    return f"{tag}{version:02d}"


def _split_prefix(wire: str, tag: str, expected_len: int) -> tuple[int, str]:
    if not isinstance(wire, str) or not wire.isascii():
        raise ValueError("commitment must be an ASCII string")
    if len(wire) != expected_len:
        raise ValueError(f"{tag} commitment must be {expected_len} chars, got {len(wire)}")
    if wire[:4] != tag:
        raise ValueError(f"not a {tag} commitment: {wire[:4]!r}")
    return _parse_uint(wire[4:6], "version"), wire[6:]


def _finish(wire: str) -> str:
    if len(wire.encode("utf-8")) > MAX_COMMITMENT_BYTES:
        raise ValueError(f"encoded commitment exceeds {MAX_COMMITMENT_BYTES} bytes")
    return wire


def _b64e(raw: bytes) -> str:
    """base64url without padding — the v2 binary carrier (4 chars per 3 bytes)."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(text: str, nbytes: int, name: str) -> bytes:
    if not text.isascii() or not re.fullmatch(r"[A-Za-z0-9_-]+", text):
        raise ValueError(f"{name} is not base64url")
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} is not base64url: {e}") from e
    if len(raw) != nbytes:
        raise ValueError(f"{name} decodes to {len(raw)} bytes, expected {nbytes}")
    if _b64e(raw) != text:  # canonical form only (no alternate encodings of the same bytes)
        raise ValueError(f"{name} is not canonical base64url")
    return raw


def _hex_bytes(value: str, nbytes: int, name: str) -> bytes:
    if not re.fullmatch(rf"[0-9a-f]{{{2 * nbytes}}}", value):
        raise ValueError(f"{name} must be {2 * nbytes} lowercase hex chars, got {value!r}")
    return bytes.fromhex(value)


_SS58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{46,50}$")


def bucket_name_for_hotkey(hotkey_ss58: str) -> str:
    """The v2 convention: a participant's bucket is named after its hotkey,
    lowercased (46-50 chars, [a-z0-9] — a valid S3/R2 bucket name). Peers derive
    it from the metagraph, so it need not travel on the wire."""
    if not _SS58_RE.fullmatch(hotkey_ss58):
        raise ValueError(f"not an ss58 hotkey: {hotkey_ss58!r}")
    return hotkey_ss58.lower()


def _b64_len(nbytes: int) -> int:
    return (4 * nbytes + 2) // 3


# --------------------------------------------------------------------------- #
# Wire objects
# --------------------------------------------------------------------------- #


_CREDS_RAW = 16 + 16 + 32                     # account_id ‖ access_key_id ‖ secret (raw)
_CREDS_B64 = _b64_len(_CREDS_RAW)             # 86 chars


class BucketCommit(FrozenModel):
    """R2 read-credential commitment (v2, 92 chars):
    prefix(6) ‖ base64url(account_id(16 B) ‖ access_key_id(16 B) ‖ secret_access_key(32 B)).

    Cloudflare R2 credentials are hex (account 32, key 32, secret 64 hex chars);
    they are packed as raw bytes. ``creds.bucket_name`` is NOT encoded — decode
    fills it from the committing hotkey (`bucket_name_for_hotkey`)."""

    version: int = WIRE_VERSION
    creds: BucketCreds

    WIRE_LEN: ClassVar[int] = 6 + _CREDS_B64  # 92

    @model_validator(mode="after")
    def _check(self) -> BucketCommit:
        _check_version(self.version)
        _hex_bytes(self.creds.account_id, 16, "account_id")
        _hex_bytes(self.creds.access_key_id, 16, "access_key_id")
        _hex_bytes(self.creds.secret_access_key, 32, "secret_access_key")
        if not self.creds.bucket_name:
            raise ValueError("bucket_name must be set (derived from the hotkey)")
        return self

    def encode(self) -> str:
        raw = (
            _hex_bytes(self.creds.account_id, 16, "account_id")
            + _hex_bytes(self.creds.access_key_id, 16, "access_key_id")
            + _hex_bytes(self.creds.secret_access_key, 32, "secret_access_key")
        )
        return _finish(_prefix(TAG_BUCKET, self.version) + _b64e(raw))

    @classmethod
    def decode(cls, wire: str, *, hotkey_ss58: str) -> BucketCommit:
        """Decode; the bucket name is derived from `hotkey_ss58` (the committer)."""
        version, body = _split_prefix(wire, TAG_BUCKET, cls.WIRE_LEN)
        raw = _b64d(body, _CREDS_RAW, "creds")
        creds = BucketCreds(
            account_id=raw[:16].hex(),
            bucket_name=bucket_name_for_hotkey(hotkey_ss58),
            access_key_id=raw[16:32].hex(),
            secret_access_key=raw[32:].hex(),
        )
        return cls(version=version, creds=creds)


_WINDOW_HEX = 6                               # window < 16**6 = 16.7M windows
_PAYLOAD_PREFIX_BYTES = 16                    # on-chain binding of H(payload): first 128 bits
_HASH_B64 = _b64_len(32)                      # 43 chars
_PPFX_B64 = _b64_len(_PAYLOAD_PREFIX_BYTES)   # 22 chars


class WindowCommit(FrozenModel):
    """Two-phase-commit phase 1 (v2, 120 chars): prefix(6) ‖ window(6 hex) ‖
    base64url(payload_hash[:16]) ‖ base64url(state_root) ‖ base64url(theta_end_hash).

    `payload_hash` may be given in full (64 hex) or as its 32-hex prefix; only the
    first 16 bytes are bound on-chain (`payload_hash_prefix`). The full hash is
    carried in the leader-signed WindowCertificate and verified on every fetch.
    state_root and theta_end_hash — the catch-up and audit bindings — are full."""

    version: int = WIRE_VERSION
    window: int
    payload_hash: str
    state_root: str
    theta_end_hash: str

    WIRE_LEN: ClassVar[int] = 6 + _WINDOW_HEX + _PPFX_B64 + 2 * _HASH_B64  # 120

    @model_validator(mode="after")
    def _check(self) -> WindowCommit:
        _check_version(self.version)
        if not 0 <= self.window < 16**_WINDOW_HEX:
            raise ValueError(f"window must be in [0, 16^{_WINDOW_HEX}), got {self.window}")
        if not re.fullmatch(r"[0-9a-f]{32}([0-9a-f]{32})?", self.payload_hash):
            raise ValueError("payload_hash must be 32 or 64 lowercase hex chars")
        _check_hex64(self.state_root, "state_root")
        _check_hex64(self.theta_end_hash, "theta_end_hash")
        return self

    @property
    def payload_hash_prefix(self) -> str:
        """The 32-hex-char (128-bit) prefix that is bound on-chain."""
        return self.payload_hash[: 2 * _PAYLOAD_PREFIX_BYTES]

    def binds_payload_hash(self, full_hex64: str) -> bool:
        return full_hex64.lower().startswith(self.payload_hash_prefix)

    def encode(self) -> str:
        return _finish(
            _prefix(TAG_WINDOW, self.version)
            + f"{self.window:0{_WINDOW_HEX}x}"
            + _b64e(bytes.fromhex(self.payload_hash_prefix))
            + _b64e(bytes.fromhex(self.state_root))
            + _b64e(bytes.fromhex(self.theta_end_hash))
        )

    @classmethod
    def decode(cls, wire: str) -> WindowCommit:
        version, body = _split_prefix(wire, TAG_WINDOW, cls.WIRE_LEN)
        w = _WINDOW_HEX
        if not re.fullmatch(rf"[0-9a-f]{{{w}}}", body[:w]):
            raise ValueError(f"window must be {w} lowercase hex chars")
        a = w + _PPFX_B64
        b = a + _HASH_B64
        return cls(
            version=version,
            window=int(body[:w], 16),
            payload_hash=_b64d(body[w:a], _PAYLOAD_PREFIX_BYTES, "payload_hash").hex(),
            state_root=_b64d(body[a:b], 32, "state_root").hex(),
            theta_end_hash=_b64d(body[b:], 32, "theta_end_hash").hex(),
        )


class ManifestCommit(FrozenModel):
    """Owner's commitment of the canonical RunManifest hash. Layout: manifest_hash(64) = 70 chars."""

    version: int = WIRE_VERSION
    manifest_hash: str

    WIRE_LEN: ClassVar[int] = 70

    @model_validator(mode="after")
    def _check(self) -> ManifestCommit:
        _check_version(self.version)
        _check_hex64(self.manifest_hash, "manifest_hash")
        return self

    def encode(self) -> str:
        return _finish(_prefix(TAG_MANIFEST, self.version) + self.manifest_hash)

    @classmethod
    def decode(cls, wire: str) -> ManifestCommit:
        version, body = _split_prefix(wire, TAG_MANIFEST, cls.WIRE_LEN)
        return cls(version=version, manifest_hash=body)


class VoteCommit(FrozenModel):
    """Stake-weighted vote for a rollback or manifest amendment. Layout after prefix:
    kind(1: R|A) ‖ target(12, zero-padded window/seq) ‖ payload_hash(64) = 83 chars."""

    version: int = WIRE_VERSION
    kind: Literal["rollback", "amendment"]
    target: int
    payload_hash: str

    WIRE_LEN: ClassVar[int] = 83

    @model_validator(mode="after")
    def _check(self) -> VoteCommit:
        _check_version(self.version)
        if not 0 <= self.target < 10**_WINDOW_W:
            raise ValueError(f"target must be in [0, 10^{_WINDOW_W}), got {self.target}")
        _check_hex64(self.payload_hash, "payload_hash")
        return self

    def encode(self) -> str:
        return _finish(
            _prefix(TAG_VOTE, self.version)
            + _VOTE_KINDS[self.kind]
            + f"{self.target:0{_WINDOW_W}d}"
            + self.payload_hash
        )

    @classmethod
    def decode(cls, wire: str) -> VoteCommit:
        version, body = _split_prefix(wire, TAG_VOTE, cls.WIRE_LEN)
        kind = _VOTE_CHARS.get(body[0])
        if kind is None:
            raise ValueError(f"unknown vote kind char {body[0]!r}")
        return cls(
            version=version,
            kind=kind,  # type: ignore[arg-type]
            target=_parse_uint(body[1 : 1 + _WINDOW_W], "target"),
            payload_hash=body[1 + _WINDOW_W :],
        )


Commitment = BucketCommit | WindowCommit | ManifestCommit | VoteCommit

_DECODERS: dict[str, type[BucketCommit] | type[WindowCommit] | type[ManifestCommit] | type[VoteCommit]] = {
    TAG_BUCKET: BucketCommit,
    TAG_WINDOW: WindowCommit,
    TAG_MANIFEST: ManifestCommit,
    TAG_VOTE: VoteCommit,
}


def decode_commitment(wire: str, *, hotkey_ss58: str | None = None) -> Commitment:
    """Decode any MOK commitment string by its kind tag. Raises ValueError on garbage.
    `hotkey_ss58` (the committer) is required to decode a BucketCommit — its bucket
    name is derived from it."""
    if not isinstance(wire, str) or len(wire) < 6:
        raise ValueError("commitment too short")
    tag = wire[:4]
    if tag == TAG_BUCKET:
        if hotkey_ss58 is None:
            raise ValueError("decoding a BucketCommit requires the committer's hotkey")
        return BucketCommit.decode(wire, hotkey_ss58=hotkey_ss58)
    decoder = _DECODERS.get(tag)
    if decoder is None:
        raise ValueError(f"unknown commitment tag {tag!r}")
    return decoder.decode(wire)
