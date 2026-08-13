"""On-chain commitment wire objects and their string codecs.

Commitment concat pattern: fixed-width fields joined into one string, with a
4-char kind tag + 2-digit version prefix so one commitment slot can carry
bucket credentials, window commits, manifest hashes, or votes.

Wire strings are CONSENSUS CONSTANTS: every byte position is fixed per
(tag, version). Golden-vector tests pin exact strings; any change requires
a SPEC_VERSION bump. All encodings are pure ASCII, at most
MAX_COMMITMENT_BYTES bytes.
"""

from __future__ import annotations

import re
from typing import ClassVar, Literal

from pydantic import model_validator

from mok_core.config.schemas import BucketCreds, FrozenModel

WIRE_VERSION = 1
MAX_COMMITMENT_BYTES = 256

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


# --------------------------------------------------------------------------- #
# Wire objects
# --------------------------------------------------------------------------- #


class BucketCommit(FrozenModel):
    """R2 read-credential commitment. Layout after the 6-char prefix:
    account_id(32) ‖ bucket_name(63) ‖ access_key_id(32) ‖ secret_access_key(64) = 197 chars."""

    version: int = WIRE_VERSION
    creds: BucketCreds

    WIRE_LEN: ClassVar[int] = 197

    @model_validator(mode="after")
    def _check(self) -> BucketCommit:
        _check_version(self.version)
        _pad(self.creds.account_id, _ACCOUNT_W, "account_id")
        _pad(self.creds.bucket_name, _BUCKET_W, "bucket_name")
        _pad(self.creds.access_key_id, _ACCESS_W, "access_key_id")
        _pad(self.creds.secret_access_key, _SECRET_W, "secret_access_key")
        return self

    def encode(self) -> str:
        return _finish(
            _prefix(TAG_BUCKET, self.version)
            + _pad(self.creds.account_id, _ACCOUNT_W, "account_id")
            + _pad(self.creds.bucket_name, _BUCKET_W, "bucket_name")
            + _pad(self.creds.access_key_id, _ACCESS_W, "access_key_id")
            + _pad(self.creds.secret_access_key, _SECRET_W, "secret_access_key")
        )

    @classmethod
    def decode(cls, wire: str) -> BucketCommit:
        version, body = _split_prefix(wire, TAG_BUCKET, cls.WIRE_LEN)
        a, b = _ACCOUNT_W, _ACCOUNT_W + _BUCKET_W
        c = b + _ACCESS_W
        creds = BucketCreds(
            account_id=_unpad(body[:a], "account_id"),
            bucket_name=_unpad(body[a:b], "bucket_name"),
            access_key_id=_unpad(body[b:c], "access_key_id"),
            secret_access_key=_unpad(body[c:], "secret_access_key"),
        )
        return cls(version=version, creds=creds)


class WindowCommit(FrozenModel):
    """Two-phase-commit phase 1: H(payload) ‖ state_root ‖ H(θ_end) pinned on-chain
    before the payload bytes are uploaded. Layout after prefix:
    window(12, zero-padded) ‖ payload_hash(64) ‖ state_root(64) ‖ theta_end_hash(64) = 210 chars."""

    version: int = WIRE_VERSION
    window: int
    payload_hash: str
    state_root: str
    theta_end_hash: str

    WIRE_LEN: ClassVar[int] = 210

    @model_validator(mode="after")
    def _check(self) -> WindowCommit:
        _check_version(self.version)
        if not 0 <= self.window < 10**_WINDOW_W:
            raise ValueError(f"window must be in [0, 10^{_WINDOW_W}), got {self.window}")
        _check_hex64(self.payload_hash, "payload_hash")
        _check_hex64(self.state_root, "state_root")
        _check_hex64(self.theta_end_hash, "theta_end_hash")
        return self

    def encode(self) -> str:
        return _finish(
            _prefix(TAG_WINDOW, self.version)
            + f"{self.window:0{_WINDOW_W}d}"
            + self.payload_hash
            + self.state_root
            + self.theta_end_hash
        )

    @classmethod
    def decode(cls, wire: str) -> WindowCommit:
        version, body = _split_prefix(wire, TAG_WINDOW, cls.WIRE_LEN)
        w = _WINDOW_W
        return cls(
            version=version,
            window=_parse_uint(body[:w], "window"),
            payload_hash=body[w : w + 64],
            state_root=body[w + 64 : w + 128],
            theta_end_hash=body[w + 128 :],
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


def decode_commitment(wire: str) -> Commitment:
    """Decode any MOK commitment string by its kind tag. Raises ValueError on garbage."""
    if not isinstance(wire, str) or len(wire) < 6:
        raise ValueError("commitment too short")
    decoder = _DECODERS.get(wire[:4])
    if decoder is None:
        raise ValueError(f"unknown commitment tag {wire[:4]!r}")
    return decoder.decode(wire)
