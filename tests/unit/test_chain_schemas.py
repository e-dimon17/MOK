"""Commitment wire formats (mok_core/chain/schemas.py): round-trips, golden
strings, byte-length bounds, garbage tolerance. Golden strings are consensus
constants — any change requires a SPEC_VERSION bump.

WIRE v2 (SPEC_VERSION 2): the live commitment pallet accepts one field of at
most 128 bytes (Raw0..Raw128). BucketCommit is 92 chars (creds as raw bytes,
bucket name derived from the hotkey); WindowCommit is 120 chars (window as 6
hex, payload hash bound by its 128-bit prefix, state_root/theta_end full).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mok_core.chain.schemas import (
    MAX_COMMITMENT_BYTES,
    WIRE_VERSION,
    BucketCommit,
    ManifestCommit,
    VoteCommit,
    WindowCommit,
    bucket_name_for_hotkey,
    decode_commitment,
)
from mok_core.config.schemas import BucketCreds

HOTKEY = "5FmoaEjSmmyvVqXbDqJAhDFeJpao517beWxBieMroU6VrnXu"
CREDS = BucketCreds(
    account_id="0123456789abcdef0123456789abcdef",
    bucket_name=bucket_name_for_hotkey(HOTKEY),
    access_key_id="fedcba9876543210fedcba9876543210",
    secret_access_key="00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
)

# consensus constant — change requires SPEC_VERSION bump
GOLDEN_BUCKET = "MOKB02ASNFZ4mrze8BI0VniavN7_7cuph2VDIQ_ty6mHZUMhAAESIzRFVmd4iZqrvM3e7_ABEiM0RVZneImaq7zN3u_w"
# consensus constant — change requires SPEC_VERSION bump
GOLDEN_WINDOW = (
    "MOKW02000042qqqqqqqqqqqqqqqqqqqqqgu7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7u7"
    "szMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMw"
)
# consensus constant — change requires SPEC_VERSION bump
GOLDEN_MANIFEST = "MOKM02" + "0d" * 32
# consensus constant — change requires SPEC_VERSION bump
GOLDEN_VOTE_ROLLBACK = "MOKV02R000000000911" + "ee" * 32
# consensus constant — change requires SPEC_VERSION bump
GOLDEN_VOTE_AMENDMENT = "MOKV02A000000000003" + "ff" * 32


def test_wire_version_is_2() -> None:
    assert WIRE_VERSION == 2
    assert MAX_COMMITMENT_BYTES == 128  # the live pallet's Raw0..Raw128 bound


class TestBucketName:
    def test_derived_from_hotkey_lowercase(self) -> None:
        assert bucket_name_for_hotkey(HOTKEY) == HOTKEY.lower()
        # valid S3/R2 bucket name: 3-63 chars, [a-z0-9-], alnum ends
        name = bucket_name_for_hotkey(HOTKEY)
        assert 3 <= len(name) <= 63 and name.isalnum() and name == name.lower()

    def test_rejects_non_ss58(self) -> None:
        for bad in ("", "local-3", "mok-miner", "0OIl" * 12):
            with pytest.raises(ValueError, match="ss58"):
                bucket_name_for_hotkey(bad)


class TestBucketCommit:
    def test_golden_string(self) -> None:
        assert BucketCommit(creds=CREDS).encode() == GOLDEN_BUCKET

    def test_byte_length(self) -> None:
        wire = BucketCommit(creds=CREDS).encode()
        assert len(wire.encode("utf-8")) == BucketCommit.WIRE_LEN == 92
        assert len(wire.encode("utf-8")) <= MAX_COMMITMENT_BYTES

    def test_round_trip_derives_bucket_name(self) -> None:
        wire = BucketCommit(creds=CREDS).encode()
        back = BucketCommit.decode(wire, hotkey_ss58=HOTKEY)
        assert back.creds == CREDS
        assert back.version == WIRE_VERSION
        # A different committer -> same creds bytes but ITS bucket name.
        other = "5DciMXcKCLk3yC98RR3wrDWWJunJVgboZmnQXvJpu9nqEQ2E"
        assert BucketCommit.decode(wire, hotkey_ss58=other).creds.bucket_name == other.lower()

    def test_non_hex_credentials_rejected(self) -> None:
        # R2 creds are hex; anything else cannot be packed as raw bytes.
        with pytest.raises(ValidationError, match="secret_access_key"):
            BucketCommit(creds=CREDS.model_copy(update={"secret_access_key": "s3cr3t" * 10 + "0000"}))
        with pytest.raises(ValidationError, match="account_id"):
            BucketCommit(creds=CREDS.model_copy(update={"account_id": "XYZ"}))

    def test_decode_garbage(self) -> None:
        for bad in ("", "MOKB02", GOLDEN_BUCKET[:-1], GOLDEN_BUCKET + "A", "MOKW02" + GOLDEN_BUCKET[6:],
                    GOLDEN_BUCKET[:6] + "!" * (len(GOLDEN_BUCKET) - 6)):
            with pytest.raises(ValueError):
                BucketCommit.decode(bad, hotkey_ss58=HOTKEY)
        # non-canonical base64url (same bytes, different trailing bits) is rejected
        tweaked = GOLDEN_BUCKET[:-1] + ("x" if GOLDEN_BUCKET[-1] != "x" else "y")
        with pytest.raises(ValueError):
            BucketCommit.decode(tweaked, hotkey_ss58=HOTKEY)


class TestWindowCommit:
    def commit(self, window: int = 0x42) -> WindowCommit:
        return WindowCommit(window=window, payload_hash="aa" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32)

    def test_golden_string(self) -> None:
        assert self.commit().encode() == GOLDEN_WINDOW

    def test_byte_length_under_128(self) -> None:
        wire = self.commit(16**6 - 1).encode()          # worst-case window
        assert len(wire.encode("utf-8")) == WindowCommit.WIRE_LEN == 120
        assert len(wire.encode("utf-8")) <= MAX_COMMITMENT_BYTES

    def test_round_trip_binds_payload_prefix(self) -> None:
        back = WindowCommit.decode(self.commit().encode())
        assert back.window == 0x42
        assert back.state_root == "bb" * 32 and back.theta_end_hash == "cc" * 32
        assert back.payload_hash == "aa" * 16                # 128-bit prefix travels
        assert back.payload_hash_prefix == "aa" * 16
        assert back.binds_payload_hash("aa" * 32)            # the real full hash
        assert back.binds_payload_hash("aa" * 16 + "ff" * 16)  # any hash with that prefix
        assert not back.binds_payload_hash("ab" * 32)
        assert self.commit().binds_payload_hash("aa" * 32)   # full-hash instances too

    def test_invalid_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            WindowCommit(window=16**6, payload_hash="aa" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32)
        with pytest.raises(ValidationError):
            WindowCommit(window=-1, payload_hash="aa" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32)
        with pytest.raises(ValidationError):
            WindowCommit(window=1, payload_hash="AA" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32)
        with pytest.raises(ValidationError):
            WindowCommit(window=1, payload_hash="aa" * 32, state_root="bb" * 31, theta_end_hash="cc" * 32)
        with pytest.raises(ValidationError):
            WindowCommit(window=1, payload_hash="aa" * 20, state_root="bb" * 32, theta_end_hash="cc" * 32)

    def test_decode_garbage(self) -> None:
        for bad in ("", "MOKW02", GOLDEN_WINDOW[:-1], GOLDEN_WINDOW + "A", "MOKB02" + GOLDEN_WINDOW[6:],
                    GOLDEN_WINDOW[:6] + "ZZZZZZ" + GOLDEN_WINDOW[12:]):
            with pytest.raises(ValueError):
                WindowCommit.decode(bad)


class TestManifestCommit:
    def test_golden_string(self) -> None:
        assert ManifestCommit(manifest_hash="0d" * 32).encode() == GOLDEN_MANIFEST

    def test_byte_length(self) -> None:
        assert len(GOLDEN_MANIFEST) == ManifestCommit.WIRE_LEN == 70 <= MAX_COMMITMENT_BYTES

    def test_round_trip(self) -> None:
        assert ManifestCommit.decode(GOLDEN_MANIFEST).manifest_hash == "0d" * 32

    def test_garbage(self) -> None:
        for bad in ("", "MOKM02", GOLDEN_MANIFEST[:-1], "MOKM02" + "0D" * 32):
            with pytest.raises(ValueError):
                ManifestCommit.decode(bad)


class TestVoteCommit:
    def test_golden_strings(self) -> None:
        assert VoteCommit(kind="rollback", target=911, payload_hash="ee" * 32).encode() == GOLDEN_VOTE_ROLLBACK
        assert VoteCommit(kind="amendment", target=3, payload_hash="ff" * 32).encode() == GOLDEN_VOTE_AMENDMENT

    def test_byte_length(self) -> None:
        assert len(GOLDEN_VOTE_ROLLBACK) == VoteCommit.WIRE_LEN == 83 <= MAX_COMMITMENT_BYTES

    def test_round_trip_both_kinds(self) -> None:
        for wire, kind, target in ((GOLDEN_VOTE_ROLLBACK, "rollback", 911), (GOLDEN_VOTE_AMENDMENT, "amendment", 3)):
            v = VoteCommit.decode(wire)
            assert (v.kind, v.target) == (kind, target)

    def test_unknown_kind_char(self) -> None:
        with pytest.raises(ValueError):
            VoteCommit.decode("MOKV02X000000000003" + "ff" * 32)

    def test_invalid_target(self) -> None:
        with pytest.raises(ValidationError):
            VoteCommit(kind="rollback", target=-1, payload_hash="ee" * 32)


class TestDecodeCommitment:
    def test_dispatch_all_kinds(self) -> None:
        assert isinstance(decode_commitment(GOLDEN_BUCKET, hotkey_ss58=HOTKEY), BucketCommit)
        assert isinstance(decode_commitment(GOLDEN_WINDOW), WindowCommit)
        assert isinstance(decode_commitment(GOLDEN_MANIFEST), ManifestCommit)
        assert isinstance(decode_commitment(GOLDEN_VOTE_ROLLBACK), VoteCommit)

    def test_bucket_requires_hotkey(self) -> None:
        with pytest.raises(ValueError, match="hotkey"):
            decode_commitment(GOLDEN_BUCKET)

    def test_garbage_raises(self) -> None:
        for bad in ("", "MOK", "XXXX02abc", "auditor.v1", "probe-two"):
            with pytest.raises(ValueError):
                decode_commitment(bad)

    def test_version_survives_round_trip(self) -> None:
        w = WindowCommit(version=7, window=1, payload_hash="aa" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32)
        assert WindowCommit.decode(w.encode()).version == 7
