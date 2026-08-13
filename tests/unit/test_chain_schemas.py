"""Commitment wire formats (mok_core/chain/schemas.py): round-trips, golden
strings, byte-length bounds, garbage tolerance. Golden strings are consensus
constants — any change requires a SPEC_VERSION bump."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mok_core.chain.schemas import (
    MAX_COMMITMENT_BYTES,
    BucketCommit,
    ManifestCommit,
    VoteCommit,
    WindowCommit,
    decode_commitment,
)
from mok_core.config.schemas import BucketCreds

CREDS = BucketCreds(
    account_id="0123456789abcdef0123456789abcdef",
    bucket_name="mok-miner-7",
    access_key_id="fedcba9876543210fedcba9876543210",
    secret_access_key="s3cr3t-key-for-golden-vector-test-0000000000000000000000000000",
)

# consensus constant — change requires SPEC_VERSION bump
GOLDEN_BUCKET = (
    "MOKB010123456789abcdef0123456789abcdefmok-miner-7"
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    "fedcba9876543210fedcba9876543210"
    "s3cr3t-key-for-golden-vector-test-0000000000000000000000000000~~"
)

# consensus constant — change requires SPEC_VERSION bump
GOLDEN_WINDOW = (
    "MOKW01000000000042"
    + "aa" * 32
    + "bb" * 32
    + "cc" * 32
)

# consensus constant — change requires SPEC_VERSION bump
GOLDEN_MANIFEST = "MOKM01" + "0d" * 32

# consensus constant — change requires SPEC_VERSION bump
GOLDEN_VOTE_ROLLBACK = "MOKV01R000000000911" + "ee" * 32
# consensus constant — change requires SPEC_VERSION bump
GOLDEN_VOTE_AMENDMENT = "MOKV01A000000000003" + "ff" * 32


class TestBucketCommit:
    def test_golden_string(self) -> None:
        assert BucketCommit(creds=CREDS).encode() == GOLDEN_BUCKET

    def test_byte_length(self) -> None:
        wire = BucketCommit(creds=CREDS).encode()
        assert len(wire.encode("utf-8")) == BucketCommit.WIRE_LEN == 197
        assert len(wire.encode("utf-8")) <= MAX_COMMITMENT_BYTES

    def test_round_trip(self) -> None:
        commit = BucketCommit(creds=CREDS)
        decoded = BucketCommit.decode(commit.encode())
        assert decoded == commit
        assert decoded.creds.bucket_name == "mok-miner-7"
        assert decoded.creds.secret_access_key == CREDS.secret_access_key

    def test_full_width_fields_round_trip(self) -> None:
        creds = BucketCreds(
            account_id="a" * 32,
            bucket_name="b" * 63,
            access_key_id="c" * 32,
            secret_access_key="d" * 64,
        )
        assert BucketCommit.decode(BucketCommit(creds=creds).encode()).creds == creds

    def test_oversized_field_rejected(self) -> None:
        creds = CREDS.model_copy(update={"account_id": "x" * 33})
        with pytest.raises(ValidationError, match="account_id"):
            BucketCommit(creds=creds)

    def test_pad_char_in_field_rejected(self) -> None:
        creds = CREDS.model_copy(update={"secret_access_key": "bad~secret"})
        with pytest.raises(ValidationError, match="pad char"):
            BucketCommit(creds=creds)

    def test_decode_garbage(self) -> None:
        with pytest.raises(ValueError, match="197"):
            BucketCommit.decode(GOLDEN_BUCKET[:-1])
        with pytest.raises(ValueError, match="not a MOKB"):
            BucketCommit.decode("XXXX" + GOLDEN_BUCKET[4:])


class TestWindowCommit:
    def _commit(self) -> WindowCommit:
        return WindowCommit(
            window=42, payload_hash="aa" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32
        )

    def test_golden_string(self) -> None:
        assert self._commit().encode() == GOLDEN_WINDOW

    def test_byte_length_under_256(self) -> None:
        wire = self._commit().encode()
        assert len(wire.encode("utf-8")) == WindowCommit.WIRE_LEN == 210
        assert len(wire.encode("utf-8")) <= MAX_COMMITMENT_BYTES
        # worst-case window index still fits
        biggest = WindowCommit(
            window=10**12 - 1, payload_hash="ff" * 32, state_root="ff" * 32, theta_end_hash="ff" * 32
        )
        assert len(biggest.encode().encode("utf-8")) <= MAX_COMMITMENT_BYTES

    def test_round_trip(self) -> None:
        commit = self._commit()
        decoded = WindowCommit.decode(commit.encode())
        assert decoded == commit
        assert decoded.window == 42

    def test_invalid_fields_rejected(self) -> None:
        with pytest.raises(ValidationError, match="window"):
            WindowCommit(window=-1, payload_hash="aa" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32)
        with pytest.raises(ValidationError, match="window"):
            WindowCommit(
                window=10**12, payload_hash="aa" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32
            )
        with pytest.raises(ValidationError, match="payload_hash"):
            WindowCommit(window=1, payload_hash="AA" * 32, state_root="bb" * 32, theta_end_hash="cc" * 32)
        with pytest.raises(ValidationError, match="state_root"):
            WindowCommit(window=1, payload_hash="aa" * 32, state_root="bb" * 31, theta_end_hash="cc" * 32)

    def test_decode_garbage(self) -> None:
        with pytest.raises(ValueError):
            WindowCommit.decode("")
        with pytest.raises(ValueError, match="window"):
            WindowCommit.decode("MOKW01" + "x" * 12 + "aa" * 32 + "bb" * 32 + "cc" * 32)
        with pytest.raises(ValueError):  # uppercase hex body
            WindowCommit.decode("MOKW01000000000042" + "AA" * 32 + "bb" * 32 + "cc" * 32)


class TestManifestCommit:
    def test_golden_string(self) -> None:
        assert ManifestCommit(manifest_hash="0d" * 32).encode() == GOLDEN_MANIFEST

    def test_byte_length(self) -> None:
        wire = ManifestCommit(manifest_hash="0d" * 32).encode()
        assert len(wire.encode("utf-8")) == ManifestCommit.WIRE_LEN == 70

    def test_round_trip(self) -> None:
        commit = ManifestCommit(manifest_hash="0d" * 32)
        assert ManifestCommit.decode(commit.encode()) == commit

    def test_garbage(self) -> None:
        with pytest.raises(ValidationError, match="manifest_hash"):
            ManifestCommit(manifest_hash="zz" * 32)
        with pytest.raises(ValueError, match="70"):
            ManifestCommit.decode(GOLDEN_MANIFEST + "00")


class TestVoteCommit:
    def test_golden_strings(self) -> None:
        rollback = VoteCommit(kind="rollback", target=911, payload_hash="ee" * 32)
        amendment = VoteCommit(kind="amendment", target=3, payload_hash="ff" * 32)
        assert rollback.encode() == GOLDEN_VOTE_ROLLBACK
        assert amendment.encode() == GOLDEN_VOTE_AMENDMENT

    def test_byte_length(self) -> None:
        wire = VoteCommit(kind="rollback", target=911, payload_hash="ee" * 32).encode()
        assert len(wire.encode("utf-8")) == VoteCommit.WIRE_LEN == 83

    def test_round_trip_both_kinds(self) -> None:
        for kind in ("rollback", "amendment"):
            commit = VoteCommit(kind=kind, target=17, payload_hash="12" * 32)
            assert VoteCommit.decode(commit.encode()) == commit

    def test_unknown_kind_char(self) -> None:
        with pytest.raises(ValueError, match="vote kind"):
            VoteCommit.decode("MOKV01X000000000911" + "ee" * 32)

    def test_invalid_target(self) -> None:
        with pytest.raises(ValidationError, match="target"):
            VoteCommit(kind="rollback", target=-1, payload_hash="ee" * 32)


class TestDecodeCommitment:
    def test_dispatch_all_kinds(self) -> None:
        assert isinstance(decode_commitment(GOLDEN_BUCKET), BucketCommit)
        assert isinstance(decode_commitment(GOLDEN_WINDOW), WindowCommit)
        assert isinstance(decode_commitment(GOLDEN_MANIFEST), ManifestCommit)
        assert isinstance(decode_commitment(GOLDEN_VOTE_ROLLBACK), VoteCommit)

    def test_garbage_raises(self) -> None:
        for garbage in ("", "MOK", "ZZZZ01" + "0" * 64, "hello world", "MOKB" + "not-a-version"):
            with pytest.raises(ValueError):
                decode_commitment(garbage)

    def test_version_survives_round_trip(self) -> None:
        commit = ManifestCommit(version=7, manifest_hash="0d" * 32)
        wire = commit.encode()
        assert wire.startswith("MOKM07")
        assert decode_commitment(wire) == commit
