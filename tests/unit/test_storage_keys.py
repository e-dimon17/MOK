"""Golden-vector and round-trip tests for the object-key wire formats."""

from __future__ import annotations

import pytest

from mok_core.storage import keys


class TestGoldenVectors:
    """Pinned key strings. These are consensus wire formats — peers reconstruct
    them from on-chain data to address each other's buckets."""

    def test_payload_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.payload_key(3, 7, "1.0.0") == "payloads/w00000003/uid00007-v1.0.0.zst"

    def test_checkpoint_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.checkpoint_key(120, "meta") == "checkpoints/w00000120/meta"

    def test_shard_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.shard_key("00c0ffee00c0ffee") == "shards/00c0ffee00c0ffee.bin"

    def test_telemetry_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.telemetry_key(41, 250) == "telemetry/w00000041/uid00250.json"

    def test_certificate_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.certificate_key(999) == "certificates/w00000999.json"

    def test_aggregator_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.aggregator_key(999) == "aggregators/w00000999.zst"

    def test_audit_report_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.audit_report_key(55, 2, 17) == "audits/w00000055/auditor00002-miner00017.json"

    def test_manifest_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.manifest_key() == "manifest.json"

    def test_attest_key(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.attest_key(12, "deadbeef01") == "attest/uid00012-deadbeef01.json"

    def test_prefixes(self) -> None:
        # consensus constant — change requires SPEC_VERSION bump
        assert keys.payload_prefix(3) == "payloads/w00000003/"
        assert keys.telemetry_prefix(3) == "telemetry/w00000003/"
        assert keys.audit_prefix(3) == "audits/w00000003/"

    def test_prefixes_prefix_their_keys(self) -> None:
        assert keys.payload_key(3, 7, "1.0.0").startswith(keys.payload_prefix(3))
        assert keys.telemetry_key(3, 7).startswith(keys.telemetry_prefix(3))
        assert keys.audit_report_key(3, 1, 2).startswith(keys.audit_prefix(3))


class TestRoundTrip:
    def test_payload(self) -> None:
        ref = keys.parse_payload_key(keys.payload_key(12345, 4095, "0.2.1rc1"))
        assert ref == keys.PayloadRef(window=12345, uid=4095, version="0.2.1rc1")

    def test_checkpoint(self) -> None:
        ref = keys.parse_checkpoint_key(keys.checkpoint_key(10, "shard-00003.distcp"))
        assert ref == keys.CheckpointRef(window=10, kind="shard-00003.distcp")

    def test_shard(self) -> None:
        ref = keys.parse_shard_key(keys.shard_key("0123456789abcdef"))
        assert ref == keys.ShardRef(hash16="0123456789abcdef")

    def test_telemetry(self) -> None:
        assert keys.parse_telemetry_key(keys.telemetry_key(0, 0)) == keys.TelemetryRef(0, 0)

    def test_certificate(self) -> None:
        assert keys.parse_certificate_key(keys.certificate_key(77)) == keys.CertificateRef(77)

    def test_aggregator(self) -> None:
        assert keys.parse_aggregator_key(keys.aggregator_key(77)) == keys.AggregatorRef(77)

    def test_audit_report(self) -> None:
        ref = keys.parse_audit_report_key(keys.audit_report_key(9, 1, 99999))
        assert ref == keys.AuditReportRef(window=9, auditor_uid=1, miner_uid=99999)

    def test_attest(self) -> None:
        ref = keys.parse_attest_key(keys.attest_key(31, "ab" * 32))
        assert ref == keys.AttestRef(uid=31, nonce="ab" * 32)

    def test_window_ordering_is_lexicographic(self) -> None:
        """Fixed-width windows make key sort order == numeric window order."""
        ks = [keys.certificate_key(w) for w in (0, 9, 10, 99, 100, 12345678)]
        assert ks == sorted(ks)


class TestValidation:
    def test_window_bounds(self) -> None:
        with pytest.raises(keys.KeyFormatError):
            keys.payload_key(-1, 0, "1.0.0")
        with pytest.raises(keys.KeyFormatError):
            keys.certificate_key(keys.MAX_WINDOW)

    def test_uid_bounds(self) -> None:
        with pytest.raises(keys.KeyFormatError):
            keys.payload_key(0, keys.MAX_UID, "1.0.0")
        with pytest.raises(keys.KeyFormatError):
            keys.audit_report_key(0, -1, 0)

    def test_version_charset(self) -> None:
        with pytest.raises(keys.KeyFormatError):
            keys.payload_key(0, 0, "1.0/evil")
        with pytest.raises(keys.KeyFormatError):
            keys.payload_key(0, 0, "")

    def test_checkpoint_kind_charset(self) -> None:
        with pytest.raises(keys.KeyFormatError):
            keys.checkpoint_key(0, "a/b")

    def test_shard_hash16(self) -> None:
        with pytest.raises(keys.KeyFormatError):
            keys.shard_key("XYZ")
        with pytest.raises(keys.KeyFormatError):
            keys.shard_key("0123456789abcde")  # 15 chars
        with pytest.raises(keys.KeyFormatError):
            keys.shard_key("0123456789ABCDEF")  # uppercase

    def test_attest_nonce(self) -> None:
        with pytest.raises(keys.KeyFormatError):
            keys.attest_key(0, "short")  # 5 chars, non-hex 's'
        with pytest.raises(keys.KeyFormatError):
            keys.attest_key(0, "ab" * 33)  # 66 chars > 64

    def test_parse_rejects_malformed(self) -> None:
        bad = [
            "payloads/w123/uid00007-v1.0.0.zst",  # window not 8 digits
            "payloads/w00000003/uid7-v1.0.0.zst",  # uid not 5 digits
            "payloads/w00000003/uid00007-v1.0.0.pt",  # wrong extension
            "payload/w00000003/uid00007-v1.0.0.zst",  # wrong root
            "payloads/w00000003/uid00007-v1.0.0.zst/extra",  # trailing junk
        ]
        for key in bad:
            with pytest.raises(keys.KeyFormatError):
                keys.parse_payload_key(key)
        with pytest.raises(keys.KeyFormatError):
            keys.parse_certificate_key("certificates/w1.json")
        with pytest.raises(keys.KeyFormatError):
            keys.parse_shard_key("shards/nothex.bin")
        with pytest.raises(keys.KeyFormatError):
            keys.parse_audit_report_key("audits/w00000001/auditor1-miner2.json")

    def test_parse_cross_kind_rejects(self) -> None:
        with pytest.raises(keys.KeyFormatError):
            keys.parse_telemetry_key(keys.certificate_key(1))
        with pytest.raises(keys.KeyFormatError):
            keys.parse_aggregator_key(keys.payload_key(1, 2, "1.0.0"))
