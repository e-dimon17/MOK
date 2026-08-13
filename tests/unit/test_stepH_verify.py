"""H/verify_bundle.py — content-level checks, signature hook, index shape, CLI."""

from __future__ import annotations

import hashlib
import json

from test_stepH_provenance import build_fixture_bundle, make_audit, make_records, patch_bundle_file

import mok_core.chain as chain_mod
from H.provenance import INDEX_FILENAME, audit_report_message, canonical_bytes
from H.verify_bundle import build_parser, main, verify

# --------------------------------------------------------------------------- #
# Content checks past the hash layer (via patch_bundle_file)
# --------------------------------------------------------------------------- #


def test_missing_bundle_dir(tmp_path):
    report = verify(tmp_path / "nope")
    assert not report.ok
    assert any("does not exist" in p for p in report.problems)


def test_missing_index(tmp_path):
    (tmp_path / "b").mkdir()
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("index.json: missing" in p for p in report.problems)


def test_index_shape_problems(tmp_path):
    build_fixture_bundle(tmp_path / "b")
    index = json.loads((tmp_path / "b" / INDEX_FILENAME).read_bytes())
    index["spec_version"] = 999
    index["surprise"] = 1
    (tmp_path / "b" / INDEX_FILENAME).write_bytes(json.dumps(index).encode())
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("unsupported spec_version 999" in p for p in report.problems)
    assert any("unexpected key 'surprise'" in p for p in report.problems)


def test_index_missing_key(tmp_path):
    build_fixture_bundle(tmp_path / "b")
    index = json.loads((tmp_path / "b" / INDEX_FILENAME).read_bytes())
    del index["built_at_block"]
    (tmp_path / "b" / INDEX_FILENAME).write_bytes(json.dumps(index).encode())
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("missing key 'built_at_block'" in p for p in report.problems)


def test_non_monotonic_windows_flagged(tmp_path):
    build_fixture_bundle(tmp_path / "b", audit_reports=[])
    records = make_records()
    blob = canonical_bytes(records[1]) + b"\n" + canonical_bytes(records[0]) + b"\n"
    patch_bundle_file(tmp_path / "b", "windows.jsonl", blob)
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("not strictly increasing" in p for p in report.problems)


def test_bad_state_root_in_windows_flagged(tmp_path):
    build_fixture_bundle(tmp_path / "b", audit_reports=[])
    line = json.dumps({"window": 0, "state_root": "zz" * 32}).encode()
    patch_bundle_file(tmp_path / "b", "windows.jsonl", line + b"\n")
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("windows.jsonl line 1: invalid WindowRecord" in p for p in report.problems)


def test_malformed_audit_line_flagged(tmp_path):
    build_fixture_bundle(tmp_path / "b")
    bad = make_audit()
    bad["wall_time_s"] = -3
    patch_bundle_file(tmp_path / "b", "audits.jsonl", canonical_bytes(bad) + b"\n")
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("wall_time_s" in p for p in report.problems)


def test_audit_for_unknown_window_flagged(tmp_path):
    build_fixture_bundle(tmp_path / "b")
    orphan = make_audit(window=7)
    orphan["committed_theta_end"] = orphan["replayed_theta_end"] = "ee" * 32
    patch_bundle_file(tmp_path / "b", "audits.jsonl", canonical_bytes(orphan) + b"\n")
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("window 7 has no WindowRecord" in p for p in report.problems)


def test_tampered_manifest_hash_flagged(tmp_path):
    build_fixture_bundle(tmp_path / "b")
    raw = json.loads((tmp_path / "b" / "manifest.json").read_bytes())
    raw["run_id"] = "forged-run"
    patch_bundle_file(tmp_path / "b", "manifest.json", canonical_bytes(raw))
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("canonical hash does not match" in p for p in report.problems)


def test_evals_shape_flagged(tmp_path):
    build_fixture_bundle(tmp_path / "b")
    patch_bundle_file(tmp_path / "b", "evals.json", b'["not", "an", "object"]')
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("evals.json: must be an object" in p for p in report.problems)


# --------------------------------------------------------------------------- #
# Audit signature hook (optional-pass semantics)
# --------------------------------------------------------------------------- #

_SECRET = b"auditor-secret"


def _sign(msg: bytes) -> str:
    return hashlib.blake2b(_SECRET + msg, digest_size=64).hexdigest()


def _signed_audit() -> dict:
    report = make_audit()
    report["signature"] = _sign(audit_report_message(report))
    return report


def test_signature_optional_pass_without_hook(tmp_path, monkeypatch):
    monkeypatch.delattr(chain_mod, "verify_audit_signature", raising=False)
    build_fixture_bundle(tmp_path / "b", audit_reports=[_signed_audit()])
    assert verify(tmp_path / "b").ok


def test_signature_checked_when_hook_present(tmp_path, monkeypatch):
    build_fixture_bundle(tmp_path / "b", audit_reports=[_signed_audit()])
    seen: list[tuple[bytes, bytes, int]] = []

    def good_verifier(message: bytes, signature: bytes, auditor_uid: int) -> bool:
        seen.append((message, signature, auditor_uid))
        return signature == hashlib.blake2b(_SECRET + message, digest_size=64).digest()

    monkeypatch.setattr(chain_mod, "verify_audit_signature", good_verifier, raising=False)
    report = verify(tmp_path / "b")
    assert report.ok, report.problems
    assert len(seen) == 1
    assert seen[0][0] == audit_report_message(make_audit())
    assert seen[0][2] == 9

    monkeypatch.setattr(chain_mod, "verify_audit_signature", lambda m, s, u: False, raising=False)
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("signature verification failed" in p for p in report.problems)


def test_signature_hook_exception_is_a_problem(tmp_path, monkeypatch):
    build_fixture_bundle(tmp_path / "b", audit_reports=[_signed_audit()])

    def broken(message: bytes, signature: bytes, auditor_uid: int) -> bool:
        raise RuntimeError("keystore offline")

    monkeypatch.setattr(chain_mod, "verify_audit_signature", broken, raising=False)
    report = verify(tmp_path / "b")
    assert not report.ok
    assert any("signature verification errored" in p for p in report.problems)


def test_unsigned_audit_never_calls_hook(tmp_path, monkeypatch):
    build_fixture_bundle(tmp_path / "b")  # fixture audit has signature ""

    def exploding(message: bytes, signature: bytes, auditor_uid: int) -> bool:
        raise AssertionError("hook must not run for unsigned reports")

    monkeypatch.setattr(chain_mod, "verify_audit_signature", exploding, raising=False)
    assert verify(tmp_path / "b").ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_ok(tmp_path, capsys):
    build_fixture_bundle(tmp_path / "b")
    assert main([str(tmp_path / "b")]) == 0
    assert "OK:" in capsys.readouterr().out


def test_cli_failure_lists_problems(tmp_path, capsys):
    build_fixture_bundle(tmp_path / "b")
    path = tmp_path / "b" / "evals.json"
    path.write_bytes(path.read_bytes() + b"!")
    assert main([str(tmp_path / "b")]) == 1
    err = capsys.readouterr().err
    assert "hash mismatch: evals.json" in err
    assert "FAILED" in err


def test_cli_json_output(tmp_path, capsys):
    build_fixture_bundle(tmp_path / "b")
    assert main([str(tmp_path / "b"), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["windows"] == 2


def test_parser_prog():
    parser = build_parser()
    args = parser.parse_args(["some/dir", "--json"])
    assert args.json is True
