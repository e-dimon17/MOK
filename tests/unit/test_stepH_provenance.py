"""H/provenance.py — bundle build → verify round trip, tamper detection, goldens.

Also hosts the shared step-H fixture builders (make_manifest / make_records /
make_audit / build_fixture_bundle) imported by the other test_stepH_* modules.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from H import provenance
from H.provenance import (
    BUNDLE_SPEC_VERSION,
    INDEX_FILENAME,
    BundleError,
    BundleManifest,
    WindowRecord,
    audit_report_message,
    audit_report_problems,
    build_bundle,
    bundle_root_hash,
)
from H.verify_bundle import verify
from mok_core.config import DatasetManifestRef, PRFSpec, RunManifest

# --------------------------------------------------------------------------- #
# Shared fixtures for all step-H tests
# --------------------------------------------------------------------------- #


def make_manifest() -> RunManifest:
    return RunManifest(
        spec_version=1,
        run_id="mok54b-stage2-fixture",
        netuid=42,
        network="test",
        config_hash="11" * 32,
        container_digest="sha256:" + "22" * 32,
        mok_commit="8f90b74",
        tk_commit="0badc0de",
        attention_backend="cudnn_det",
        start_block=1000,
        blocks_per_window=100,
        prf=PRFSpec(run_seed_hex="00" * 32),
        datasets=(
            DatasetManifestRef(
                name="bulk",
                merkle_root="33" * 32,
                num_shards=4,
                shard_bytes=512,
                seq_len=128,
                tokens_total=262144,
                tokenizer_hash="44" * 32,
            ),
        ),
        init_checkpoint_hash="55" * 32,
    )


def make_records() -> list[WindowRecord]:
    return [
        WindowRecord(
            window=0,
            state_root="aa" * 32,
            payload_hashes={1: "bb" * 32, 2: "cc" * 32},
            telemetry_hash="dd" * 32,
        ),
        WindowRecord(
            window=1,
            state_root="ee" * 32,
            certificate={"window": 1, "leader_uid": 3, "included_uids": [1, 2]},
        ),
    ]


def make_audit(*, window: int = 1, miner: int = 1, auditor: int = 9, match: bool = True) -> dict:
    committed = "ee" * 32
    replayed = committed if match else "ff" * 32
    return {
        "miner_uid": miner,
        "window": window,
        "theta_start_root": "aa" * 32,
        "committed_theta_end": committed,
        "replayed_theta_end": replayed,
        "match": match,
        "divergences": [] if match else [{"name": "lm_head.weight", "expected": "ab", "actual": "cd"}],
        "wall_time_s": 123.5,
        "auditor_uid": auditor,
        "signature": "",
    }


def build_fixture_bundle(out_dir: Path, **overrides) -> BundleManifest:
    weights = out_dir.parent / "release_weights"
    weights.mkdir(parents=True, exist_ok=True)
    wfile = weights / "model-00001.safetensors"
    wfile.write_bytes(b"MOK-WEIGHTS\x00\x01\x02")
    kwargs = {
        "manifest": make_manifest(),
        "window_records": make_records(),
        "audit_reports": [make_audit()],
        "weights_files": [wfile],
        "eval_results": {"mmlu": {"acc": 0.65}},
        "extra": {"note": "fixture"},
        "built_at_block": 123456,
    }
    kwargs.update(overrides)
    return build_bundle(out_dir, **kwargs)


def patch_bundle_file(bundle: Path, rel: str, data: bytes) -> None:
    """Rewrite one bundle file and re-consistent the index (digest + root) so
    content-level checks can be exercised past the hash layer."""
    index = json.loads((bundle / INDEX_FILENAME).read_bytes())
    path = bundle / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    index["files"][rel] = provenance.blake2b_hex(data)
    index["root_hash"] = bundle_root_hash(index["files"])
    (bundle / INDEX_FILENAME).write_bytes(
        json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    )


# --------------------------------------------------------------------------- #
# Build -> verify round trip
# --------------------------------------------------------------------------- #


def test_round_trip_ok(tmp_path):
    index = build_fixture_bundle(tmp_path / "bundle")
    report = verify(tmp_path / "bundle")
    assert report.ok, report.problems
    assert report.problems == []
    assert report.files_checked == len(index.files)
    assert report.windows == 2
    assert report.audits == 1


def test_index_written_matches_return_and_manifest_hash(tmp_path):
    index = build_fixture_bundle(tmp_path / "bundle")
    on_disk = BundleManifest.model_validate(json.loads((tmp_path / "bundle" / INDEX_FILENAME).read_bytes()))
    assert on_disk == index
    assert index.spec_version == BUNDLE_SPEC_VERSION
    assert index.manifest_hash == make_manifest().manifest_hash()
    assert index.built_at_block == 123456
    # manifest.json bytes hash to exactly the manifest_hash (canonical serialization)
    assert index.files["manifest.json"] == index.manifest_hash
    expected = {
        "manifest.json",
        "windows.jsonl",
        "audits.jsonl",
        "evals.json",
        "weights/model-00001.safetensors",
        "replay/replay_window.py",
    }
    assert set(index.files) == expected


def test_build_is_deterministic(tmp_path):
    a = build_fixture_bundle(tmp_path / "a")
    b = build_fixture_bundle(tmp_path / "b")
    assert a == b
    for rel in a.files:
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes()
    assert (tmp_path / "a" / INDEX_FILENAME).read_bytes() == (tmp_path / "b" / INDEX_FILENAME).read_bytes()


@pytest.mark.parametrize(
    "rel",
    [
        "manifest.json",
        "windows.jsonl",
        "audits.jsonl",
        "evals.json",
        "weights/model-00001.safetensors",
        "replay/replay_window.py",
    ],
)
def test_tamper_any_file_fails_naming_it(tmp_path, rel):
    build_fixture_bundle(tmp_path / "bundle")
    path = tmp_path / "bundle" / rel
    path.write_bytes(path.read_bytes() + b"x")
    report = verify(tmp_path / "bundle")
    assert not report.ok
    assert any(f"hash mismatch: {rel}" == p for p in report.problems), report.problems


def test_missing_file_reported(tmp_path):
    build_fixture_bundle(tmp_path / "bundle")
    (tmp_path / "bundle" / "evals.json").unlink()
    report = verify(tmp_path / "bundle")
    assert not report.ok
    assert "missing file: evals.json" in report.problems


def test_unlisted_file_reported(tmp_path):
    build_fixture_bundle(tmp_path / "bundle")
    (tmp_path / "bundle" / "weights" / "sneaky.bin").write_bytes(b"backdoor")
    report = verify(tmp_path / "bundle")
    assert not report.ok
    assert "unlisted file: weights/sneaky.bin" in report.problems


def test_root_hash_tamper_detected(tmp_path):
    build_fixture_bundle(tmp_path / "bundle")
    index = json.loads((tmp_path / "bundle" / INDEX_FILENAME).read_bytes())
    index["root_hash"] = "0" * 64
    (tmp_path / "bundle" / INDEX_FILENAME).write_bytes(json.dumps(index).encode())
    report = verify(tmp_path / "bundle")
    assert not report.ok
    assert any("root_hash does not recompute" in p for p in report.problems)


# --------------------------------------------------------------------------- #
# Golden vectors
# --------------------------------------------------------------------------- #


def test_root_hash_golden_vector(tmp_path):
    """Fixed fixture (replay script excluded — its source evolves) pins the
    release root. # consensus constant — change requires a bundle spec bump."""
    index = build_fixture_bundle(tmp_path / "bundle", include_replay_script=False)
    assert index.root_hash == "3530f8d2da971005eed3d6e231429a7137723f57cf1e2aed4cc286589aa3b84a"


def test_bundle_root_hash_order_independent_and_pair_bound():
    files = {"a.txt": "11" * 32, "b/c.bin": "22" * 32}
    same = {"b/c.bin": "22" * 32, "a.txt": "11" * 32}
    assert bundle_root_hash(files) == bundle_root_hash(same)
    # moving a digest to a different relpath must change the root
    swapped = {"a.txt": "22" * 32, "b/c.bin": "11" * 32}
    assert bundle_root_hash(swapped) != bundle_root_hash(files)
    with pytest.raises(BundleError):
        bundle_root_hash({"a": "not-hex"})


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": -1, "state_root": "aa" * 32},
        {"window": 0, "state_root": "zz" * 32},               # non-hex
        {"window": 0, "state_root": "AA" * 32},               # uppercase rejected
        {"window": 0, "state_root": "aa" * 31},               # wrong length
        {"window": 0, "state_root": "aa" * 32, "payload_hashes": {1: "xyz"}},
        {"window": 0, "state_root": "aa" * 32, "payload_hashes": {-2: "bb" * 32}},
        {"window": 0, "state_root": "aa" * 32, "telemetry_hash": "beef"},
    ],
)
def test_window_record_bad_inputs_rejected(kwargs):
    with pytest.raises(ValidationError):
        WindowRecord(**kwargs)


def test_duplicate_windows_rejected(tmp_path):
    records = [make_records()[0], make_records()[0]]
    with pytest.raises(BundleError, match="distinct windows"):
        build_fixture_bundle(tmp_path / "bundle", window_records=records, audit_reports=[])


def test_nonempty_out_dir_rejected(tmp_path):
    out = tmp_path / "bundle"
    out.mkdir()
    (out / "stale.txt").write_text("old release")
    with pytest.raises(BundleError, match="non-empty"):
        build_fixture_bundle(out)


def test_missing_weights_file_rejected(tmp_path):
    with pytest.raises(BundleError, match="does not exist"):
        build_fixture_bundle(tmp_path / "bundle", weights_files=[tmp_path / "nope.safetensors"])


def test_bad_audit_report_rejected_at_build(tmp_path):
    bad = make_audit()
    del bad["wall_time_s"]
    with pytest.raises(BundleError, match="missing field 'wall_time_s'"):
        build_fixture_bundle(tmp_path / "b1", audit_reports=[bad])

    inconsistent = make_audit(match=True)
    inconsistent["replayed_theta_end"] = "ff" * 32
    with pytest.raises(BundleError, match="inconsistent"):
        build_fixture_bundle(tmp_path / "b2", audit_reports=[inconsistent])

    orphan = make_audit(window=99)
    with pytest.raises(BundleError, match="no WindowRecord"):
        build_fixture_bundle(tmp_path / "b3", audit_reports=[orphan])


def test_audit_report_problems_field_checks():
    good = make_audit()
    assert audit_report_problems(good) == []
    for key, bad_value in [
        ("miner_uid", -1),
        ("miner_uid", True),
        ("window", "1"),
        ("theta_start_root", "aa"),
        ("match", "yes"),
        ("divergences", "none"),
        ("wall_time_s", float("nan")),
        ("wall_time_s", -1.0),
        ("signature", "abc"),  # odd-length hex
        ("signature", None),
    ]:
        report = dict(good)
        report[key] = bad_value
        assert audit_report_problems(report), f"{key}={bad_value!r} should be rejected"


def test_audit_report_message_excludes_signature():
    a = make_audit()
    b = dict(a, signature="ab" * 64)
    assert audit_report_message(a) == audit_report_message(b)
    assert len(audit_report_message(a)) == 32
    c = dict(a, wall_time_s=999.0)
    assert audit_report_message(c) != audit_report_message(a)


# --------------------------------------------------------------------------- #
# Weights reference mode
# --------------------------------------------------------------------------- #


def test_hashed_reference_weights_mode(tmp_path):
    index = build_fixture_bundle(tmp_path / "bundle", copy_weights=False)
    rel = "weights/model-00001.safetensors.ref.json"
    assert rel in index.files
    assert "weights/model-00001.safetensors" not in index.files
    ref = json.loads((tmp_path / "bundle" / rel).read_bytes())
    assert ref["filename"] == "model-00001.safetensors"
    assert ref["bytes"] == len(b"MOK-WEIGHTS\x00\x01\x02")
    assert ref["blake2b"] == provenance.blake2b_hex(b"MOK-WEIGHTS\x00\x01\x02")
    assert verify(tmp_path / "bundle").ok


def test_malformed_weights_reference_flagged(tmp_path):
    build_fixture_bundle(tmp_path / "bundle", copy_weights=False)
    rel = "weights/model-00001.safetensors.ref.json"
    patch_bundle_file(tmp_path / "bundle", rel, b'{"filename": "x"}')
    report = verify(tmp_path / "bundle")
    assert not report.ok
    assert f"{rel}: malformed weights reference" in report.problems
