"""release/replay_window.py — argument surface + monkeypatched replay path.

The heavy path (torch DCP load, subnet.core.replay.WindowReplayer) runs only on the
GPU suite; here `_replay` is monkeypatched and we verify plumbing: manifest
resolution, report normalization/validation, JSON output, and exit codes.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from test_release_provenance import build_fixture_bundle, make_audit, make_manifest

from release import replay_window
from release.provenance import canonical_bytes
from release.replay_window import ReplayCLIError, build_parser, load_manifest_arg, main, report_to_dict

# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def test_parser_full_surface(tmp_path):
    args = build_parser().parse_args(
        [
            "--bundle", str(tmp_path / "bundle"),
            "--window", "1234",
            "--miner-uid", "17",
            "--theta-start", str(tmp_path / "ckpt"),
            "--config", str(tmp_path / "bulk.yaml"),
            "--backend", "reference",
            "--device", "cpu",
            "--out", str(tmp_path / "report.json"),
        ]
    )
    assert args.window == 1234
    assert args.miner_uid == 17
    assert args.bundle == tmp_path / "bundle"
    assert args.manifest is None
    assert args.backend == "reference"
    assert args.out == tmp_path / "report.json"


def test_parser_defaults(tmp_path):
    args = build_parser().parse_args(
        ["--manifest", "m.json", "--window", "0", "--miner-uid", "1", "--theta-start", "d"]
    )
    assert args.backend == "mok"
    assert args.device == "cuda"
    assert args.config is None
    assert args.out is None


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--window", "1", "--miner-uid", "2", "--theta-start", "d"],           # no source
        ["--bundle", "b", "--manifest", "m", "--window", "1", "--miner-uid", "2", "--theta-start", "d"],
        ["--bundle", "b", "--miner-uid", "2", "--theta-start", "d"],           # no window
        ["--bundle", "b", "--window", "1", "--theta-start", "d"],              # no uid
        ["--bundle", "b", "--window", "1", "--miner-uid", "2"],                # no theta-start
        ["--bundle", "b", "--window", "1", "--miner-uid", "2", "--theta-start", "d", "--backend", "x"],
    ],
)
def test_parser_rejects(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
# Manifest resolution
# --------------------------------------------------------------------------- #


def test_load_manifest_from_bundle(tmp_path):
    build_fixture_bundle(tmp_path / "bundle")
    manifest = load_manifest_arg(tmp_path / "bundle", None)
    assert manifest == make_manifest()


def test_load_manifest_from_file(tmp_path):
    path = tmp_path / "m.json"
    path.write_bytes(canonical_bytes(make_manifest()))
    manifest = load_manifest_arg(None, path)
    assert manifest.run_id == "mok54b-stage2-fixture"


def test_load_manifest_missing_or_garbage(tmp_path):
    with pytest.raises(ReplayCLIError, match="not found"):
        load_manifest_arg(None, tmp_path / "absent.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ReplayCLIError, match="invalid manifest"):
        load_manifest_arg(None, bad)


# --------------------------------------------------------------------------- #
# Report normalization
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class FakeAuditReport:
    miner_uid: int
    window: int
    theta_start_root: str
    committed_theta_end: str
    replayed_theta_end: str
    match: bool
    divergences: list
    wall_time_s: float
    auditor_uid: int
    signature: str


def test_report_to_dict_accepts_dict_and_dataclass():
    as_dict = make_audit()
    assert report_to_dict(as_dict) == as_dict
    as_dc = FakeAuditReport(**as_dict)
    assert report_to_dict(as_dc) == as_dict


def test_report_to_dict_rejects_malformed():
    bad = make_audit()
    bad["committed_theta_end"] = "short"
    with pytest.raises(ReplayCLIError, match="malformed AuditReport"):
        report_to_dict(bad)
    with pytest.raises(ReplayCLIError, match="unsupported AuditReport type"):
        report_to_dict(42)


# --------------------------------------------------------------------------- #
# main() with a monkeypatched replay
# --------------------------------------------------------------------------- #


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--bundle", str(tmp_path / "bundle"),
        "--window", "1",
        "--miner-uid", "17",
        "--theta-start", str(tmp_path / "ckpt"),
        *extra,
    ]


def test_main_match_writes_report_and_exits_zero(tmp_path, monkeypatch, capsys):
    build_fixture_bundle(tmp_path / "bundle")
    calls = {}

    def fake_replay(manifest, **kwargs):
        calls["manifest"] = manifest
        calls.update(kwargs)
        return make_audit(match=True, miner=17)

    monkeypatch.setattr(replay_window, "_replay", fake_replay)
    out = tmp_path / "report.json"
    rc = main(_argv(tmp_path, "--out", str(out)))
    assert rc == 0
    assert calls["manifest"] == make_manifest()
    assert calls["window"] == 1
    assert calls["miner_uid"] == 17
    assert calls["theta_start"] == tmp_path / "ckpt"
    assert calls["backend"] == "mok"
    assert json.loads(out.read_text()) == make_audit(match=True, miner=17)
    assert "MATCH" in capsys.readouterr().err


def test_main_mismatch_exits_one(tmp_path, monkeypatch, capsys):
    build_fixture_bundle(tmp_path / "bundle")
    monkeypatch.setattr(replay_window, "_replay", lambda m, **kw: make_audit(match=False))
    rc = main(_argv(tmp_path))
    captured = capsys.readouterr()
    assert rc == 1
    assert "MISMATCH" in captured.err
    # without --out the report goes to stdout
    assert json.loads(captured.out)["match"] is False


def test_main_replay_error_exits_two(tmp_path, monkeypatch, capsys):
    build_fixture_bundle(tmp_path / "bundle")

    def boom(manifest, **kwargs):
        raise ReplayCLIError("config hash mismatch")

    monkeypatch.setattr(replay_window, "_replay", boom)
    rc = main(_argv(tmp_path))
    assert rc == 2
    assert "config hash mismatch" in capsys.readouterr().err


def test_main_missing_bundle_manifest_exits_two(tmp_path, monkeypatch):
    monkeypatch.setattr(replay_window, "_replay", lambda m, **kw: make_audit())
    rc = main(_argv(tmp_path))  # bundle was never built
    assert rc == 2


def test_replay_requires_config(tmp_path):
    with pytest.raises(ReplayCLIError, match="--config is required"):
        replay_window._replay(
            make_manifest(),
            window=1,
            miner_uid=17,
            theta_start=tmp_path / "ckpt",
            config_path=None,
            backend="reference",
            device="cpu",
        )


def test_replay_rejects_wrong_config_hash(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text("{}\n")  # RunConfig defaults — hashes to something != fixture's "11"*32
    with pytest.raises(ReplayCLIError, match="config hash mismatch"):
        replay_window._replay(
            make_manifest(),
            window=1,
            miner_uid=17,
            theta_start=tmp_path / "ckpt",
            config_path=cfg_path,
            backend="reference",
            device="cpu",
        )
