"""Tests for fleet/cli.py — parsers + flow orchestration (mocks for heavy paths,
real tiny runs for `init-publish --local-only` and the calibrate subcommands)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import test_window_runner as twr
import torch
import yaml

import fleet.cli as cli
from fleet.attestation.challenge import make_challenge
from fleet.attestation.reference_step import AttestationResponse
from mok_core.config.schemas import BucketCreds
from mok_core.data.shards import shard_filename
from mok_core.determinism import hash_named_tensors
from mok_core.model import build_reference_model

BLOCK_HASH_HEX = "01" * 32
UID = 3


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


# --------------------------------------------------------------------------- #
# Tiny config + dataset builders (real files for the CLI to consume)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cfg_yaml(tmp_path_factory) -> Path:
    """The twr tiny RunConfig, written as a full base YAML."""
    cfg = twr.make_run_cfg()
    path = tmp_path_factory.mktemp("cli-cfg") / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """A dataprep-shaped dataset dir: content-addressed shards + shard_index.json."""
    root = tmp_path_factory.mktemp("cli-data")
    hashes = []
    for i in range(twr.NUM_SHARDS):
        data = twr.shard_array(i).tobytes()
        digest = hashlib.blake2b(data, digest_size=32).digest()
        (root / shard_filename(digest)).write_bytes(data)
        hashes.append(digest.hex())
    index = {"name": "bulk", "seq_len": twr.SEQ_LEN, "shard_hashes": hashes}
    (root / "shard_index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# mok-attest
# --------------------------------------------------------------------------- #


def test_attest_challenge_from_block_hash(tmp_path: Path) -> None:
    out = tmp_path / "challenge.json"
    rc = cli.attest_main(
        ["challenge", "--block-hash", BLOCK_HASH_HEX, "--block", "77", "--out", str(out)]
    )
    assert rc == 0
    emitted = read_json(out)
    expected = make_challenge(bytes.fromhex(BLOCK_HASH_HEX), 77)
    assert emitted["challenge_id"] == expected.challenge_id
    assert emitted["seed"] == expected.seed
    assert emitted["issued_block"] == 77


def test_attest_challenge_from_chain_requires_config() -> None:
    with pytest.raises(SystemExit, match="--from-chain requires --config"):
        cli.attest_main(["challenge", "--from-chain"])


def test_attest_respond_delegates_to_run_reference(tmp_path: Path, monkeypatch) -> None:
    challenge = make_challenge(bytes.fromhex(BLOCK_HASH_HEX), 1)
    challenge_file = tmp_path / "c.json"
    challenge_file.write_text(json.dumps(challenge.model_dump()), encoding="utf-8")
    response = AttestationResponse(
        challenge_id=challenge.challenge_id, state_root="ab" * 32, wall_time_s=1.0, fingerprint={}
    )
    seen: dict[str, Any] = {}

    def fake_run_reference(ch, *, backend, device, comm=None):
        seen.update(challenge=ch, backend=backend, device=device)
        return response

    monkeypatch.setattr(cli, "run_reference", fake_run_reference)
    # In-process invocation: earlier suite tests may have initialized CUDA (torch
    # DCP's writer does), which the process-entry determinism guard must reject.
    monkeypatch.setattr(cli, "enforce_determinism", lambda: None)
    out = tmp_path / "r.json"
    rc = cli.attest_main(
        ["respond", "--challenge", str(challenge_file), "--backend", "reference",
         "--device", "cpu", "--out", str(out)]
    )
    assert rc == 0
    assert seen["challenge"] == challenge and seen["backend"] == "reference"
    assert read_json(out)["state_root"] == "ab" * 32


def test_attest_verify_exit_codes(tmp_path: Path) -> None:
    challenge = make_challenge(bytes.fromhex(BLOCK_HASH_HEX), 1)
    response = AttestationResponse(
        challenge_id=challenge.challenge_id, state_root="ab" * 32, wall_time_s=9.0, fingerprint={}
    )
    cf, rf = tmp_path / "c.json", tmp_path / "r.json"
    cf.write_text(json.dumps(challenge.model_dump()), encoding="utf-8")
    rf.write_text(json.dumps(response.model_dump()), encoding="utf-8")
    common = ["verify", "--challenge", str(cf), "--response", str(rf),
              "--issued-ts", "0", "--received-ts", "60"]
    ok_out = tmp_path / "ok.json"
    assert cli.attest_main([*common, "--expected-root", "ab" * 32, "--out", str(ok_out)]) == 0
    assert read_json(ok_out)["ok"] is True
    bad_out = tmp_path / "bad.json"
    assert cli.attest_main([*common, "--expected-root", "cd" * 32, "--out", str(bad_out)]) == 1
    assert read_json(bad_out)["ok"] is False


# --------------------------------------------------------------------------- #
# mok-onboard
# --------------------------------------------------------------------------- #


def test_onboard_all_skips(cfg_yaml: Path, capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "enforce_determinism", lambda: None)  # in-process: CUDA may exist
    rc = cli.onboard_main(
        ["--config", str(cfg_yaml)]
        + [f"--skip-{s}" for s in ("preflight", "wallet", "register", "bucket", "init", "attest")]
    )
    assert rc == 0
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert [ln["step"] for ln in lines] == [
        "preflight", "wallet", "register", "bucket", "init", "attest", "done",
    ]
    assert all(ln.get("skipped") for ln in lines[:-1])


def test_onboard_flow_orchestration(cfg_yaml: Path, monkeypatch, capsys) -> None:
    """preflight -> wallet -> register -> bucket -> self-attest with mocks;
    each step consumes the previous step's products."""
    order: list[str] = []
    report = MagicMock()
    report.ok = True
    report.checks = []
    report.strict.side_effect = lambda: order.append("strict")
    monkeypatch.setattr(cli, "run_preflight", lambda **kw: (order.append("preflight"), report)[1])
    monkeypatch.setattr(cli, "enforce_determinism", lambda: None)  # in-process: CUDA may exist

    wallet = object()
    monkeypatch.setattr(
        cli, "ensure_wallet", lambda cfg, interactive: (order.append("wallet"), wallet)[1]
    )

    chain = MagicMock()
    chain.current_block.return_value = 5
    chain.block_hash.return_value = bytes.fromhex(BLOCK_HASH_HEX)
    seen_wallet: list[Any] = []

    def fake_make_chain(cfg, w=None):
        seen_wallet.append(w)
        return chain

    monkeypatch.setattr(cli, "_make_chain", fake_make_chain)
    monkeypatch.setattr(cli, "register", lambda c: (order.append("register"), 42)[1])
    creds = BucketCreds(account_id="a", bucket_name="b", access_key_id="k", secret_access_key="s")
    monkeypatch.setattr(cli, "bucket_creds_from_env", lambda hotkey, env=None: creds)
    monkeypatch.setattr(
        cli, "commit_bucket_credentials", lambda c, cr: (order.append(f"bucket:{cr.bucket_name}"), True)[1]
    )

    def fake_run_reference(ch, *, backend, device, comm=None):
        order.append("attest")
        return AttestationResponse(
            challenge_id=ch.challenge_id, state_root="ab" * 32, wall_time_s=30.0, fingerprint={}
        )

    monkeypatch.setattr(cli, "run_reference", fake_run_reference)

    rc = cli.onboard_main(
        ["--config", str(cfg_yaml), "--skip-init", "--backend", "reference", "--device", "cpu"]
    )
    assert rc == 0
    assert order == ["preflight", "strict", "wallet", "register", "bucket:b", "attest", "attest"]
    assert seen_wallet[0] is wallet  # the chain client got the onboarded wallet
    lines = [json.loads(ln) for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    attest_line = next(ln for ln in lines if ln["step"] == "attest")
    assert attest_line["ok"] is True and attest_line["deterministic"] is True


def test_onboard_nondeterministic_node_fails_attest(cfg_yaml: Path, monkeypatch) -> None:
    roots = iter(["aa" * 32, "bb" * 32])
    chain = MagicMock()
    chain.current_block.return_value = 5
    chain.block_hash.return_value = bytes.fromhex(BLOCK_HASH_HEX)
    monkeypatch.setattr(cli, "_make_chain", lambda cfg, w=None: chain)

    def flaky(ch, *, backend, device, comm=None):
        return AttestationResponse(
            challenge_id=ch.challenge_id, state_root=next(roots), wall_time_s=1.0, fingerprint={}
        )

    monkeypatch.setattr(cli, "run_reference", flaky)
    monkeypatch.setattr(cli, "enforce_determinism", lambda: None)  # in-process: CUDA may exist
    rc = cli.onboard_main(
        ["--config", str(cfg_yaml), "--skip-preflight", "--skip-wallet", "--skip-register",
         "--skip-bucket", "--skip-init"]
    )
    assert rc == 1


def test_onboard_init_requires_owner_uid(cfg_yaml: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "_make_chain", lambda cfg, w=None: MagicMock())
    monkeypatch.setattr(cli, "enforce_determinism", lambda: None)  # in-process: CUDA may exist
    with pytest.raises(SystemExit, match="--owner-uid"):
        cli.onboard_main(
            ["--config", str(cfg_yaml), "--skip-preflight", "--skip-wallet", "--skip-register",
             "--skip-bucket", "--skip-attest"]
        )


# --------------------------------------------------------------------------- #
# mok-init-publish
# --------------------------------------------------------------------------- #


def test_init_publish_local_only_real_run(cfg_yaml: Path, tmp_path: Path) -> None:
    out = tmp_path / "root.json"
    rc = cli.init_publish_main(
        ["--config", str(cfg_yaml), "--local-dir", str(tmp_path / "ckpt"),
         "--seed", "42", "--local-only", "--out", str(out)]
    )
    assert rc == 0
    emitted = read_json(out)
    expected = hash_named_tensors(
        build_reference_model(twr.make_model_cfg(), 42).iter_master_params()
    )
    assert emitted["init_state_root"] == expected
    assert emitted["local_only"] is True
    assert (tmp_path / "ckpt" / "w00000000" / "meta.json").is_file()


# --------------------------------------------------------------------------- #
# mok-calibrate
# --------------------------------------------------------------------------- #


def calibrate_common(cfg_yaml: Path, data_dir: Path, tmp_path: Path) -> list[str]:
    return [
        "--config", str(cfg_yaml),
        "--data-dir", str(data_dir),
        "--work-dir", str(tmp_path / "work"),
        "--run-seed", twr.RUN_SEED.hex(),
        "--seed", str(twr.SEED),
        "--uid", str(UID),
        "--backend", "reference",
        "--device", "cpu",
    ]


def test_calibrate_rehearse_real_run(cfg_yaml: Path, data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = cli.calibrate_main(
        ["rehearse", *calibrate_common(cfg_yaml, data_dir, tmp_path),
         "--windows", "1", "--out", str(out)]
    )
    assert rc == 0
    report = read_json(out)
    assert report["windows"] == [0]
    assert report["determinism_check"] is True
    assert len(report["loss_curve"]) == 1 and report["loss_curve"][0] > 0.0


def test_calibrate_sweep_real_run_emits_tuned_yaml(
    cfg_yaml: Path, data_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "sweep.json"
    tuned = tmp_path / "mok_tuned.yaml"
    rc = cli.calibrate_main(
        ["sweep", *calibrate_common(cfg_yaml, data_dir, tmp_path),
         "--windows-per-point", "1", "--sms", "2", "--minibatch", "256,512",
         "--tuned-out", str(tuned), "--out", str(out)]
    )
    assert rc == 0
    emitted = read_json(out)
    assert len(emitted["results"]) == 2
    assert emitted["best"] in [r["point"] for r in emitted["results"]]
    loaded = yaml.safe_load(tuned.read_text(encoding="utf-8"))
    assert loaded["mok"]["minibatch_size"] in (256, 512)
    assert emitted["tuned_yaml"] == str(tuned)


def test_calibrate_adam_ab_real_run(cfg_yaml: Path, data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "ab.json"
    rc = cli.calibrate_main(
        ["adam-ab", *calibrate_common(cfg_yaml, data_dir, tmp_path),
         "--windows", "1", "--k", "2", "--out", str(out)]
    )
    assert rc == 0
    report = read_json(out)
    assert report["k"] == 2
    assert len(report["losses_reset_every_window"]) == 1
    # single window: both arms start fresh -> identical, delta 0 -> keep reset=1
    assert report["delta_final_loss"] == pytest.approx(0.0)
    assert report["keep_reset_every_window"] is True


def test_parsers_reject_unknown_subcommands() -> None:
    with pytest.raises(SystemExit):
        cli.attest_main(["frobnicate"])
    with pytest.raises(SystemExit):
        cli.calibrate_main(["--config", "x.yaml"])  # missing subcommand
