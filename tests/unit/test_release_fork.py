"""CPU tests for D/release_fork.py — the interim-release fork procedure.

The fork is a pure manifest operation: given a lineage checkpoint's meta.json
and the run's (manifest, config), it must append exactly one phase amendment
(anneal data + wsd_linear_decay LR anchored at the consensus decay-start step),
write canonical `manifest.json` + the operator runbook, and leave the original
manifest untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from C.core.checkpoint import CheckpointMeta
from C.core.phase import lr_at, resolve_phase
from C.core.window_runner import run_state_at
from D.release_fork import (
    MANIFEST_FILENAME,
    RUNBOOK_FILENAME,
    ForkError,
    ForkResult,
    fork_release,
    load_checkpoint_meta,
    main,
)
from mok_core.config import (
    InnerOptConfig,
    LRSpec,
    ModelConfig,
    RunConfig,
    WindowConfig,
)
from mok_core.config.canonical import canonical_bytes, config_hash
from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest
from mok_core.determinism import hash_bytes

RUN_SEED = bytes(range(32))
CKPT_WINDOW = 9            # checkpoint holds θ_start(10)
EFFECTIVE_WINDOW = 12      # amendment lands >= 2 windows ahead
DECAY_TOKENS = 400_000
INNER_STEPS = 2


def make_cfg() -> RunConfig:
    return RunConfig(
        model=ModelConfig(
            num_layers=2,
            num_dense_layers=0,
            hidden_size=256,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=128,
            vocab_size=512,
            seq_len=256,
            num_experts=8,
            top_k=2,
            intermediate_size=256,
            ep_size=4,
        ),
        window=WindowConfig(
            inner_steps=INNER_STEPS,
            tokens_per_rank_microbatch=512,
            grad_accum=1,
            accum_ramp_start=1,
        ),
        inner=InnerOptConfig(lr=LRSpec(kind="wsd_flat", peak_lr=3e-4, warmup_steps=4)),
    )


def dataset_ref(name: str) -> DatasetManifestRef:
    return DatasetManifestRef(
        name=name,
        merkle_root=("cd" if name == "bulk" else "ef") * 32,
        num_shards=8,
        shard_bytes=2 * 256 * 8,
        seq_len=256,
        tokens_total=8 * 8 * 256,
        tokenizer_hash="ab" * 32,
    )


def make_manifest(cfg: RunConfig, *, with_anneal: bool = True) -> RunManifest:
    datasets = (dataset_ref("bulk"), dataset_ref("anneal")) if with_anneal else (dataset_ref("bulk"),)
    return RunManifest(
        spec_version=1,
        run_id="release-fork-test",
        netuid=11,
        network="test",
        config_hash=config_hash(cfg),
        container_digest="sha256:" + "22" * 32,
        mok_commit="deadbeef",
        tk_commit="cafebabe",
        attention_backend="cudnn_det",
        start_block=100,
        blocks_per_window=225,
        prf=PRFSpec(run_seed_hex=RUN_SEED.hex()),
        datasets=datasets,
        init_checkpoint_hash="33" * 32,
    )


def write_checkpoint_meta(dir_: Path, manifest: RunManifest) -> CheckpointMeta:
    meta = CheckpointMeta(
        window=CKPT_WINDOW,
        global_step=CKPT_WINDOW + 1,
        tokens_consumed=(CKPT_WINDOW + 1) * INNER_STEPS * 512 * 4,
        state_root="ab" * 32,
        manifest_hash=manifest.manifest_hash(),
        spec_version=1,
    )
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "meta.json").write_bytes(meta.canonical())
    return meta


@pytest.fixture()
def rig(tmp_path: Path) -> dict:
    cfg = make_cfg()
    manifest = make_manifest(cfg)
    meta = write_checkpoint_meta(tmp_path / "ckpt", manifest)
    return {
        "cfg": cfg,
        "manifest": manifest,
        "meta": meta,
        "ckpt": tmp_path / "ckpt",
        "out": tmp_path / "fork",
    }


def do_fork(rig: dict, **overrides) -> ForkResult:
    kwargs = {
        "manifest": rig["manifest"],
        "cfg": rig["cfg"],
        "decay_tokens": DECAY_TOKENS,
        "effective_window": EFFECTIVE_WINDOW,
        "committed_block": 5000,
    }
    kwargs.update(overrides)
    return fork_release(rig["ckpt"], rig["out"], **kwargs)


# --------------------------------------------------------------------------- #
# The fork produces a valid manifest
# --------------------------------------------------------------------------- #


def test_fork_produces_valid_amended_manifest(rig: dict) -> None:
    result = do_fork(rig)
    forked = result.manifest

    assert isinstance(forked, RunManifest)  # model_validate passed => all invariants hold
    assert len(forked.phase_table) == len(rig["manifest"].phase_table) + 1
    entry = forked.phase_table[-1]
    assert entry.start_window == EFFECTIVE_WINDOW
    assert entry.name == "anneal_release"
    assert entry.overrides.data == "anneal"
    assert entry.overrides.lr is not None and entry.overrides.lr.kind == "wsd_linear_decay"
    assert entry.overrides.requires_restart is False  # same shapes: no relaunch

    assert len(forked.amendments) == 1
    amendment = forked.amendments[0]
    assert amendment.kind == "phase"
    assert amendment.seq == 0
    assert amendment.effective_window == EFFECTIVE_WINDOW
    assert amendment.committed_block == 5000

    # on-disk manifest.json is the canonical bytes: file hash == manifest_hash
    data = result.manifest_path.read_bytes()
    assert data == canonical_bytes(forked)
    assert hash_bytes(data) == forked.manifest_hash()
    assert RunManifest.model_validate_json(data.decode()) == forked

    # runbook exists and states the operator-critical facts
    runbook = result.runbook_path.read_text()
    assert result.runbook_path.name == RUNBOOK_FILENAME
    assert forked.manifest_hash() in runbook
    assert rig["manifest"].manifest_hash() in runbook
    assert rig["meta"].state_root in runbook
    assert f"window {CKPT_WINDOW}" in runbook
    assert str(EFFECTIVE_WINDOW) in runbook
    assert "run_miner.sh" in runbook and "MAIN branch" in runbook


def test_phase_resolves_at_effective_window(rig: dict) -> None:
    result = do_fork(rig)
    forked, cfg = result.manifest, rig["cfg"]

    # Just before the fork: still the bulk phase at flat LR.
    before = resolve_phase(forked, cfg, EFFECTIVE_WINDOW - 1)
    assert before.data == "bulk"
    assert before.lr.kind == "wsd_flat"
    assert not before.requires_restart

    # At the fork: anneal data + decay segment anchored at the consensus step.
    at = resolve_phase(forked, cfg, EFFECTIVE_WINDOW)
    assert at.name == "anneal_release"
    assert at.data == "anneal"
    assert at.lr.kind == "wsd_linear_decay"
    assert at.lr.decay_total_tokens == DECAY_TOKENS
    assert at.lr.peak_lr == cfg.inner.lr.peak_lr

    expected_start = run_state_at(cfg, rig["manifest"], EFFECTIVE_WINDOW, world_size=cfg.model.ep_size)
    assert at.lr.warmup_steps == expected_start.global_inner_step == EFFECTIVE_WINDOW * INNER_STEPS
    assert result.decay_start_step == expected_start.global_inner_step

    # LR closed form: peak at the decay start, strictly decreasing, 0 at budget end.
    step0 = at.lr.warmup_steps
    tps = cfg.tokens_per_inner_step
    assert lr_at(at.lr, step0, tps) == pytest.approx(cfg.inner.lr.peak_lr)
    assert lr_at(at.lr, step0 + 1, tps) < lr_at(at.lr, step0, tps)
    steps_to_zero = -(-DECAY_TOKENS // tps)  # ceil
    assert lr_at(at.lr, step0 + steps_to_zero, tps) == pytest.approx(0.0)


def test_original_manifest_untouched(rig: dict) -> None:
    original = rig["manifest"]
    hash_before = original.manifest_hash()
    dump_before = original.model_dump()

    result = do_fork(rig)

    assert original.manifest_hash() == hash_before
    assert original.model_dump() == dump_before
    assert original.phase_table == (original.phase_table[0],)
    assert original.amendments == ()
    assert result.manifest is not original
    assert result.manifest.manifest_hash() != hash_before


def test_fork_run_id_override(rig: dict) -> None:
    result = do_fork(rig, fork_run_id="release-v0.9")
    assert result.manifest.run_id == "release-v0.9"
    assert rig["manifest"].run_id == "release-fork-test"
    # amendment still intact under the run_id rewrite
    assert result.manifest.amendments[0].kind == "phase"


# --------------------------------------------------------------------------- #
# Preconditions
# --------------------------------------------------------------------------- #


def test_rejects_wrong_config(rig: dict) -> None:
    other = rig["cfg"].model_copy(update={"outer": rig["cfg"].outer.model_copy(update={"lr": 0.5})})
    with pytest.raises(ForkError, match="config_hash"):
        do_fork(rig, cfg=other)


def test_rejects_wrong_lineage_checkpoint(rig: dict, tmp_path: Path) -> None:
    foreign = make_manifest(rig["cfg"]).with_amendment(
        kind="capacity", effective_window=2, committed_block=1, capacity_multiplier=0.5
    )
    with pytest.raises(ForkError, match="lineage"):
        do_fork(rig, manifest=foreign)


def test_rejects_unknown_anneal_dataset(rig: dict) -> None:
    manifest = make_manifest(rig["cfg"], with_anneal=False)
    write_checkpoint_meta(rig["ckpt"], manifest)  # rebind the checkpoint to this manifest
    with pytest.raises(ForkError, match="anneal"):
        do_fork(rig, manifest=manifest)


def test_rejects_effective_window_at_or_before_checkpoint(rig: dict) -> None:
    with pytest.raises(ForkError, match="effective_window"):
        do_fork(rig, effective_window=CKPT_WINDOW)


def test_rejects_nonpositive_decay_tokens(rig: dict) -> None:
    with pytest.raises(ForkError, match="decay_tokens"):
        do_fork(rig, decay_tokens=0)


def test_refuses_to_overwrite_published_fork(rig: dict) -> None:
    do_fork(rig)
    with pytest.raises(FileExistsError):
        do_fork(rig)


def test_missing_checkpoint_meta(rig: dict, tmp_path: Path) -> None:
    with pytest.raises(ForkError, match="meta.json"):
        fork_release(
            tmp_path / "nowhere",
            rig["out"],
            manifest=rig["manifest"],
            cfg=rig["cfg"],
            decay_tokens=DECAY_TOKENS,
            effective_window=EFFECTIVE_WINDOW,
            committed_block=5000,
        )
    assert load_checkpoint_meta(rig["ckpt"]) == rig["meta"]  # the happy path reader


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _write_cli_inputs(rig: dict, tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "current-manifest.json"
    manifest_path.write_bytes(canonical_bytes(rig["manifest"]))
    config_path = tmp_path / "run-config.yaml"
    config_path.write_text(yaml.safe_dump(rig["cfg"].model_dump(mode="json")), encoding="utf-8")
    # the YAML round-trips to the exact config (config_hash gate would catch drift)
    return manifest_path, config_path


def test_cli_forks_and_reports(rig: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, config_path = _write_cli_inputs(rig, tmp_path)
    rc = main(
        [
            "--checkpoint", str(rig["ckpt"]),
            "--out", str(rig["out"]),
            "--manifest", str(manifest_path),
            "--config", str(config_path),
            "--decay-tokens", str(DECAY_TOKENS),
            "--effective-window", str(EFFECTIVE_WINDOW),
            "--committed-block", "5000",
            "--fork-run-id", "release-v0.9",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "manifest_hash" in out and "decay start step" in out

    forked = RunManifest.model_validate_json((rig["out"] / MANIFEST_FILENAME).read_text())
    assert forked.run_id == "release-v0.9"
    assert forked.phase_table[-1].overrides.data == "anneal"
    assert json.loads((rig["out"] / MANIFEST_FILENAME).read_text())["amendments"][0]["kind"] == "phase"


def test_cli_reports_precondition_failures(rig: dict, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path, config_path = _write_cli_inputs(rig, tmp_path)
    rc = main(
        [
            "--checkpoint", str(rig["ckpt"]),
            "--out", str(rig["out"]),
            "--manifest", str(manifest_path),
            "--config", str(config_path),
            "--decay-tokens", "0",  # invalid
            "--effective-window", str(EFFECTIVE_WINDOW),
            "--committed-block", "5000",
        ]
    )
    assert rc == 2
    assert "release fork failed" in capsys.readouterr().err
