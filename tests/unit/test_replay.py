"""Tests for C/core/replay.py — bitwise audits, fraud detection, consensus sampling.

The miner side runs `run_training_phase` directly (the exact function
`WindowRunner.run_window` calls — the loopback equivalence is pinned in
test_window_runner.py); the auditor side replays it through `WindowReplayer`
on a fresh θ_start replica. Data/model builders are imported from
test_window_runner (same tiny reference-backend rig).
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import test_window_runner as twr
import torch

from C.core.compress import ErrorFeedback
from C.core.inner_loop import InnerLoop
from C.core.phase import resolve_phase
from C.core.replay import (
    AuditReport,
    PreconditionError,
    ReplayTask,
    WindowReplayer,
    audit_sampler,
    report_message,
    sign_report,
    verify_report,
)
from C.core.window_runner import RunState, SingleNodeComm, build_window_plan, run_training_phase
from mok_core.chain.schemas import WindowCommit
from mok_core.config.manifest import RunManifest
from mok_core.determinism import hash_named_tensors, per_tensor_digests
from mok_core.model import MoKTransformer, build_reference_model

UID = twr.UID
WINDOW = twr.WINDOW
RUN_SEED = twr.RUN_SEED
STATE0 = RunState(0, 0, 0)


# --------------------------------------------------------------------------- #
# Fixtures (rig shared with test_window_runner via its builder functions)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def template_model(_single_thread) -> MoKTransformer:
    return build_reference_model(twr.make_model_cfg(), twr.SEED)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shards")
    twr.write_shard_files(root)
    return root


@pytest.fixture(scope="module")
def manifest(data_dir: Path) -> RunManifest:
    return twr.build_manifest(twr.build_index(data_dir))


@pytest.fixture(scope="module")
def miner_run(template_model: MoKTransformer, manifest: RunManifest, data_dir: Path) -> dict:
    """The miner's honest window: run_training_phase from θ_start, with the
    real compression pipeline, producing the on-chain WindowCommit."""
    cfg = twr.make_run_cfg()
    phase = resolve_phase(manifest, cfg, WINDOW)
    model = copy.deepcopy(template_model)
    with twr.make_shard_lookup_factory(data_dir)(
        build_window_plan(manifest, phase, run_seed=RUN_SEED, uid=UID, window=WINDOW, rank=0, world_size=1)
    ) as shard_lookup:
        artifacts = run_training_phase(
            model,
            cfg,
            manifest,
            phase,
            uid=UID,
            window=WINDOW,
            rank=0,
            world_size=1,
            comm=SingleNodeComm(),
            shard_lookup=shard_lookup,
            global_state=STATE0,
            compressor=twr.make_compressor(model, cfg),
            error_feedback=ErrorFeedback(beta=cfg.compression.ef_beta),
        )
    assert artifacts.payload_hash is not None
    commit = WindowCommit(
        window=WINDOW,
        payload_hash=artifacts.payload_hash,
        state_root=artifacts.state_root_start,
        theta_end_hash=artifacts.theta_end_root,
    )
    return {"artifacts": artifacts, "commit": commit, "cfg": cfg, "phase": phase}


def make_replayer(
    model: MoKTransformer, manifest: RunManifest, data_dir: Path, auditor_uid: int = 9
) -> WindowReplayer:
    return WindowReplayer(
        model,
        twr.make_run_cfg(),
        manifest,
        comm=SingleNodeComm(),
        shard_lookup_factory=twr.make_shard_lookup_factory(data_dir),
        auditor_uid=auditor_uid,
    )


# --------------------------------------------------------------------------- #
# (2) Replay matches an honest miner bitwise
# --------------------------------------------------------------------------- #


def test_replay_matches_honest_commit(
    miner_run: dict, template_model: MoKTransformer, manifest: RunManifest, data_dir: Path
) -> None:
    commit: WindowCommit = miner_run["commit"]
    replica = copy.deepcopy(template_model)
    replayer = make_replayer(replica, manifest, data_dir)

    report = replayer.replay(
        ReplayTask(miner_uid=UID, window=WINDOW, commit=commit), global_state=STATE0
    )
    assert report.match
    assert report.replayed_theta_end == commit.theta_end_hash  # BITWISE
    assert report.committed_theta_end == commit.theta_end_hash
    assert report.theta_start_root == commit.state_root
    assert report.divergences == []
    assert report.wall_time_s > 0.0
    assert report.signature == ""

    # the auditor's replica is untouched — a second replay still passes
    assert hash_named_tensors(replica.iter_master_params()) == commit.state_root
    report2 = replayer.replay(
        ReplayTask(miner_uid=UID, window=WINDOW, commit=commit), global_state=STATE0
    )
    assert report2.match


def test_replay_precondition_rejects_wrong_theta_start(
    miner_run: dict, template_model: MoKTransformer, manifest: RunManifest, data_dir: Path
) -> None:
    commit: WindowCommit = miner_run["commit"]
    replica = copy.deepcopy(template_model)
    with torch.no_grad():
        replica.embed.weight.view(-1)[0] += 1.0  # replica is NOT at θ_start
    replayer = make_replayer(replica, manifest, data_dir)
    with pytest.raises(PreconditionError, match="state_root"):
        replayer.replay(ReplayTask(miner_uid=UID, window=WINDOW, commit=commit))

    bad_window = ReplayTask(miner_uid=UID, window=WINDOW + 1, commit=commit)
    with pytest.raises(PreconditionError, match="window"):
        make_replayer(copy.deepcopy(template_model), manifest, data_dir).replay(bad_window)


# --------------------------------------------------------------------------- #
# (3) Replay catches fraud
# --------------------------------------------------------------------------- #


def test_replay_catches_tampered_weight(
    miner_run: dict, template_model: MoKTransformer, manifest: RunManifest, data_dir: Path
) -> None:
    """A miner that does the work but commits a doctored θ_end (one weight
    element edited — 'fake work') is caught, and the divergence report names
    exactly the tampered tensor."""
    cfg = miner_run["cfg"]
    phase = miner_run["phase"]

    # the fraudulent miner's θ_end: honest inner loop, then one edited element
    model_f = copy.deepcopy(template_model)
    plan = build_window_plan(
        manifest, phase, run_seed=RUN_SEED, uid=UID, window=WINDOW, rank=0, world_size=1
    )
    with twr.make_shard_lookup_factory(data_dir)(plan) as shard_lookup:
        InnerLoop(
            model_f, cfg, phase, rank=0, world_size=1, comm=SingleNodeComm(), device="cpu"
        ).run_window(plan, shard_lookup, WINDOW, 0, 0)
    with torch.no_grad():
        model_f.embed.weight.view(-1)[0] += 1.0
    fraud_digests = per_tensor_digests(model_f.iter_master_params())
    fraud_root = hash_named_tensors(model_f.iter_master_params())
    assert fraud_root != miner_run["commit"].theta_end_hash

    fraud_commit = WindowCommit(
        window=WINDOW,
        payload_hash="ab" * 32,
        state_root=miner_run["commit"].state_root,  # θ_start is genuine
        theta_end_hash=fraud_root,                  # θ_end is not
    )
    replayer = make_replayer(copy.deepcopy(template_model), manifest, data_dir)
    report = replayer.replay(
        ReplayTask(miner_uid=UID, window=WINDOW, commit=fraud_commit),
        global_state=STATE0,
        expected_digests=fraud_digests,
    )
    assert not report.match
    assert report.replayed_theta_end == miner_run["commit"].theta_end_hash  # honest replay
    assert report.divergences  # non-empty evidence
    assert [d["name"] for d in report.divergences] == ["embed.weight"]
    assert report.divergences[0]["expected"] != report.divergences[0]["actual"]


def test_replay_mismatch_without_digests_reports_root(
    miner_run: dict, template_model: MoKTransformer, manifest: RunManifest, data_dir: Path
) -> None:
    fraud_commit = WindowCommit(
        window=WINDOW,
        payload_hash="ab" * 32,
        state_root=miner_run["commit"].state_root,
        theta_end_hash="ff" * 32,
    )
    report = make_replayer(copy.deepcopy(template_model), manifest, data_dir).replay(
        ReplayTask(miner_uid=UID, window=WINDOW, commit=fraud_commit), global_state=STATE0
    )
    assert not report.match
    assert report.divergences == [
        {
            "name": "<state_root>",
            "expected": "ff" * 32,
            "actual": miner_run["commit"].theta_end_hash,
        }
    ]


# --------------------------------------------------------------------------- #
# (4) audit_sampler — goldens + partition properties
# --------------------------------------------------------------------------- #

_BH = bytes([1]) * 32


def test_audit_sampler_golden() -> None:
    # consensus constant — change requires SPEC_VERSION bump
    got = audit_sampler(RUN_SEED, _BH, 42, list(range(20)), 0.25, [100, 7, 55])
    assert got == {
        7: [(2, 42), (17, 42)],
        55: [(4, 42), (18, 42)],
        100: [(14, 42)],
    }
    # a different window reshuffles the draw
    assert audit_sampler(RUN_SEED, _BH, 43, list(range(20)), 0.25, [100, 7, 55]) == {
        7: [(3, 43)],
        55: [(6, 43)],
        100: [(7, 43)],
    }


def test_audit_sampler_partition_properties() -> None:
    miners = list(range(64))
    auditors = [9, 4, 30]
    a = audit_sampler(RUN_SEED, _BH, 7, miners, 0.3, auditors)
    b = audit_sampler(RUN_SEED, _BH, 7, list(reversed(miners)), 0.3, list(reversed(auditors)))
    assert a == b  # deterministic and order-insensitive in its inputs

    assigned = [task for tasks in a.values() for task in tasks]
    uids = [uid for uid, w in assigned]
    assert len(uids) == len(set(uids))  # each sampled miner audited by exactly one auditor
    assert all(w == 7 for _, w in assigned)
    assert set(a) == set(auditors)
    # round-robin balance: sizes differ by at most one, largest lists first (sorted auditors)
    sizes = [len(a[x]) for x in sorted(a)]
    assert max(sizes) - min(sizes) <= 1
    assert sizes == sorted(sizes, reverse=True)


def test_audit_sampler_edges() -> None:
    assert audit_sampler(RUN_SEED, _BH, 3, [5, 1, 9], 0.0, [2]) == {2: []}
    everyone = audit_sampler(RUN_SEED, _BH, 3, [5, 1, 9], 1.0, [2])
    assert everyone == {2: [(1, 3), (5, 3), (9, 3)]}  # sorted-uid deal order
    assert audit_sampler(RUN_SEED, _BH, 3, [], 0.5, [2]) == {2: []}
    with pytest.raises(ValueError, match="rho"):
        audit_sampler(RUN_SEED, _BH, 3, [1], 1.5, [2])
    with pytest.raises(ValueError, match="auditor"):
        audit_sampler(RUN_SEED, _BH, 3, [1], 0.5, [])


# --------------------------------------------------------------------------- #
# Report signing
# --------------------------------------------------------------------------- #


def _report() -> AuditReport:
    return AuditReport(
        miner_uid=UID,
        window=WINDOW,
        theta_start_root="aa" * 32,
        committed_theta_end="bb" * 32,
        replayed_theta_end="cc" * 32,
        match=False,
        divergences=[{"name": "embed.weight", "expected": "00", "actual": "01"}],
        wall_time_s=1.5,
        auditor_uid=9,
    )


def _keyed_sign(key: bytes):
    def sign(msg: bytes) -> bytes:
        return hashlib.blake2b(msg, key=key, digest_size=32).digest()

    def verify(msg: bytes, sig: bytes) -> bool:
        return hashlib.blake2b(msg, key=key, digest_size=32).digest() == sig

    return sign, verify


def test_sign_and_verify_report_roundtrip() -> None:
    sign, verify = _keyed_sign(b"auditor-key")
    signed = sign_report(_report(), sign)
    assert signed.signature and signed.signature != _report().signature
    assert verify_report(signed, verify)
    # the signature covers the unsigned fields only — message is stable
    assert report_message(signed) == report_message(_report())
    # wire dict round-trip (what exchange.put_audit_report publishes) verifies too
    assert verify_report(signed.to_json(), verify)


def test_verify_report_rejects_tampering() -> None:
    sign, verify = _keyed_sign(b"auditor-key")
    signed = sign_report(_report(), sign)
    tampered = signed.to_json()
    tampered["match"] = True
    assert not verify_report(tampered, verify)
    assert not verify_report(_report(), verify)  # unsigned
    garbage = signed.to_json()
    garbage["signature"] = "zz-not-hex"
    assert not verify_report(garbage, verify)
    _, other_verify = _keyed_sign(b"other-key")
    assert not verify_report(signed, other_verify)
