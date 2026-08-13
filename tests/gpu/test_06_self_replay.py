"""GPU twin of the CPU replay gate: miner window -> same-node bitwise replay.

A miner's window runs through `run_training_phase` on the mok backend (8-rank
EP), producing the on-chain commitment (θ_start root, θ_end root). A fresh
replica at the same θ_start then replays the window through `WindowReplayer`
— THE SAME function auditors run — and must report `match=True` with the
replayed root equal to the committed θ_end bitwise. A tampered commitment
must yield `match=False` and, when fraudulent per-tensor digests are supplied,
a divergence report naming exactly the tampered tensor.
"""

from __future__ import annotations

import _synthetic as synth
import pytest
import torch

from C.core.phase import resolve_phase
from C.core.replay import PreconditionError, ReplayTask, WindowReplayer
from C.core.window_runner import RunState, run_training_phase, shared_master_root
from mok_core.chain.schemas import WindowCommit

pytestmark = pytest.mark.usefixtures("mok_available")

MINER_UID = synth.UID
AUDITOR_UID = 42


@pytest.fixture(scope="module")
def miner_window(dist_ctx, mok_available, toy_cfg, toy_dataset):
    """One honest miner window: artifacts + the WindowCommit it would sign."""
    if dist_ctx.world_size != toy_cfg.model.ep_size:
        pytest.skip(f"toy4L pins ep_size={toy_cfg.model.ep_size}; world_size={dist_ctx.world_size}")
    cfg = synth.load_toy_run_config(inner_steps=5)  # short window: replay runs it twice more
    phase = resolve_phase(toy_dataset.manifest, cfg, synth.WINDOW)
    from C.core.window_runner import build_window_plan

    plan = build_window_plan(
        toy_dataset.manifest,
        phase,
        run_seed=synth.RUN_SEED,
        uid=MINER_UID,
        window=synth.WINDOW,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
    )
    model = synth.build_mok_model(cfg, dist_ctx.device)
    with synth.make_shard_lookup_factory(toy_dataset.data_dir)(plan) as shard_lookup:
        artifacts = run_training_phase(
            model,
            cfg,
            toy_dataset.manifest,
            phase,
            uid=MINER_UID,
            window=synth.WINDOW,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            comm=dist_ctx.comm,
            shard_lookup=shard_lookup,
            global_state=RunState(0, 0, 0),
            device=dist_ctx.device,
            plan=plan,
        )
    state_root = dist_ctx.comm.broadcast_object(artifacts.state_root_start, 0)
    theta_end = dist_ctx.comm.broadcast_object(artifacts.theta_end_root, 0)
    commit = WindowCommit(
        window=synth.WINDOW,
        payload_hash="0" * 64,  # replay verdicts bind θ_end, not payload bytes
        state_root=state_root,
        theta_end_hash=theta_end,
    )
    del model
    torch.cuda.empty_cache()
    dist_ctx.barrier()
    return {"cfg": cfg, "artifacts": artifacts, "commit": commit}


def _replayer(dist_ctx, cfg, toy_dataset, replica) -> WindowReplayer:
    return WindowReplayer(
        replica,
        cfg,
        toy_dataset.manifest,
        comm=dist_ctx.comm,
        shard_lookup_factory=synth.make_shard_lookup_factory(toy_dataset.data_dir),
        auditor_uid=AUDITOR_UID,
        rank=dist_ctx.rank,
        world_size=dist_ctx.world_size,
        device=dist_ctx.device,
    )


def test_self_replay_matches(dist_ctx, toy_dataset, miner_window) -> None:
    cfg, commit = miner_window["cfg"], miner_window["commit"]
    replica = synth.build_mok_model(cfg, dist_ctx.device)  # same seed => θ_start bitwise
    replayer = _replayer(dist_ctx, cfg, toy_dataset, replica)

    report = replayer.replay(
        ReplayTask(miner_uid=MINER_UID, window=synth.WINDOW, commit=commit),
        global_state=RunState(0, 0, 0),
    )
    assert report.match, (
        f"REPLAY GATE FAILED: replayed θ_end {report.replayed_theta_end} != "
        f"committed {report.committed_theta_end} on the same node — audits would slash honest miners"
    )
    assert report.replayed_theta_end == commit.theta_end_hash
    assert report.theta_start_root == commit.state_root
    assert report.divergences == []
    assert report.auditor_uid == AUDITOR_UID and report.miner_uid == MINER_UID
    assert report.wall_time_s > 0.0

    # the auditor's replica is left untouched at θ_start...
    root_after = dist_ctx.comm.broadcast_object(
        shared_master_root(replica, rank=dist_ctx.rank, world_size=dist_ctx.world_size, comm=dist_ctx.comm),
        0,
    )
    assert root_after == commit.state_root
    # ...so a second replay from the same replica also passes.
    second = replayer.replay(
        ReplayTask(miner_uid=MINER_UID, window=synth.WINDOW, commit=commit),
        global_state=RunState(0, 0, 0),
    )
    assert second.match
    del replica
    torch.cuda.empty_cache()
    dist_ctx.barrier()


def test_tampered_commit_yields_mismatch_with_named_divergence(dist_ctx, toy_dataset, miner_window) -> None:
    """A fraudulent θ_end commitment is caught; fraudulent per-tensor digests
    pin the exact tensor. This is the slash-evidence path."""
    cfg, artifacts = miner_window["cfg"], miner_window["artifacts"]
    honest = miner_window["commit"]
    fraudulent = honest.model_copy(update={"theta_end_hash": "f" * 64})

    # Tamper THIS rank's first owned θ_end digest — rank-local audit evidence.
    tampered_digests = dict(artifacts.theta_end_digests)
    tampered_name = sorted(tampered_digests)[0]
    tampered_digests[tampered_name] = b"\x00" * 32

    replica = synth.build_mok_model(cfg, dist_ctx.device)
    replayer = _replayer(dist_ctx, cfg, toy_dataset, replica)
    report = replayer.replay(
        ReplayTask(miner_uid=MINER_UID, window=synth.WINDOW, commit=fraudulent),
        global_state=RunState(0, 0, 0),
        expected_digests=tampered_digests,
    )
    assert not report.match
    assert report.replayed_theta_end == honest.theta_end_hash  # the replay itself was honest
    assert [d["name"] for d in report.divergences] == [tampered_name]

    # replica still at θ_start even after a mismatch verdict
    root_after = dist_ctx.comm.broadcast_object(
        shared_master_root(replica, rank=dist_ctx.rank, world_size=dist_ctx.world_size, comm=dist_ctx.comm),
        0,
    )
    assert root_after == honest.state_root
    del replica
    torch.cuda.empty_cache()
    dist_ctx.barrier()


def test_replay_refuses_wrong_theta_start(dist_ctx, toy_dataset, miner_window) -> None:
    """A replica NOT at the committed θ_start must be refused (PreconditionError)
    — auditors run catch_up first, never replay from the wrong lineage point."""
    cfg, commit = miner_window["cfg"], miner_window["commit"]
    replica = synth.build_mok_model(cfg, dist_ctx.device)
    with torch.no_grad():  # one flipped element => wrong θ_start root
        replica.embed.weight[0, 0] += 1.0
    replayer = _replayer(dist_ctx, cfg, toy_dataset, replica)
    with pytest.raises(PreconditionError, match="state_root"):
        replayer.replay(
            ReplayTask(miner_uid=MINER_UID, window=synth.WINDOW, commit=commit),
            global_state=RunState(0, 0, 0),
        )
    del replica
    torch.cuda.empty_cache()
    dist_ctx.barrier()
