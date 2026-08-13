"""Tests for B/calibration/adam_ab.py — the Adam-reset A/B.

The load-bearing pin: the injected-optimizer window loop with reset_every=1
reproduces the REAL ``InnerLoop.run_window`` bitwise (state-root equality), so
the A/B compares optimizer lifetimes and nothing else.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import test_window_runner as twr
import torch

from B.calibration.adam_ab import ABReport, run_adam_ab, run_arm
from B.calibration.local_harness import local_manifest
from C.core.inner_loop import InnerLoop
from C.core.phase import resolve_phase
from C.core.window_runner import build_window_plan
from C.core.zero1 import SingleProcessComm
from mok_core.data import ShardReader
from mok_core.determinism import hash_named_tensors
from mok_core.model import build_reference_model

UID = 3


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("ab-shards")
    twr.write_shard_files(data_dir)
    index = twr.build_index(data_dir)
    cfg = twr.make_run_cfg()

    def shard_path(i: int) -> Path:
        return data_dir / f"shard-{i}.bin"

    manifest = local_manifest(cfg, index, shard_path=shard_path, run_seed=twr.RUN_SEED)
    template = build_reference_model(twr.make_model_cfg(), twr.SEED)
    return {"cfg": cfg, "manifest": manifest, "shard_path": shard_path, "template": template}


def test_reset1_arm_is_bitwise_the_real_inner_loop(rig) -> None:
    """run_arm(reset_every=1) over one window == InnerLoop.run_window, bitwise."""
    cfg, manifest = rig["cfg"], rig["manifest"]

    arm_model = copy.deepcopy(rig["template"])
    run_arm(
        arm_model, cfg, manifest, n_windows=1, reset_every=1, shard_path=rig["shard_path"], uid=UID
    )

    loop_model = copy.deepcopy(rig["template"])
    phase = resolve_phase(manifest, cfg, 0)
    plan = build_window_plan(
        manifest, phase, run_seed=twr.RUN_SEED, uid=UID, window=0, rank=0, world_size=1
    )
    readers = {i: ShardReader(rig["shard_path"](i), phase.seq_len) for i in set(plan.shard_ids)}
    try:
        InnerLoop(
            loop_model, cfg, phase, rank=0, world_size=1, comm=SingleProcessComm(), device="cpu"
        ).run_window(plan, readers.__getitem__, 0, global_inner_step0=0, tokens_consumed0=0)
    finally:
        for reader in readers.values():
            reader.close()

    assert hash_named_tensors(arm_model.iter_master_params()) == hash_named_tensors(
        loop_model.iter_master_params()
    )


def test_arms_share_window0_then_diverge(rig) -> None:
    """Window 0 is identical in both arms (fresh optimizer either way); from
    window 1 on, the K-arm's persistent Adam state changes the trajectory."""
    a = run_arm(
        copy.deepcopy(rig["template"]), rig["cfg"], rig["manifest"],
        n_windows=3, reset_every=1, shard_path=rig["shard_path"], uid=UID,
    )
    b = run_arm(
        copy.deepcopy(rig["template"]), rig["cfg"], rig["manifest"],
        n_windows=3, reset_every=3, shard_path=rig["shard_path"], uid=UID,
    )
    assert a[0] == b[0]
    assert a[1:] != b[1:]


def test_run_arm_is_deterministic(rig) -> None:
    kwargs = {
        "n_windows": 2, "reset_every": 2, "shard_path": rig["shard_path"], "uid": UID,
    }
    a = run_arm(copy.deepcopy(rig["template"]), rig["cfg"], rig["manifest"], **kwargs)
    b = run_arm(copy.deepcopy(rig["template"]), rig["cfg"], rig["manifest"], **kwargs)
    assert a == b


def test_run_adam_ab_report_and_recommendation_rule(rig) -> None:
    report = run_adam_ab(
        2,
        rig["cfg"],
        rig["manifest"],
        model=rig["template"],
        shard_path=rig["shard_path"],
        k=2,
        uid=UID,
    )
    assert isinstance(report, ABReport)
    assert report.n_windows == 2 and report.k == 2
    assert len(report.losses_reset_every_window) == len(report.losses_reset_every_k) == 2
    assert report.delta_final_loss == pytest.approx(
        report.losses_reset_every_window[-1] - report.losses_reset_every_k[-1]
    )
    # decision rule: delta < threshold  <=>  keep reset=1
    assert report.keep_reset_every_window == (report.delta_final_loss < report.threshold_nats)
    expected = (
        "inner.adam_reset_every_windows=1"
        if report.keep_reset_every_window
        else "inner.adam_reset_every_windows=2"
    )
    assert report.recommendation == expected
    # the input model is never mutated (both arms run on deepcopies)
    assert hash_named_tensors(rig["template"].iter_master_params()) == hash_named_tensors(
        build_reference_model(twr.make_model_cfg(), twr.SEED).iter_master_params()
    )


def test_threshold_flips_the_recommendation(rig) -> None:
    base = run_adam_ab(
        1, rig["cfg"], rig["manifest"], model=rig["template"], shard_path=rig["shard_path"],
        k=2, uid=UID, threshold_nats=float("inf"),
    )
    assert base.keep_reset_every_window
    flipped = run_adam_ab(
        1, rig["cfg"], rig["manifest"], model=rig["template"], shard_path=rig["shard_path"],
        k=2, uid=UID, threshold_nats=float("-inf"),
    )
    assert not flipped.keep_reset_every_window
    assert flipped.recommendation == "inner.adam_reset_every_windows=2"


def test_run_adam_ab_validates_inputs(rig) -> None:
    with pytest.raises(ValueError, match="n_windows"):
        run_adam_ab(0, rig["cfg"], rig["manifest"], model=rig["template"],
                    shard_path=rig["shard_path"])
    with pytest.raises(ValueError, match="k must be"):
        run_adam_ab(1, rig["cfg"], rig["manifest"], model=rig["template"],
                    shard_path=rig["shard_path"], k=1)
