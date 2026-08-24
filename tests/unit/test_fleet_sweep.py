"""Tests for fleet/calibration/sweep.py — grid mechanics, timing, YAML emission."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import test_window_runner as twr
import torch
import yaml

from fleet.calibration.local_harness import local_manifest
from fleet.calibration.sweep import (
    SweepPoint,
    SweepResult,
    apply_point,
    default_grid,
    emit_tuned_yaml,
    run_sweep,
    select_best,
)
from mok_core.config import MoKRuntimeConfig
from mok_core.model import build_reference_model

UID = 3

TINY_POINTS = (
    SweepPoint(fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=256),
    SweepPoint(fwd_num_comm_sms=4, bwd_num_comm_sms=4, minibatch_size=512),
)


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("sweep-shards")
    twr.write_shard_files(data_dir)
    index = twr.build_index(data_dir)
    cfg = twr.make_run_cfg()

    def shard_path(i: int) -> Path:
        return data_dir / f"shard-{i}.bin"

    manifest = local_manifest(cfg, index, shard_path=shard_path, run_seed=twr.RUN_SEED)
    template = build_reference_model(twr.make_model_cfg(), twr.SEED)
    return {
        "cfg": cfg,
        "manifest": manifest,
        "shard_path": shard_path,
        "factory": lambda c: copy.deepcopy(template),
    }


# --------------------------------------------------------------------------- #
# Grid + apply_point
# --------------------------------------------------------------------------- #


def test_default_grid_shape() -> None:
    grid = default_grid()
    assert len(grid) == 9
    assert all(p.fwd_num_comm_sms == p.bwd_num_comm_sms for p in grid)
    assert {p.minibatch_size for p in grid} == {2048, 4096, 8192}
    assert grid[4] == SweepPoint(fwd_num_comm_sms=36, bwd_num_comm_sms=36, minibatch_size=4096)


def test_apply_point_revalidates(rig) -> None:
    cfg = apply_point(rig["cfg"], TINY_POINTS[0])
    assert cfg.mok.fwd_num_comm_sms == 2
    assert cfg.mok.minibatch_size == 256
    assert cfg.model == rig["cfg"].model  # everything else untouched
    with pytest.raises(ValueError, match="even"):
        apply_point(rig["cfg"], SweepPoint(fwd_num_comm_sms=3, bwd_num_comm_sms=2, minibatch_size=256))
    with pytest.raises(ValueError, match="256"):
        apply_point(rig["cfg"], SweepPoint(fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=100))


# --------------------------------------------------------------------------- #
# run_sweep on CPU (2 tiny points, fake timer)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def results(rig) -> list[SweepResult]:
    ticks = iter(float(i) for i in range(1000))
    return run_sweep(
        rig["cfg"],
        rig["manifest"],
        model_factory=rig["factory"],
        shard_path=rig["shard_path"],
        points=TINY_POINTS,
        windows_per_point=2,
        uid=UID,
        timer=lambda: next(ticks),
    )


def test_run_sweep_times_each_point(results) -> None:
    assert [r.point for r in results] == list(TINY_POINTS)
    # fake timer ticks once before and once after each window -> mean == 1.0
    assert all(r.mean_window_s == 1.0 for r in results)
    assert all(r.final_loss > 0.0 for r in results)
    assert all(isinstance(r.mok, MoKRuntimeConfig) for r in results)
    # identical model + data -> the training result is point-independent on CPU
    assert results[0].final_loss == results[1].final_loss


def test_select_best_prefers_faster_then_earlier(results) -> None:
    assert select_best(results) is results[0]  # tie -> earlier grid point
    slower = SweepResult(point=results[0].point, mok=results[0].mok, mean_window_s=9.0, final_loss=1.0)
    faster = SweepResult(point=results[1].point, mok=results[1].mok, mean_window_s=0.5, final_loss=1.0)
    assert select_best([slower, faster]) is faster
    with pytest.raises(ValueError):
        select_best([])


def test_run_sweep_validates_inputs(rig) -> None:
    with pytest.raises(ValueError, match="windows_per_point"):
        run_sweep(
            rig["cfg"],
            rig["manifest"],
            model_factory=rig["factory"],
            shard_path=rig["shard_path"],
            points=TINY_POINTS,
            windows_per_point=0,
        )
    with pytest.raises(ValueError, match="empty"):
        run_sweep(
            rig["cfg"],
            rig["manifest"],
            model_factory=rig["factory"],
            shard_path=rig["shard_path"],
            points=(),
        )


# --------------------------------------------------------------------------- #
# emit_tuned_yaml
# --------------------------------------------------------------------------- #


def test_emit_tuned_yaml_is_a_loadable_overlay(results, tmp_path: Path) -> None:
    best = select_best(results)
    out = emit_tuned_yaml(best.mok, tmp_path / "mok_tuned.yaml", provenance="unit-test grid")
    text = out.read_text(encoding="utf-8")
    assert text.startswith("#")
    assert "provenance: unit-test grid" in text
    loaded = yaml.safe_load(text)
    assert set(loaded) == {"mok"}
    assert MoKRuntimeConfig(**loaded["mok"]) == best.mok
