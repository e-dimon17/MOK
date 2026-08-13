"""Tests for B/calibration/rehearsal.py — the full-protocol loopback dress
rehearsal on CPU (real WindowRunner over MemoryStorage + ScriptedChain)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import test_window_runner as twr
import torch

from B.calibration.local_harness import local_manifest
from B.calibration.rehearsal import CalibrationReport, run_calibration_windows
from mok_core.model import MoKTransformer, build_reference_model
from mok_core.storage import keys

UID = 3


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
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("rehearsal-shards")
    twr.write_shard_files(root)
    return root


@pytest.fixture(scope="module")
def rig(data_dir: Path):
    index = twr.build_index(data_dir)
    cfg = twr.make_run_cfg()

    def shard_path(i: int) -> Path:
        return data_dir / f"shard-{i}.bin"

    manifest = local_manifest(cfg, index, shard_path=shard_path, run_seed=twr.RUN_SEED)
    return {"index": index, "cfg": cfg, "manifest": manifest, "shard_path": shard_path}


@pytest.fixture(scope="module")
def report(rig, template_model, tmp_path_factory) -> dict:
    model = copy.deepcopy(template_model)
    work = tmp_path_factory.mktemp("rehearsal-work")
    rep = run_calibration_windows(
        2,
        rig["cfg"],
        rig["manifest"],
        model=model,
        index=rig["index"],
        shard_path=rig["shard_path"],
        work_dir=work,
        uid=UID,
    )
    return {"report": rep, "model": model, "work": work}


def test_rehearsal_completes_and_trains(report) -> None:
    rep: CalibrationReport = report["report"]
    assert rep.ok
    assert rep.windows == (0, 1)
    assert len(rep.loss_curve) == len(rep.window_wall_times) == len(rep.capacity_utils) == 2
    assert all(loss > 0.0 for loss in rep.loss_curve)
    assert all(t > 0.0 for t in rep.window_wall_times)
    # the model actually moved: consecutive post-outer roots differ
    assert rep.state_roots[0] != rep.state_roots[1]


def test_rehearsal_determinism_check_passes(report) -> None:
    assert report["report"].determinism_check is True


def test_rehearsal_ran_the_full_protocol(report, rig) -> None:
    """The storage holds the two-phase artifacts: payloads + certificates +
    debug slices, and a window-0 checkpoint was pruned/saved locally."""
    work: Path = report["work"]
    bucket_root = work / "storage" / f"loopback-uid{UID:05d}"
    for window in (0, 1):
        assert (bucket_root / keys.payload_key(window, UID, "1")).is_file()
        assert (bucket_root / keys.certificate_key(window)).is_file()
        assert (bucket_root / f"debug/w{window:08d}/uid{UID:05d}.json").is_file()
    # checkpoint_every_windows=2 -> window 0 checkpointed locally
    assert (work / "ckpt" / "w00000000" / "meta.json").is_file()


def test_rehearsal_wall_times_use_injected_timer(rig, template_model, tmp_path: Path) -> None:
    ticks = iter(float(i) for i in range(100))
    rep = run_calibration_windows(
        1,
        rig["cfg"],
        rig["manifest"],
        model=copy.deepcopy(template_model),
        index=rig["index"],
        shard_path=rig["shard_path"],
        work_dir=tmp_path,
        uid=UID,
        timer=lambda: next(ticks),
        determinism_probe=False,
    )
    assert rep.window_wall_times == (1.0,)  # t1 - t0
    assert rep.determinism_check is True  # probe skipped -> vacuously true


def test_rehearsal_is_reproducible_from_identical_theta(rig, template_model, tmp_path, report) -> None:
    """A second rehearsal from the same θ_start reproduces window 0's root
    bitwise — the loopback stack (storage, certificate, outer step) is
    deterministic end to end."""
    rep = run_calibration_windows(
        1,
        rig["cfg"],
        rig["manifest"],
        model=copy.deepcopy(template_model),
        index=rig["index"],
        shard_path=rig["shard_path"],
        work_dir=tmp_path,
        uid=UID,
        determinism_probe=False,
    )
    assert rep.state_roots[0] == report["report"].state_roots[0]


def test_rehearsal_rejects_nonpositive_windows(rig, template_model, tmp_path) -> None:
    with pytest.raises(ValueError, match="n_windows"):
        run_calibration_windows(
            0,
            rig["cfg"],
            rig["manifest"],
            model=copy.deepcopy(template_model),
            index=rig["index"],
            shard_path=rig["shard_path"],
            work_dir=tmp_path,
        )
