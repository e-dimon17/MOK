"""THE LAUNCH GATE — same window twice, bitwise identical state roots.

This is the property the whole subnet audits: a window is a pure function of
(θ_start, uid, window, manifest). Two full toy4L windows are run from identical
θ_start on the mok backend (8-rank EP, MXFP8, deterministic attention, pinned
NCCL) and the resulting master-weight state roots must be EQUAL, first through
the raw InnerLoop and then through the full `run_training_phase` (which also
asserts the bitwise θ_start restore internally).

Parametrized over `inner.adam_reset_every_windows` in {1, 5}: 1 is protocol
decision #1 (fresh AdamW per window — InnerLoop always constructs
`Zero1Adam.fresh`), 5 is the calibration fallback knob; determinism must hold
under either configured value.

Do NOT weaken these assertions. If this test fails on a candidate node or
container, the node cannot pass audits — fix the environment, never the test.
"""

from __future__ import annotations

import _synthetic as synth
import pytest
import torch

from subnet.core.phase import resolve_phase
from subnet.core.window_runner import RunState, run_training_phase

pytestmark = pytest.mark.usefixtures("mok_available")


def _cfg(adam_reset: int):
    return synth.load_toy_run_config(adam_reset_every_windows=adam_reset)


def _require_ep_world(dist_ctx, cfg) -> None:
    if dist_ctx.world_size != cfg.model.ep_size:
        pytest.skip(f"toy4L pins ep_size={cfg.model.ep_size}; world_size={dist_ctx.world_size}")


@pytest.mark.parametrize("adam_reset", [1, 5])
def test_inner_loop_window_twice_is_bitwise(dist_ctx, mok_available, toy_dataset, adam_reset) -> None:
    """Full 20-step toy window via InnerLoop, twice, from identical θ_start."""
    cfg = _cfg(adam_reset)
    _require_ep_world(dist_ctx, cfg)

    roots: list[str] = []
    weight_sums: list[float] = []
    for _ in range(2):
        dist_ctx.barrier()  # fresh lockstep entry for each run
        root, model = synth.run_toy_window(
            cfg,
            toy_dataset.manifest,
            toy_dataset.data_dir,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
            device=dist_ctx.device,
            comm=dist_ctx.comm,
        )
        roots.append(dist_ctx.comm.broadcast_object(root, 0))
        # per-rank sanity: the trained weights are finite on this rank's shard
        weight_sums.append(float(next(iter(model.parameters())).float().abs().sum()))
        del model
        torch.cuda.empty_cache()
        dist_ctx.barrier()

    assert roots[0] is not None and len(roots[0]) == 64
    assert roots[0] == roots[1], (
        f"LAUNCH GATE FAILED (adam_reset={adam_reset}): identical windows produced "
        f"different state roots {roots[0]} != {roots[1]} — this node/container is "
        "nondeterministic and cannot pass audits"
    )
    assert all(torch.isfinite(torch.tensor(weight_sums)))


@pytest.mark.parametrize("adam_reset", [1, 5])
def test_run_training_phase_twice_theta_end_roots_equal(
    dist_ctx, mok_available, toy_dataset, adam_reset
) -> None:
    """The window-runner training phase (snapshot -> InnerLoop -> Δ-extract ->
    bitwise θ_start restore) twice: θ_end roots AND θ_start roots must match.

    Run without compression state at world_size 8 (payload building across EP
    ranks needs globally-unique expert shard names — GPU milestone-8 design
    point, see ENGINEERING_NOTES); the replay verdict binds θ_end, not payload
    bytes, so this is exactly what auditors compare.
    """
    cfg = _cfg(adam_reset)
    _require_ep_world(dist_ctx, cfg)
    phase = resolve_phase(toy_dataset.manifest, cfg, synth.WINDOW)
    factory = synth.make_shard_lookup_factory(toy_dataset.data_dir)

    def one_run() -> tuple[str, str]:
        dist_ctx.barrier()
        model = synth.build_mok_model(cfg, dist_ctx.device)
        from subnet.core.window_runner import build_window_plan

        plan = build_window_plan(
            toy_dataset.manifest,
            phase,
            run_seed=synth.RUN_SEED,
            uid=synth.UID,
            window=synth.WINDOW,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )
        with factory(plan) as shard_lookup:
            artifacts = run_training_phase(
                model,
                cfg,
                toy_dataset.manifest,
                phase,
                uid=synth.UID,
                window=synth.WINDOW,
                rank=dist_ctx.rank,
                world_size=dist_ctx.world_size,
                comm=dist_ctx.comm,
                shard_lookup=shard_lookup,
                global_state=RunState(0, 0, 0),
                device=dist_ctx.device,
                plan=plan,
            )
        start = dist_ctx.comm.broadcast_object(artifacts.state_root_start, 0)
        end = dist_ctx.comm.broadcast_object(artifacts.theta_end_root, 0)
        assert artifacts.deltas, "training phase produced no pseudo-gradients"
        del model
        torch.cuda.empty_cache()
        dist_ctx.barrier()
        return start, end

    start_a, end_a = one_run()
    start_b, end_b = one_run()
    assert start_a == start_b, "θ_start roots differ across identical inits — init is nondeterministic"
    assert end_a == end_b, (
        f"LAUNCH GATE FAILED (adam_reset={adam_reset}): θ_end roots differ "
        f"({end_a} != {end_b}) — replay audits would slash an honest miner"
    )
    assert end_a != start_a, "the window did not move the weights — degenerate gate"
