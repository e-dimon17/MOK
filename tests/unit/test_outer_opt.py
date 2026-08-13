"""Tests for C/core/outer_opt.py — deterministic merge math and outer-step lockstep."""

from __future__ import annotations

import copy

import pytest
import torch

from C.core.outer_opt import (
    ReplicatedOuterStep,
    deterministic_segment_mean,
    median_norm_clip_factors,
)
from mok_core.config import OuterOptConfig
from mok_core.determinism import tensor_bytes

# --------------------------------------------------------------------------- #
# deterministic_segment_mean
# --------------------------------------------------------------------------- #


def _dense_reference_mean(
    indices: list[torch.Tensor], values: list[torch.Tensor], numel: int
) -> torch.Tensor:
    sums = [0.0] * numel
    counts = [0] * numel
    for idx_t, val_t in zip(indices, values, strict=True):
        for i, v in zip(idx_t.tolist(), val_t.tolist(), strict=True):
            sums[i] += v
            counts[i] += 1
    return torch.tensor(
        [s / c if c else 0.0 for s, c in zip(sums, counts, strict=True)], dtype=torch.float32
    )


def _random_sparse(seed: int, n_peers: int, numel: int, nnz: int):
    g = torch.Generator().manual_seed(seed)
    indices = [torch.randint(0, numel, (nnz,), generator=g, dtype=torch.int64) for _ in range(n_peers)]
    values = [torch.randn(nnz, generator=g, dtype=torch.float32) for _ in range(n_peers)]
    return indices, values


def test_segment_mean_matches_dense_reference():
    indices, values = _random_sparse(seed=7, n_peers=3, numel=50, nnz=40)  # heavy duplication
    out = deterministic_segment_mean(indices, values, 50)
    ref = _dense_reference_mean(indices, values, 50)
    assert out.dtype == torch.float32
    assert out.shape == (50,)
    assert torch.allclose(out, ref, atol=1e-6)


def test_segment_mean_bitwise_repeatable():
    indices, values = _random_sparse(seed=11, n_peers=5, numel=257, nnz=100)
    a = deterministic_segment_mean(indices, values, 257)
    b = deterministic_segment_mean([i.clone() for i in indices], [v.clone() for v in values], 257)
    assert torch.equal(a, b)
    assert tensor_bytes(a) == tensor_bytes(b)


def test_segment_mean_untouched_indices_are_zero():
    out = deterministic_segment_mean([torch.tensor([1, 3])], [torch.tensor([2.0, 4.0])], 6)
    assert torch.equal(out, torch.tensor([0.0, 2.0, 0.0, 4.0, 0.0, 0.0]))


def test_segment_mean_empty_inputs():
    assert torch.equal(deterministic_segment_mean([], [], 4), torch.zeros(4))
    out = deterministic_segment_mean([torch.empty(0, dtype=torch.int64)], [torch.empty(0)], 4)
    assert torch.equal(out, torch.zeros(4))


def test_segment_mean_accepts_int32_indices():
    out = deterministic_segment_mean(
        [torch.tensor([0, 0], dtype=torch.int32)], [torch.tensor([1.0, 3.0])], 2
    )
    assert torch.equal(out, torch.tensor([2.0, 0.0]))


def test_segment_mean_validation():
    with pytest.raises(ValueError):
        deterministic_segment_mean([torch.tensor([4])], [torch.tensor([1.0])], 4)  # out of range
    with pytest.raises(ValueError):
        deterministic_segment_mean([torch.tensor([-1])], [torch.tensor([1.0])], 4)
    with pytest.raises(ValueError):
        deterministic_segment_mean([torch.tensor([0, 1])], [torch.tensor([1.0])], 4)  # length mismatch
    with pytest.raises(ValueError):
        deterministic_segment_mean([torch.tensor([0])], [], 4)  # peer count mismatch
    with pytest.raises(ValueError):
        deterministic_segment_mean([], [], 0)  # bad numel


# --------------------------------------------------------------------------- #
# median_norm_clip_factors
# --------------------------------------------------------------------------- #


def test_clip_factors_basic():
    factors = median_norm_clip_factors(torch.tensor([1.0, 2.0, 4.0]))
    assert torch.allclose(factors, torch.tensor([1.0, 1.0, 0.5]))


def test_clip_factors_even_count_uses_lower_median():
    factors = median_norm_clip_factors(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # median -> 2.0
    assert torch.allclose(factors, torch.tensor([1.0, 1.0, 2.0 / 3.0, 0.5]))


def test_clip_factors_zero_norm_guard():
    factors = median_norm_clip_factors(torch.tensor([0.0, 3.0, 3.0]))
    assert torch.allclose(factors, torch.tensor([1.0, 1.0, 1.0]))
    assert torch.isfinite(factors).all()


def test_clip_factors_empty():
    assert median_norm_clip_factors(torch.empty(0)).numel() == 0


# --------------------------------------------------------------------------- #
# ReplicatedOuterStep
# --------------------------------------------------------------------------- #


def _full_coverage_peer(g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.arange(g.numel(), dtype=torch.int64), g.reshape(-1).clone()


def test_nesterov_two_step_hand_computed():
    cfg = OuterOptConfig(kind="nesterov", lr=0.7, momentum=0.9, clip="median_norm")
    g = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    step = ReplicatedOuterStep(cfg, {"w": torch.Size([4])})
    p = {"w": torch.zeros(4, dtype=torch.float32)}
    sparse = {"w": [_full_coverage_peer(g)]}
    norms = {"w": torch.tensor([float(torch.linalg.vector_norm(g))])}  # single peer: factor 1

    r1 = step.apply(p, sparse, {}, norms)
    # m1 = g; d1 = g + 0.9*m1 = 1.9 g; p1 = -0.7 * 1.9 g
    assert torch.allclose(p["w"], -0.7 * 1.9 * g, rtol=1e-6)
    assert torch.allclose(step.state_dict()["w"], g, rtol=1e-6)
    assert r1.applied_peers == 1
    assert r1.per_param_l2["w"] == pytest.approx(float(torch.linalg.vector_norm(g)))
    assert r1.global_grad_l2 == pytest.approx(float(torch.linalg.vector_norm(g)))

    step.apply(p, sparse, {}, norms)
    # m2 = 0.9 g + g = 1.9 g; d2 = g + 0.9*m2 = 2.71 g; p2 = p1 - 0.7 * 2.71 g
    assert torch.allclose(p["w"], -0.7 * (1.9 + 2.71) * g, rtol=1e-5)
    assert torch.allclose(step.state_dict()["w"], 1.9 * g, rtol=1e-6)


def test_plain_sgd_leaves_momentum_untouched():
    cfg = OuterOptConfig(kind="sgd", lr=0.5, momentum=0.9, clip="none")
    g = torch.tensor([1.0, -2.0], dtype=torch.float32)
    step = ReplicatedOuterStep(cfg, {"w": torch.Size([2])})
    p = {"w": torch.zeros(2)}
    step.apply(p, {"w": [_full_coverage_peer(g)]}, {}, {})
    assert torch.allclose(p["w"], -0.5 * g)
    assert torch.equal(step.state_dict()["w"], torch.zeros(2))


def test_median_clip_applied_to_peer_values():
    cfg = OuterOptConfig(kind="sgd", lr=1.0, momentum=0.0, clip="median_norm")
    step = ReplicatedOuterStep(cfg, {"w": torch.Size([2])})
    p = {"w": torch.zeros(2)}
    # Peers hit disjoint indices; norms [1, 3] -> lower median 1 -> factors [1, 1/3].
    sparse = {
        "w": [
            (torch.tensor([0]), torch.tensor([1.0])),
            (torch.tensor([1]), torch.tensor([3.0])),
        ]
    }
    step.apply(p, sparse, {}, {"w": torch.tensor([1.0, 3.0])})
    assert torch.allclose(p["w"], torch.tensor([-1.0, -1.0]))  # 3.0 clipped to 1.0


def test_dense_contribs_mean_with_clip():
    cfg = OuterOptConfig(kind="sgd", lr=1.0, momentum=0.0, clip="median_norm")
    step = ReplicatedOuterStep(cfg, {"bias": torch.Size([2])})
    p = {"bias": torch.zeros(2)}
    dense = {"bias": [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 8.0])]}
    # norms [2, 8] -> lower median 2 -> factors [1, 0.25]; mean of [2,0] and [0,2] = [1,1]
    step.apply(p, {}, dense, {"bias": torch.tensor([2.0, 8.0])})
    assert torch.allclose(p["bias"], torch.tensor([-1.0, -1.0]))


def test_bf16_param_updated_via_fp32_math():
    cfg = OuterOptConfig(kind="sgd", lr=1.0, momentum=0.0, clip="none")
    step = ReplicatedOuterStep(cfg, {"w": torch.Size([2])})
    p = {"w": torch.tensor([1.0, 1.0], dtype=torch.bfloat16)}
    g = torch.tensor([0.25, -0.25])
    step.apply(p, {"w": [_full_coverage_peer(g)]}, {}, {})
    assert p["w"].dtype == torch.bfloat16
    assert torch.equal(p["w"].float(), torch.tensor([0.75, 1.25]))  # exact in bf16


def test_untouched_param_left_alone():
    cfg = OuterOptConfig()
    step = ReplicatedOuterStep(cfg, {"a": torch.Size([2]), "b": torch.Size([2])})
    p = {"a": torch.ones(2), "b": torch.ones(2)}
    g = torch.tensor([1.0, 1.0])
    report = step.apply(p, {"a": [_full_coverage_peer(g)]}, {}, {"a": torch.tensor([1.0])})
    assert torch.equal(p["b"], torch.ones(2))
    assert "b" not in report.per_param_l2


def test_apply_validation_errors():
    cfg = OuterOptConfig()
    step = ReplicatedOuterStep(cfg, {"w": torch.Size([2])})
    g = torch.ones(2)
    with pytest.raises(KeyError):
        step.apply({"unknown": torch.zeros(2)}, {"unknown": [_full_coverage_peer(g)]}, {}, {})
    with pytest.raises(ValueError):
        step.apply({"w": torch.zeros(3)}, {"w": [_full_coverage_peer(g)]}, {}, {})  # shape mismatch
    with pytest.raises(ValueError):  # median clip needs norms
        step.apply({"w": torch.zeros(2)}, {"w": [_full_coverage_peer(g)]}, {}, {})
    with pytest.raises(ValueError):  # norm count != peer count
        step.apply(
            {"w": torch.zeros(2)}, {"w": [_full_coverage_peer(g)]}, {}, {"w": torch.tensor([1.0, 2.0])}
        )


# --------------------------------------------------------------------------- #
# Lockstep + state_dict
# --------------------------------------------------------------------------- #

_SHAPES = {"layer.weight": torch.Size([8, 4]), "layer.bias": torch.Size([8]), "router.w": torch.Size([4])}


def _window_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    peer_sparse = {
        "layer.weight": [
            (
                torch.randint(0, 32, (12,), generator=g, dtype=torch.int64),
                torch.randn(12, generator=g),
            )
            for _ in range(3)
        ],
        "router.w": [
            (torch.randint(0, 4, (3,), generator=g, dtype=torch.int64), torch.randn(3, generator=g))
            for _ in range(3)
        ],
    }
    dense = {"layer.bias": [torch.randn(8, generator=g) for _ in range(3)]}
    norms = {
        name: torch.rand(3, generator=g) + 0.5
        for name in ("layer.weight", "router.w", "layer.bias")
    }
    return peer_sparse, dense, norms


def _fresh_params(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {name: torch.randn(shape, generator=g) for name, shape in _SHAPES.items()}


def _assert_byte_identical(a: ReplicatedOuterStep, b: ReplicatedOuterStep, pa, pb):
    for name in _SHAPES:
        assert tensor_bytes(pa[name]) == tensor_bytes(pb[name]), name
    sa, sb = a.state_dict(), b.state_dict()
    for name in _SHAPES:
        assert tensor_bytes(sa[name]) == tensor_bytes(sb[name]), name


def test_lockstep_two_instances_byte_identical():
    cfg = OuterOptConfig(kind="nesterov", lr=0.7, momentum=0.9, clip="median_norm")
    step_a = ReplicatedOuterStep(cfg, dict(_SHAPES))
    step_b = ReplicatedOuterStep(cfg, dict(_SHAPES))
    params_a, params_b = _fresh_params(), _fresh_params()
    for window in range(3):
        inputs = _window_inputs(seed=100 + window)
        report_a = step_a.apply(params_a, *copy.deepcopy(inputs))
        report_b = step_b.apply(params_b, *copy.deepcopy(inputs))
        _assert_byte_identical(step_a, step_b, params_a, params_b)
        assert report_a == report_b


def test_state_dict_round_trip_resumes_lockstep():
    cfg = OuterOptConfig(kind="nesterov", lr=0.7, momentum=0.9, clip="median_norm")
    step_a = ReplicatedOuterStep(cfg, dict(_SHAPES))
    params_a = _fresh_params()
    step_a.apply(params_a, *_window_inputs(seed=1))

    # Cold-restarted instance: load momentum + copy params, then both take window 2.
    step_b = ReplicatedOuterStep(cfg, dict(_SHAPES))
    step_b.load_state_dict(step_a.state_dict())
    params_b = {name: t.clone() for name, t in params_a.items()}
    step_a.apply(params_a, *_window_inputs(seed=2))
    step_b.apply(params_b, *_window_inputs(seed=2))
    _assert_byte_identical(step_a, step_b, params_a, params_b)


def test_load_state_dict_validation():
    step = ReplicatedOuterStep(OuterOptConfig(), {"w": torch.Size([2])})
    with pytest.raises(ValueError):
        step.load_state_dict({})  # missing key
    with pytest.raises(ValueError):
        step.load_state_dict({"w": torch.zeros(3)})  # wrong shape
    with pytest.raises(ValueError):
        step.load_state_dict({"w": torch.zeros(2), "extra": torch.zeros(1)})
