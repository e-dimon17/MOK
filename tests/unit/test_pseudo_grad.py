"""Tests for C/core/pseudo_grad.py — snapshot/restore round-trip exactness."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from C.core.pseudo_grad import CpuSnapshot, restore_and_extract_delta
from mok_core.config import ModelConfig
from mok_core.determinism import hash_named_tensors
from mok_core.model import build_reference_model


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny-op tests thrash 16-way intra-op parallelism; pin one thread here
    (restored afterwards so other modules keep the session default)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _make_params(seed: int = 0) -> dict[str, nn.Parameter]:
    torch.manual_seed(seed)
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "blocks.0.moe.routed_gate": ((2, 4, 8), torch.bfloat16),
        "blocks.0.moe.router.proj.weight": ((4, 8), torch.float32),
        "embed.weight": ((16, 8), torch.bfloat16),
        "lm_head.weight": ((16, 8), torch.float32),
    }
    return {n: nn.Parameter(torch.randn(s, dtype=dt)) for n, (s, dt) in shapes.items()}


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
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
    )


class TestTake:
    def test_snapshot_is_cpu_clone_in_master_dtype(self) -> None:
        params = _make_params(1)
        snap = CpuSnapshot.take(params)
        assert set(snap.tensors) == set(params)
        assert len(snap) == len(params)
        assert "embed.weight" in snap
        for name, param in params.items():
            saved = snap.tensors[name]
            assert saved.device.type == "cpu"
            assert saved.dtype == param.dtype
            assert saved.shape == param.shape
            assert torch.equal(saved, param.detach())
            assert saved.data_ptr() != param.data_ptr()  # a real copy

    def test_snapshot_frozen_against_later_mutation(self) -> None:
        params = _make_params(2)
        snap = CpuSnapshot.take(params)
        before = {n: t.clone() for n, t in snap.tensors.items()}
        with torch.no_grad():
            for p in params.values():
                p.add_(1.0)
        for n in before:
            assert torch.equal(snap.tensors[n], before[n])

    def test_accepts_iterable_of_pairs(self) -> None:
        params = _make_params(3)
        snap = CpuSnapshot.take(iter(params.items()))
        assert set(snap.tensors) == set(params)

    def test_pin_default_resolves_to_cuda_availability(self) -> None:
        snap = CpuSnapshot.take(_make_params(4))
        assert snap.pinned == torch.cuda.is_available()

    def test_duplicate_names_rejected(self) -> None:
        p = nn.Parameter(torch.zeros(2))
        with pytest.raises(ValueError, match="duplicate"):
            CpuSnapshot.take([("a", p), ("a", p)])

    def test_names_sorted(self) -> None:
        snap = CpuSnapshot.take(_make_params(5))
        assert list(snap.names) == sorted(snap.names)


class TestRestoreAndExtractDelta:
    def test_round_trip_exact_and_delta_correct(self) -> None:
        params = _make_params(6)
        start = {n: p.detach().clone() for n, p in params.items()}
        snap = CpuSnapshot.take(params)
        torch.manual_seed(60)
        with torch.no_grad():
            for p in params.values():
                p.add_(torch.randn_like(p) * 0.1)  # the "inner training" drift
        end = {n: p.detach().clone() for n, p in params.items()}
        live_tensors = {n: params[n] for n in params}

        deltas = restore_and_extract_delta(params, snap)

        for name, param in params.items():
            # DiLoCo restore: θ back to θ_start, bitwise, in place on the same object
            assert params[name] is live_tensors[name]
            assert torch.equal(param.detach(), start[name]), name
            # Δ = θ_start − θ_end, fp32 CPU
            delta = deltas[name]
            assert delta.dtype == torch.float32
            assert delta.device.type == "cpu"
            assert delta.is_contiguous()
            expected = start[name].to(torch.float32) - end[name].to(torch.float32)
            assert torch.equal(delta, expected), name

    def test_unchanged_param_gives_zero_delta(self) -> None:
        params = _make_params(7)
        snap = CpuSnapshot.take(params)
        deltas = restore_and_extract_delta(params, snap)
        for name in params:
            assert torch.equal(deltas[name], torch.zeros_like(deltas[name])), name

    def test_double_round_trip_stable(self) -> None:
        """take -> drift -> restore -> drift -> restore lands on θ_start both times."""
        params = _make_params(8)
        snap = CpuSnapshot.take(params)
        start = {n: p.detach().clone() for n, p in params.items()}
        for seed in (80, 81):
            torch.manual_seed(seed)
            with torch.no_grad():
                for p in params.values():
                    p.add_(torch.randn_like(p))
            restore_and_extract_delta(params, snap)
            for n, p in params.items():
                assert torch.equal(p.detach(), start[n]), (seed, n)

    def test_name_mismatch_rejected(self) -> None:
        params = _make_params(9)
        snap = CpuSnapshot.take(params)
        subset = dict(list(params.items())[:-1])
        with pytest.raises(ValueError, match="mismatch"):
            restore_and_extract_delta(subset, snap)
        extra = dict(params)
        extra["extra.weight"] = nn.Parameter(torch.zeros(2))
        with pytest.raises(ValueError, match="mismatch"):
            restore_and_extract_delta(extra, snap)

    def test_shape_or_dtype_mismatch_rejected(self) -> None:
        params = _make_params(10)
        snap = CpuSnapshot.take(params)
        bad = dict(params)
        bad["embed.weight"] = nn.Parameter(torch.zeros(3, 3, dtype=torch.bfloat16))
        with pytest.raises(ValueError, match="embed.weight"):
            restore_and_extract_delta(bad, snap)
        bad = dict(params)
        bad["embed.weight"] = nn.Parameter(torch.zeros(16, 8, dtype=torch.float32))
        with pytest.raises(ValueError, match="embed.weight"):
            restore_and_extract_delta(bad, snap)


class TestModelStateRoot:
    def test_full_master_domain_round_trip(self) -> None:
        """Snapshot over iter_master_params (incl. balance_bias buffers) restores
        the exact state_root — the property window replay depends on."""
        model = build_reference_model(_tiny_cfg(), seed=9)
        root_before = hash_named_tensors(model.iter_master_params())
        snap = CpuSnapshot.take(model.iter_master_params())
        torch.manual_seed(90)
        with torch.no_grad():
            for _, tensor in model.iter_master_params():
                tensor.add_(torch.randn_like(tensor) * 0.01)
        assert hash_named_tensors(model.iter_master_params()) != root_before
        deltas = restore_and_extract_delta(dict(model.iter_master_params()), snap)
        assert hash_named_tensors(model.iter_master_params()) == root_before
        assert set(deltas) == {name for name, _ in model.iter_master_params()}
        assert any(name.endswith("balance_bias") for name in deltas)
