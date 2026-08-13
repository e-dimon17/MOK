"""Tests for C/core/moe_health.py — bias nudge direction, util formula, requant no-op."""

from __future__ import annotations

import copy
import sys

import pytest
import torch

from C.core.moe_health import MoeHealth
from mok_core.config import ModelConfig, MoKRuntimeConfig, RunConfig
from mok_core.model import init_model

E = 16
EP = 4
RATE = 1e-3


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny-op tests thrash 16-way intra-op parallelism; pin one thread here
    (restored afterwards so other modules keep the session default)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _model_cfg(num_experts: int = E) -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        num_dense_layers=0,
        hidden_size=256,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=128,
        vocab_size=512,
        seq_len=256,
        num_experts=num_experts,
        top_k=2,
        intermediate_size=256,
        ep_size=EP,
        bias_update_rate=RATE,
    )


def _run_cfg(capacity_multiplier: float = 1.0) -> RunConfig:
    return RunConfig(
        model=_model_cfg(),
        mok=MoKRuntimeConfig(schedule_capacity_multiplier=capacity_multiplier),
    )


@pytest.fixture(scope="module")
def template(_single_thread):
    # Built with the PROTOCOL ep_size (not the reference ep=1): MoeHealth never
    # runs a forward, and the router/balance_bias geometry is what matters here.
    return init_model(_model_cfg(), seed=3, backend="reference")


@pytest.fixture
def model(template):
    return copy.deepcopy(template)  # pristine zero biases for every test


def _balanced_load(total_per_expert: int = 8) -> torch.Tensor:
    return torch.full((E,), total_per_expert, dtype=torch.int64)


class TestBiasUpdate:
    def test_sign_rule_direction(self, model) -> None:
        health = MoeHealth(model, _run_cfg())
        hot = torch.zeros(E, dtype=torch.int64)
        hot[0] = 100  # expert 0 overloaded, everyone else starved
        health.post_step([hot, _balanced_load()])
        bias0 = model.moe_layers()[0].router.balance_bias
        # b -= rate * sign(load - mean): overloaded down, starved up — exactly one nudge
        assert float(bias0[0]) == pytest.approx(-RATE)
        assert torch.allclose(bias0[1:], torch.full((E - 1,), RATE))

    def test_balanced_load_leaves_bias_zero(self, model) -> None:
        MoeHealth(model, _run_cfg()).post_step([_balanced_load(), _balanced_load()])
        for layer in model.moe_layers():
            assert torch.equal(layer.router.balance_bias, torch.zeros(E))

    def test_layers_updated_independently(self, model) -> None:
        hot = torch.zeros(E, dtype=torch.int64)
        hot[3] = 64
        MoeHealth(model, _run_cfg()).post_step([_balanced_load(), hot])
        layer0, layer1 = model.moe_layers()
        assert torch.equal(layer0.router.balance_bias, torch.zeros(E))
        assert float(layer1.router.balance_bias[3]) == pytest.approx(-RATE)

    def test_nudges_accumulate_across_steps(self, model) -> None:
        health = MoeHealth(model, _run_cfg())
        hot = torch.zeros(E, dtype=torch.int64)
        hot[0] = 100
        for _ in range(3):
            health.post_step([hot, _balanced_load()])
        assert float(model.moe_layers()[0].router.balance_bias[0]) == pytest.approx(-3 * RATE)


class TestCapacityUtil:
    """util = max_r rows_r / (tokens_per_launch * top_k * max(2, ceil(ep*cm)) * M).

    All tests pin tokens_per_launch=16 so one launch dispatches
    16 * EP * top_k = 128 rows: balanced load = 8/expert = 32 rows/rank."""

    T = 16  # tokens_per_launch; capacity at cm=1.0: 16 * 2 * 4 = 128 rows

    def _health(self, model, cm: float, override: float | None = None) -> MoeHealth:
        return MoeHealth(
            model, _run_cfg(capacity_multiplier=cm),
            capacity_multiplier=override, tokens_per_launch=self.T,
        )

    def test_perfect_balance_util_is_inverse_factor(self, model) -> None:
        health = self._health(model, cm=1.0)  # factor = max(2, ceil(4*1.0)) = 4
        util = health.post_step([_balanced_load(), _balanced_load()])
        assert util == pytest.approx(32 / 128)  # 1/factor

    def test_one_rank_takes_all_hits_the_trap(self, model) -> None:
        """A full launch's 128 rows all onto rank 0 -> util == 1.0 (the GPU trap)."""
        load = torch.zeros(E, dtype=torch.int64)
        load[: E // EP] = 32  # experts 0..3 live on rank 0: 128 rows
        health = self._health(model, cm=1.0)
        assert health.post_step([load, _balanced_load()]) == pytest.approx(1.0)

    def test_multiplier_scales_capacity(self, model) -> None:
        health = self._health(model, cm=0.5)  # factor = max(2, ceil(2)) = 2; capacity 64
        assert health.post_step([_balanced_load(), _balanced_load()]) == pytest.approx(32 / 64)

    def test_exact_formula_on_skewed_load(self, model) -> None:
        # per-rank sums: [10+2+0+0, 4+4+0+0, 8+0+0+0, 0+0+0+4] = [12, 8, 8, 4]
        load = torch.tensor([10, 2, 0, 0, 4, 4, 0, 0, 8, 0, 0, 0, 0, 0, 0, 4], dtype=torch.int64)
        health = self._health(model, cm=1.0)
        zero = torch.zeros(E, dtype=torch.int64)
        assert health.post_step([load, zero]) == pytest.approx(12 / 128)

    def test_microbatches_scale_capacity(self, model) -> None:
        health = self._health(model, cm=1.0)
        util = health.post_step([_balanced_load(), _balanced_load()], microbatches=2)
        assert util == pytest.approx(32 / 256)

    def test_zero_load_gives_zero_util(self, model) -> None:
        zero = torch.zeros(E, dtype=torch.int64)
        health = self._health(model, cm=1.0)
        assert health.post_step([zero, zero]) == pytest.approx(0.0)

    def test_max_util_and_alert_threshold(self, model) -> None:
        health = self._health(model, cm=1.0)
        health.post_step([_balanced_load(), _balanced_load()])
        assert health.max_util == pytest.approx(0.25)
        assert health.capacity_alert(0.25)
        assert not health.capacity_alert(0.4)  # the anneal threshold, not yet reached
        hot = torch.zeros(E, dtype=torch.int64)
        hot[: E // EP] = 32
        health.post_step([hot, _balanced_load()])
        assert health.max_util == pytest.approx(1.0)  # running max sticks
        health.post_step([_balanced_load(), _balanced_load()])
        assert health.max_util == pytest.approx(1.0)
        assert health.capacity_alert(0.4)

    def test_phase_override_beats_cfg_default(self, model) -> None:
        health = self._health(model, cm=1.0, override=0.5)
        assert health.capacity_multiplier == 0.5
        assert health.post_step([_balanced_load(), _balanced_load()]) == pytest.approx(0.5)


class TestReferenceRequantNoOp:
    def test_no_mok_import_and_no_cache_on_reference(self, model) -> None:
        assert model.backend == "reference"
        health = MoeHealth(model, _run_cfg())
        health.post_step([_balanced_load(), _balanced_load()])
        assert all(layer.quant_cache is None for layer in model.moe_layers())
        assert "mok" not in sys.modules  # the SM103-only wheel never loaded


class TestValidation:
    def test_wrong_layer_count(self, model) -> None:
        health = MoeHealth(model, _run_cfg())
        with pytest.raises(ValueError, match="per-layer"):
            health.post_step([_balanced_load()])

    def test_wrong_load_shape(self, model) -> None:
        health = MoeHealth(model, _run_cfg())
        with pytest.raises(ValueError, match="layer 0"):
            health.post_step([torch.zeros(E + 1, dtype=torch.int64), _balanced_load()])

    def test_nonpositive_capacity_multiplier(self, model) -> None:
        with pytest.raises(ValueError, match="capacity_multiplier"):
            MoeHealth(model, _run_cfg(), capacity_multiplier=0.0)

    def test_expert_count_mismatch(self, model) -> None:
        cfg8 = RunConfig(model=_model_cfg(num_experts=8))
        with pytest.raises(ValueError, match="experts"):
            MoeHealth(model, cfg8)
