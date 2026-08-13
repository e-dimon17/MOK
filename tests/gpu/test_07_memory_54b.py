"""The full 54B model: meta-device shape audit + guarded real-allocation smoke.

The meta-device test runs anywhere (no CUDA touched) and pins the playbook's
headline arithmetic: 49.3B total / 5.5B active parameters at the canonical
ModelConfig (32 layers, first 3 dense SwiGLU at Id=9216 -- the run keeps the
"MoK-54B" codename), with the EP-8 per-rank split the fleet actually allocates.

The real allocation smoke is OFF by default (`MOK_TEST_54B=1` enables it): it
materializes one rank's 49.3B/EP-8 replica + MXFP8 caches on a B300 and checks
the resident footprint. Budget context (playbook step C, ~208 GB/rank total):
masters ~18 GB (bf16 replicated 2.3B + fp32 lm_head 0.27B + bf16 expert shard
5.8B) + MXFP8 copies ~12 GB (fp8 + transposed fp8 + scales) — the remaining
~175 GB is Adam state, gradients, activations and workspace, which this smoke
does not allocate.
"""

from __future__ import annotations

import os

import _synthetic as synth
import pytest
import torch

from mok_core.config import ModelConfig
from mok_core.model import init_model, is_expert_local

TOTAL_PARAM_RANGE = (48.5e9, 50.0e9)     # "49.3B total" (3 dense + 29 MoE layers)
ACTIVE_PARAM_RANGE = (5.3e9, 5.7e9)      # "5.5B active"
SMOKE_ENV = "MOK_TEST_54B"
SMOKE_BUDGET_BYTES = 60 * 1024**3        # masters + quant caches only (see module docstring)


def _counts(cfg: ModelConfig) -> tuple[int, int, int, int]:
    """(replicated, expert_local, global_total, active) from a meta-device build."""
    model = init_model(cfg, synth.INIT_SEED, device="meta", backend="mok")
    expert_local = 0
    replicated = 0
    for name, tensor in model.iter_master_params():
        if is_expert_local(name):
            expert_local += tensor.numel()
        else:
            replicated += tensor.numel()
    global_total = replicated + expert_local * cfg.ep_size
    # Active per token: everything replicated (attention, shared expert, router,
    # embeddings, LM head, norms) + top_k routed experts' three matrices.
    moe_layers = cfg.num_layers - cfg.num_dense_layers
    per_expert = 3 * cfg.intermediate_size * cfg.hidden_size * moe_layers
    active = replicated + cfg.top_k * per_expert
    return replicated, expert_local, global_total, active


def test_54b_param_count_on_meta_device() -> None:
    cfg = ModelConfig()  # the canonical Stage-2 architecture — defaults ARE the run
    replicated, expert_local, global_total, active = _counts(cfg)

    lo, hi = TOTAL_PARAM_RANGE
    assert lo < global_total < hi, f"global param count {global_total:,} outside 49.3B envelope"
    a_lo, a_hi = ACTIVE_PARAM_RANGE
    assert a_lo < active < a_hi, f"active param count {active:,} outside 5.5B envelope"

    # EP-8 split: each rank hosts 128/8 = 16 experts holding exactly 1/8 of the
    # routed tree, which dominates the model.
    assert cfg.num_local_experts == 16
    per_expert = (
        3 * cfg.intermediate_size * cfg.hidden_size * (cfg.num_layers - cfg.num_dense_layers)
    )
    assert expert_local == cfg.num_local_experts * per_expert
    assert expert_local * cfg.ep_size == global_total - replicated
    assert expert_local * cfg.ep_size > 0.9 * global_total  # the routed tree IS the model
    per_rank = replicated + expert_local
    assert 8.0e9 < per_rank < 9.5e9, f"per-rank master count {per_rank:,} off the EP-8 budget"


def test_54b_meta_build_registers_all_master_names() -> None:
    cfg = ModelConfig()
    model = init_model(cfg, synth.INIT_SEED, device="meta", backend="mok")
    names = [name for name, _ in model.iter_master_params()]
    assert len(names) == len(set(names))
    moe_layers = cfg.num_layers - cfg.num_dense_layers
    assert sum(1 for n in names if n.endswith("balance_bias")) == moe_layers
    assert sum(1 for n in names if is_expert_local(n)) == 3 * moe_layers  # gate/up/down stacks
    shapes = model.param_shapes()
    assert shapes["embed.weight"] == (cfg.vocab_size, cfg.hidden_size)
    # blocks 0..num_dense_layers-1 are dense SwiGLU; the first MoE block follows
    assert shapes["blocks.0.moe.w_gate.weight"] == (cfg.dense_intermediate_size, cfg.hidden_size)
    assert shapes[f"blocks.{cfg.num_dense_layers}.moe.routed_gate"] == (
        16, cfg.intermediate_size, cfg.hidden_size,
    )


@pytest.mark.skipif(os.environ.get(SMOKE_ENV) != "1", reason=f"set {SMOKE_ENV}=1 to run the 54B allocation smoke")
def test_54b_real_allocation_smoke(dist_ctx, mok_available, toy_cfg) -> None:
    """Materialize one rank's 49.3B/EP-8 masters + MXFP8 caches on the B300."""
    cfg = ModelConfig()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dist_ctx.device)
    baseline = torch.cuda.memory_allocated(dist_ctx.device)

    model = init_model(cfg, synth.INIT_SEED, device=dist_ctx.device, backend="mok", mok_runtime=toy_cfg.mok)
    synth.prepare_mok_model(model)  # fp8 + transposed fp8 + scale caches
    torch.cuda.synchronize(dist_ctx.device)

    allocated = torch.cuda.memory_allocated(dist_ctx.device) - baseline
    peak = torch.cuda.max_memory_allocated(dist_ctx.device) - baseline
    print(
        f"[rank {dist_ctx.rank}] 49.3B EP-8 masters+quant: allocated={allocated / 1024**3:.1f} GiB, "
        f"peak={peak / 1024**3:.1f} GiB (budget {SMOKE_BUDGET_BYTES / 1024**3:.0f} GiB; "
        "full training budget ~208 GB/rank incl. Adam/grads/activations)"
    )
    assert allocated > 14 * 1024**3, "49.3B masters unexpectedly small — wrong config?"
    assert allocated < SMOKE_BUDGET_BYTES, (
        f"masters+quant footprint {allocated / 1024**3:.1f} GiB exceeds the {SMOKE_BUDGET_BYTES / 1024**3:.0f} "
        "GiB envelope — the 208 GB/rank training budget will not close"
    )
    del model
    torch.cuda.empty_cache()
    dist_ctx.barrier()
