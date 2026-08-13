"""Integration test for C/core/inner_loop.py — the CPU determinism gate.

Builds a REAL data path (uint16 shard files -> DatasetShardIndex -> manifest ->
WindowBatchPlan) and a tiny reference-backend model, then runs full windows on
CPU. The two load-bearing assertions:

  1. run_window twice from identical θ_start -> IDENTICAL state_root
     (bitwise window replay — the property the whole subnet audits), and
  2. final_loss < entry_loss (the window actually trains).

Synthetic data is a learnable 4-token cycle (3 -> 5 -> 7 -> 11 -> 3) with a
per-shard phase shift plus one unique constant row per shard (keeps every
shard's bytes — and hence Merkle leaves — distinct).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from C.core.inner_loop import InnerLoop, WindowResult
from C.core.phase import PhaseConfig, resolve_phase
from C.core.zero1 import SingleProcessComm
from mok_core.config import InnerOptConfig, LRSpec, ModelConfig, RunConfig, WindowConfig
from mok_core.config.manifest import DatasetManifestRef, PRFSpec, RunManifest
from mok_core.data import (
    DatasetShardIndex,
    ShardReader,
    WindowBatchPlan,
    shard_leaf_hash,
    verify_index_matches_ref,
)
from mok_core.determinism import hash_named_tensors, per_tensor_digests
from mok_core.model import MoKTransformer, build_reference_model

SEED = 9
RUN_SEED = bytes(range(32))
SEQ_LEN = 256
SEQS_PER_SHARD = 8
NUM_SHARDS = 8
VOCAB = 512
CYCLE = (3, 5, 7, 11)
UID = 3
WINDOW = 5
INNER_STEPS = 3
GRAD_ACCUM = 2
TOKENS_PER_MB = 512  # 2 sequences of 256


def _shard_array(shard_idx: int) -> np.ndarray:
    rows = []
    for r in range(SEQS_PER_SHARD):
        if r == SEQS_PER_SHARD - 1:
            rows.append(np.full(SEQ_LEN, 100 + shard_idx, dtype="<u2"))  # unique bytes per shard
        else:
            phase = (shard_idx + r) % len(CYCLE)
            rows.append(
                np.array([CYCLE[(phase + j) % len(CYCLE)] for j in range(SEQ_LEN)], dtype="<u2")
            )
    return np.stack(rows)


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        num_dense_layers=0,
        hidden_size=512,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        vocab_size=VOCAB,
        seq_len=SEQ_LEN,
        num_experts=8,
        top_k=2,
        intermediate_size=256,
        ep_size=4,
    )


def _run_cfg() -> RunConfig:
    return RunConfig(
        model=_model_cfg(),
        window=WindowConfig(
            inner_steps=INNER_STEPS,
            tokens_per_rank_microbatch=TOKENS_PER_MB,
            grad_accum=GRAD_ACCUM,
            accum_ramp_start=GRAD_ACCUM,  # no ramp: full accum from token 0
        ),
        inner=InnerOptConfig(lr=LRSpec(kind="const", const_lr=0.02)),
    )


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny CPU windows thrash 16-way intra-op parallelism; pin one thread here
    (restored afterwards so other modules keep the session default). Thread
    count is fixed for the whole module, so the determinism assertions compare
    like with like."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def template_model(_single_thread) -> MoKTransformer:
    """One deterministic θ_start for the whole module; runs take deepcopies —
    'identical model clones', exactly what window replay starts from."""
    return build_reference_model(_model_cfg(), SEED)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shards")
    for i in range(NUM_SHARDS):
        (root / f"shard-{i}.bin").write_bytes(_shard_array(i).tobytes())
    return root


@pytest.fixture(scope="module")
def manifest(data_dir: Path) -> RunManifest:
    hashes = [shard_leaf_hash(data_dir / f"shard-{i}.bin").hex() for i in range(NUM_SHARDS)]
    assert len(set(hashes)) == NUM_SHARDS  # every shard's bytes are distinct
    index = DatasetShardIndex(name="bulk", seq_len=SEQ_LEN, shard_hashes=hashes)
    ref = DatasetManifestRef(
        name="bulk",
        merkle_root=index.merkle().root.hex(),
        num_shards=NUM_SHARDS,
        shard_bytes=2 * SEQ_LEN * SEQS_PER_SHARD,
        seq_len=SEQ_LEN,
        tokens_total=NUM_SHARDS * SEQS_PER_SHARD * SEQ_LEN,
        tokenizer_hash="ab" * 32,
    )
    verify_index_matches_ref(index, ref)  # the real gate a miner runs at startup
    return RunManifest(
        spec_version=1,
        run_id="innerloop-test",
        netuid=11,
        network="test",
        config_hash="11" * 32,
        container_digest="sha256:" + "22" * 32,
        mok_commit="deadbeef",
        tk_commit="cafebabe",
        attention_backend="cudnn_det",
        start_block=100,
        blocks_per_window=225,
        prf=PRFSpec(run_seed_hex=RUN_SEED.hex()),
        datasets=(ref,),
        init_checkpoint_hash="33" * 32,
    )


@pytest.fixture(scope="module")
def plan(manifest: RunManifest) -> WindowBatchPlan:
    return WindowBatchPlan.build(
        manifest,
        run_seed=RUN_SEED,
        uid=UID,
        window=WINDOW,
        rank=0,
        world_size=1,
        tokens_per_rank_microbatch=TOKENS_PER_MB,
        grad_accum=GRAD_ACCUM,
        inner_steps=INNER_STEPS,
        seq_len=SEQ_LEN,
        dataset="bulk",
    )


@pytest.fixture(scope="module")
def readers(data_dir: Path, plan: WindowBatchPlan) -> Iterator[dict[int, ShardReader]]:
    lookup = {i: ShardReader(data_dir / f"shard-{i}.bin", SEQ_LEN) for i in plan.shard_ids}
    yield lookup
    for reader in lookup.values():
        reader.close()


def _phase(manifest: RunManifest, cfg: RunConfig) -> PhaseConfig:
    return resolve_phase(manifest, cfg, WINDOW)


def _run_once(
    template_model: MoKTransformer,
    manifest: RunManifest,
    plan: WindowBatchPlan,
    readers: dict[int, ShardReader],
    *,
    null_round: bool = False,
) -> tuple[MoKTransformer, WindowResult]:
    cfg = _run_cfg()
    model = copy.deepcopy(template_model)  # bitwise clone of θ_start
    loop = InnerLoop(
        model,
        cfg,
        _phase(manifest, cfg),
        rank=0,
        world_size=1,
        comm=SingleProcessComm(),
        device="cpu",
    )
    result = loop.run_window(
        plan,
        readers.__getitem__,
        WINDOW,
        global_inner_step0=0,
        tokens_consumed0=0,
        null_round=null_round,
    )
    return model, result


# --------------------------------------------------------------------------- #
# THE determinism gate
# --------------------------------------------------------------------------- #


def test_same_window_twice_identical_state_root_and_loss_decreases(
    template_model: MoKTransformer,
    manifest: RunManifest,
    plan: WindowBatchPlan,
    readers: dict[int, ShardReader],
) -> None:
    model_a, res_a = _run_once(template_model, manifest, plan, readers)
    model_b, res_b = _run_once(template_model, manifest, plan, readers)

    root_a = hash_named_tensors(model_a.iter_master_params())
    root_b = hash_named_tensors(model_b.iter_master_params())
    assert root_a == root_b  # bitwise window replay — the engine's core property

    # the window moved the weights (it is not vacuously identical)
    init_root = hash_named_tensors(template_model.iter_master_params())
    assert root_a != init_root

    # telemetry is deterministic too
    assert res_a.entry_loss == res_b.entry_loss
    assert res_a.mean_loss == res_b.mean_loss
    assert res_a.final_loss == res_b.final_loss
    assert res_a.grad_norm_mean == res_b.grad_norm_mean
    assert res_a.router_entropy_mean == res_b.router_entropy_mean
    assert torch.equal(res_a.expert_load, res_b.expert_load)

    # the window trains
    assert res_a.final_loss < res_a.entry_loss


# --------------------------------------------------------------------------- #
# accounting + telemetry shape
# --------------------------------------------------------------------------- #


def test_window_accounting(
    template_model: MoKTransformer,
    manifest: RunManifest,
    plan: WindowBatchPlan,
    readers: dict[int, ShardReader],
) -> None:
    _, res = _run_once(template_model, manifest, plan, readers)
    assert res.tokens == INNER_STEPS * GRAD_ACCUM * TOKENS_PER_MB  # world_size == 1
    assert res.global_inner_steps_done == INNER_STEPS
    assert res.expert_load.dtype == torch.int64
    assert res.expert_load.shape == (_model_cfg().num_experts,)
    # every routed token counted: T * top_k per layer per microbatch
    expected_assignments = INNER_STEPS * GRAD_ACCUM * TOKENS_PER_MB * 2 * _model_cfg().num_layers
    assert int(res.expert_load.sum()) == expected_assignments
    assert res.grad_norm_mean > 0.0
    assert res.router_entropy_mean > 0.0
    # cfg.model.ep_size = 4 grouping: util in [1/cm, ep/cm] with cm = 1.0
    # trap-relative util: (0, 1]; ~1/factor at perfect balance, 1.0 == the trap
    assert 0.0 < res.capacity_util_max <= 1.0
    assert res.entry_loss > 0.0
    assert min(res.entry_loss, res.final_loss) <= res.mean_loss <= max(res.entry_loss, res.final_loss)


def test_global_step_offset_changes_nothing_at_const_lr(
    template_model: MoKTransformer,
    manifest: RunManifest,
    plan: WindowBatchPlan,
    readers: dict[int, ShardReader],
) -> None:
    cfg = _run_cfg()
    model = copy.deepcopy(template_model)
    loop = InnerLoop(
        model, cfg, _phase(manifest, cfg), rank=0, world_size=1, comm=SingleProcessComm(), device="cpu"
    )
    res = loop.run_window(plan, readers.__getitem__, WINDOW, 1000, 12345)
    assert res.global_inner_steps_done == 1000 + INNER_STEPS
    assert res.tokens == INNER_STEPS * GRAD_ACCUM * TOKENS_PER_MB


# --------------------------------------------------------------------------- #
# null rounds
# --------------------------------------------------------------------------- #


def test_null_round_moves_only_balance_bias(
    template_model: MoKTransformer,
    manifest: RunManifest,
    plan: WindowBatchPlan,
    readers: dict[int, ShardReader],
) -> None:
    model, res = _run_once(template_model, manifest, plan, readers, null_round=True)
    before = per_tensor_digests(template_model.iter_master_params())
    after = per_tensor_digests(model.iter_master_params())
    for name in before:
        if name.endswith("balance_bias"):
            continue  # MoeHealth still nudges biases in a null round
        assert after[name] == before[name], name
    # the full hot path still ran and produced telemetry
    assert res.entry_loss > 0.0
    assert res.tokens == INNER_STEPS * GRAD_ACCUM * TOKENS_PER_MB
    assert res.grad_norm_mean > 0.0
    # no optimizer step -> per-step losses only move via bias-driven routing,
    # so the window cannot have trained
    assert res.final_loss == pytest.approx(res.entry_loss, rel=0.10)


# --------------------------------------------------------------------------- #
# plan validation
# --------------------------------------------------------------------------- #


def test_plan_mismatch_rejected(
    template_model: MoKTransformer,
    manifest: RunManifest,
    plan: WindowBatchPlan,
    readers: dict[int, ShardReader],
) -> None:
    cfg = _run_cfg()
    model = copy.deepcopy(template_model)
    loop = InnerLoop(
        model, cfg, _phase(manifest, cfg), rank=0, world_size=1, comm=SingleProcessComm(), device="cpu"
    )
    with pytest.raises(ValueError, match="plan.window"):
        loop.run_window(plan, readers.__getitem__, WINDOW + 1, 0, 0)

    other_plan = WindowBatchPlan.build(
        manifest,
        run_seed=RUN_SEED,
        uid=UID,
        window=WINDOW,
        rank=0,
        world_size=1,
        tokens_per_rank_microbatch=TOKENS_PER_MB,
        grad_accum=GRAD_ACCUM,
        inner_steps=INNER_STEPS + 1,
        seq_len=SEQ_LEN,
        dataset="bulk",
    )
    with pytest.raises(ValueError, match="plan.inner_steps"):
        loop.run_window(other_plan, readers.__getitem__, WINDOW, 0, 0)


def test_rank_out_of_range_rejected(
    template_model: MoKTransformer, manifest: RunManifest
) -> None:
    cfg = _run_cfg()
    with pytest.raises(ValueError, match="rank"):
        InnerLoop(
            template_model,
            cfg,
            _phase(manifest, cfg),
            rank=1,
            world_size=1,
            comm=SingleProcessComm(),
            device="cpu",
        )
