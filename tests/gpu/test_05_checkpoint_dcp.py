"""Multi-rank DCP checkpoint round-trip + catch_up on GPU tensors.

Every rank of the node saves its OWN master shard (replicated tensors + its EP
expert block) through the Checkpointer's DCP path and reloads it bitwise —
`state_root` stable across save/load is what makes checkpoints valid θ_start
lineage points. Then `catch_up` replays ONE fabricated certified window over
CUDA-resident masters (exchange/chain doubles, no network) and must be
deterministic: same certified inputs, same post-outer-step root, twice.

No `mok` dependency: checkpointing and the outer step never touch the kernel.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import _synthetic as synth
import pytest
import torch

from C.core.checkpoint import Checkpointer, CheckpointMeta
from C.core.compress import ChunkingTransformer, Quantizer, TopKCompressor
from C.core.exchange import CertifiedGather
from C.core.outer_opt import ReplicatedOuterStep
from C.core.payload import PayloadMeta, WindowPayload
from C.core.window_runner import DENSE_SUFFIX
from mok_core.config import RunConfig
from mok_core.config.schemas import BucketCreds
from mok_core.determinism import hash_named_tensors

PEER_UID = 7


@pytest.fixture(scope="module")
def gpu_model(dist_ctx, toy_cfg: RunConfig):
    """One deterministic mok-backend master set per rank (no kernel calls)."""
    from mok_core.model import init_model

    return init_model(
        toy_cfg.model, synth.INIT_SEED, device=dist_ctx.device, backend="mok", mok_runtime=toy_cfg.mok
    )


# --------------------------------------------------------------------------- #
# DCP round-trip
# --------------------------------------------------------------------------- #


def test_dcp_save_load_round_trip_all_ranks(dist_ctx, gpu_model, toy_cfg, tmp_path: Path) -> None:
    """All 8 ranks save concurrently (per-rank shard dirs — the node checkpoint
    layout until milestone-8 makes EP shard names globally unique), reload, and
    verify every tensor bitwise + the state_root unchanged."""
    live = dict(gpu_model.iter_master_params())
    live_root = hash_named_tensors(live.items())
    meta = CheckpointMeta(
        window=synth.WINDOW,
        global_step=1,
        tokens_consumed=1024,
        state_root=live_root,
        manifest_hash="ab" * 32,
        spec_version=1,
    )
    ckpt = Checkpointer(None, tmp_path / f"ckpt-rank{dist_ctx.rank}")
    ckpt.save_local(synth.WINDOW, live, outer_state={}, meta=meta)
    dist_ctx.barrier()  # everyone finished writing before anyone asserts

    state, outer_state, loaded_meta = ckpt.load_local(synth.WINDOW)
    assert loaded_meta == meta
    assert outer_state == {}
    assert set(state) == set(live)
    for name, loaded in state.items():
        expected = live[name].detach().cpu()
        assert loaded.dtype == expected.dtype, f"{name}: dtype changed across DCP"
        assert torch.equal(loaded, expected), f"{name}: bytes changed across DCP round-trip"
    assert hash_named_tensors(state.items()) == live_root  # state_root stable
    dist_ctx.barrier()


def test_dcp_reload_into_fresh_replica_restores_root(dist_ctx, gpu_model, toy_cfg, tmp_path: Path) -> None:
    """Loading a checkpoint into a fresh (differently-seeded) replica restores
    the exact saved root — the catch-up bootstrap path."""
    from mok_core.model import init_model

    live = dict(gpu_model.iter_master_params())
    live_root = hash_named_tensors(live.items())
    ckpt = Checkpointer(None, tmp_path / f"reload-rank{dist_ctx.rank}")
    meta = CheckpointMeta(
        window=synth.WINDOW,
        global_step=1,
        tokens_consumed=1024,
        state_root=live_root,
        manifest_hash="ab" * 32,
        spec_version=1,
    )
    ckpt.save_local(synth.WINDOW, live, outer_state={}, meta=meta)

    replica = init_model(
        toy_cfg.model, synth.INIT_SEED + 1, device=dist_ctx.device, backend="mok", mok_runtime=toy_cfg.mok
    )
    replica_params = dict(replica.iter_master_params())
    assert hash_named_tensors(replica_params.items()) != live_root  # genuinely different θ

    state, _, _ = ckpt.load_local(synth.WINDOW)
    with torch.no_grad():
        for name, tensor in state.items():
            replica_params[name].copy_(tensor.to(replica_params[name].device))
    assert hash_named_tensors(replica.iter_master_params()) == live_root
    dist_ctx.barrier()


# --------------------------------------------------------------------------- #
# catch_up over one fabricated certified window, on GPU tensors
# --------------------------------------------------------------------------- #


def _fabricate_certified_window(
    model_params: dict[str, torch.Tensor], cfg: RunConfig, window: int, theta_start_root: str
) -> tuple[Any, CertifiedGather]:
    """A deterministic one-peer certified window: seeded pseudo-gradients,
    compressed with the run's own compression config."""
    from C.core.certificate import WindowCertificate

    comp_names = sorted(n for n in model_params if not n.endswith(DENSE_SUFFIX))
    compressor = TopKCompressor(
        ChunkingTransformer(
            {n: model_params[n].shape for n in comp_names}, target_chunk=cfg.compression.target_chunk
        ),
        Quantizer(bins=cfg.compression.quant_bins, range_sigmas=cfg.compression.quant_range_sigmas),
        topk=cfg.compression.topk,
    )
    generator = torch.Generator(device="cpu").manual_seed(1234)  # same on every rank/run
    compressed = {}
    for name in comp_names:
        delta = torch.randn(model_params[name].shape, generator=generator, dtype=torch.float32) * 1e-3
        compressed[name] = compressor.compress(name, delta)
    dense = {
        n: torch.randn(model_params[n].shape, generator=generator, dtype=torch.float32) * 1e-4
        for n in sorted(model_params)
        if n.endswith(DENSE_SUFFIX)
    }
    payload = WindowPayload(
        uid=PEER_UID,
        window=window,
        compressed=compressed,
        dense=dense,
        metadata=PayloadMeta(
            sample_digest="00" * 32,
            sample_count=1,
            theta_end_hash="11" * 32,
            state_root=theta_start_root,
            global_step=window,
            spec_version=1,
        ),
    )
    cert = WindowCertificate(
        window=window,
        included_uids=(PEER_UID,),
        payload_hashes={PEER_UID: "22" * 32},
        theta_start_root=theta_start_root,
        leader_uid=0,
    )
    gather = CertifiedGather(payloads=OrderedDict({PEER_UID: payload}), missing={})
    return cert, gather


class _FakeExchange:
    def __init__(self, cert: Any, gather: CertifiedGather) -> None:
        self.cert = cert
        self.gather = gather

    async def get_certificate(self, storage: Any, bucket: Any, window: int) -> Any:
        assert window == self.cert.window
        return self.cert

    async def gather_from_aggregator(self, storage: Any, cert: Any, bucket: Any, **kwargs: Any) -> Any:
        assert cert is self.cert
        return self.gather


def test_catch_up_one_fabricated_window_on_gpu_tensors(dist_ctx, gpu_model, toy_cfg, toy_dataset) -> None:
    from C.core.checkpoint import catch_up

    model_params = dict(gpu_model.iter_master_params())
    assert all(t.is_cuda for t in model_params.values())
    snapshot = {n: t.detach().cpu().clone() for n, t in model_params.items()}
    start_root = hash_named_tensors(model_params.items())
    target = synth.WINDOW + 1

    cert, gather = _fabricate_certified_window(model_params, toy_cfg, target, start_root)
    chain = SimpleNamespace(get_window_commits=lambda _w: {})
    creds = BucketCreds(
        account_id="test", bucket_name="leader", access_key_id="k", secret_access_key="s"
    )

    def run_once() -> str:
        outer = ReplicatedOuterStep(
            toy_cfg.outer, {n: torch.Size(t.shape) for n, t in model_params.items()}
        )
        report = asyncio.run(
            catch_up(
                model_params,
                outer,
                _FakeExchange(cert, gather),
                None,  # storage unused by the exchange double
                chain,
                toy_dataset.manifest,
                toy_cfg,
                synth.WINDOW,
                target,
                leader_bucket=creds,
            )
        )
        assert report.applied_windows == (target,)
        assert report.skipped_void == ()
        assert report.unverified_windows == (target,)  # chain double had no commits
        return report.final_root

    root_a = run_once()
    assert root_a != start_root, "the fabricated window did not move the masters"
    assert all(t.is_cuda for t in model_params.values())  # updated in place, still on GPU

    # restore θ_start bitwise and replay the same certified window: same root.
    with torch.no_grad():
        for name, saved in snapshot.items():
            model_params[name].copy_(saved.to(model_params[name].device))
    assert hash_named_tensors(model_params.items()) == start_root
    root_b = run_once()
    assert root_a == root_b, "catch_up is nondeterministic over identical certified inputs"
    dist_ctx.barrier()
