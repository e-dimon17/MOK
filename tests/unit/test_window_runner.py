"""Tests for subnet/core/window_runner.py — the full window protocol on CPU.

Builds the complete loopback rig: real shard files -> DatasetShardIndex ->
RunManifest -> WindowBatchPlan, a tiny reference-backend model, moto S3
storage, a MagicMock chain, a fake window clock, and the runner in
self-as-leader bootstrap mode. One `run_window` call exercises phases 0-9
end to end; the replay test then proves the audited path reproduces the
committed θ_end bitwise.

Module fixtures are shared with tests/unit/test_replay.py via the plain
builder functions below (imported there, not re-run).
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mok_core.chain.schemas import WindowCommit
from mok_core.config import (
    InnerOptConfig,
    LRSpec,
    ModelConfig,
    RunConfig,
    StorageConfig,
    WindowConfig,
)
from mok_core.config.manifest import (
    DatasetManifestRef,
    PhaseEntry,
    PhaseOverrides,
    PRFSpec,
    RunManifest,
    VoidRange,
)
from mok_core.config.schemas import BucketCreds
from mok_core.data import DatasetShardIndex, ShardCache, ShardReader, shard_leaf_hash
from mok_core.determinism import hash_named_tensors
from mok_core.model import MoKTransformer, build_reference_model
from mok_core.storage import StorageClient, keys
from subnet.core.checkpoint import Checkpointer
from subnet.core.compress import ChunkingTransformer, ErrorFeedback, Quantizer, TopKCompressor
from subnet.core.outer_opt import ReplicatedOuterStep
from subnet.core.phase import resolve_phase
from subnet.core.replay import ReplayTask, WindowReplayer
from subnet.core.window_runner import (
    DENSE_SUFFIX,
    RunState,
    SingleNodeComm,
    WindowRunner,
    run_state_at,
    run_training_phase,
)

SEED = 7
RUN_SEED = bytes(range(32))
SEQ_LEN = 256
SEQS_PER_SHARD = 8
NUM_SHARDS = 8
VOCAB = 512
CYCLE = (3, 5, 7, 11)
UID = 3
WINDOW = 4
INNER_STEPS = 2
GRAD_ACCUM = 1
TOKENS_PER_MB = 512  # 2 sequences of 256


# --------------------------------------------------------------------------- #
# Shared builders (imported by test_replay.py)
# --------------------------------------------------------------------------- #


def make_model_cfg() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        num_dense_layers=0,
        hidden_size=256,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=128,
        vocab_size=VOCAB,
        seq_len=SEQ_LEN,
        num_experts=8,
        top_k=2,
        intermediate_size=256,
        ep_size=4,
    )


def make_run_cfg() -> RunConfig:
    return RunConfig(
        model=make_model_cfg(),
        window=WindowConfig(
            inner_steps=INNER_STEPS,
            tokens_per_rank_microbatch=TOKENS_PER_MB,
            grad_accum=GRAD_ACCUM,
            accum_ramp_start=GRAD_ACCUM,  # no ramp
            checkpoint_every_windows=2,
            upload_grace_s=90,
        ),
        inner=InnerOptConfig(lr=LRSpec(kind="const", const_lr=0.02)),
        storage=StorageConfig(gather_timeout_s=30.0),
    )


def shard_array(shard_idx: int) -> np.ndarray:
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


def write_shard_files(root: Path) -> None:
    for i in range(NUM_SHARDS):
        (root / f"shard-{i}.bin").write_bytes(shard_array(i).tobytes())


def build_index(data_dir: Path) -> DatasetShardIndex:
    hashes = [shard_leaf_hash(data_dir / f"shard-{i}.bin").hex() for i in range(NUM_SHARDS)]
    assert len(set(hashes)) == NUM_SHARDS
    return DatasetShardIndex(name="bulk", seq_len=SEQ_LEN, shard_hashes=hashes)


def build_manifest(index: DatasetShardIndex, **overrides: Any) -> RunManifest:
    ref = DatasetManifestRef(
        name="bulk",
        merkle_root=index.merkle().root.hex(),
        num_shards=NUM_SHARDS,
        shard_bytes=2 * SEQ_LEN * SEQS_PER_SHARD,
        seq_len=SEQ_LEN,
        tokens_total=NUM_SHARDS * SEQS_PER_SHARD * SEQ_LEN,
        tokenizer_hash="ab" * 32,
    )
    fields: dict[str, Any] = {
        "spec_version": 1,
        "run_id": "window-runner-test",
        "netuid": 11,
        "network": "test",
        "config_hash": "11" * 32,
        "container_digest": "sha256:" + "22" * 32,
        "mok_commit": "deadbeef",
        "tk_commit": "cafebabe",
        "attention_backend": "cudnn_det",
        "start_block": 100,
        "blocks_per_window": 225,
        "prf": PRFSpec(run_seed_hex=RUN_SEED.hex()),
        "datasets": (ref,),
        "init_checkpoint_hash": "33" * 32,
    }
    fields.update(overrides)
    return RunManifest(**fields)


def make_compressor(model: MoKTransformer, cfg: RunConfig) -> TopKCompressor:
    shapes = {n: s for n, s in model.param_shapes().items() if not n.endswith(DENSE_SUFFIX)}
    return TopKCompressor(
        ChunkingTransformer(shapes, target_chunk=cfg.compression.target_chunk),
        Quantizer(bins=cfg.compression.quant_bins, range_sigmas=cfg.compression.quant_range_sigmas),
        topk=cfg.compression.topk,
    )


def make_outer_step(model: MoKTransformer, cfg: RunConfig) -> ReplicatedOuterStep:
    return ReplicatedOuterStep(
        cfg.outer, {n: torch.Size(s) for n, s in model.param_shapes().items()}
    )


def make_shard_lookup_factory(data_dir: Path):
    """WindowReplayer-style shard_lookup_factory over the local shard files."""

    @contextmanager
    def factory(plan):
        readers = {i: ShardReader(data_dir / f"shard-{i}.bin", SEQ_LEN) for i in set(plan.shard_ids)}
        try:
            yield readers.__getitem__
        finally:
            for reader in readers.values():
                reader.close()

    return factory


class FakeClock:
    """boundary_ts(w) = 1000*w; `now` is set by the test."""

    def __init__(self, now_ts: float) -> None:
        self.now_ts = now_ts

    def boundary_ts(self, window: int) -> float:
        return 1000.0 * window

    def now(self) -> float:
        return self.now_ts


class RecordingMetrics:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, kind: str, **payload: Any) -> None:
        self.events.append((kind, payload))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def template_model(_single_thread) -> MoKTransformer:
    return build_reference_model(make_model_cfg(), SEED)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("shards")
    write_shard_files(root)
    return root


@pytest.fixture(scope="module")
def index(data_dir: Path) -> DatasetShardIndex:
    return build_index(data_dir)


@pytest.fixture(scope="module")
def manifest(index: DatasetShardIndex) -> RunManifest:
    return build_manifest(index)


@pytest.fixture(scope="module")
def moto_endpoint():
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture(scope="module")
def admin(moto_endpoint: str):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=moto_endpoint,
        region_name="us-east-1",
        aws_access_key_id="admin",
        aws_secret_access_key="admin",
    )


def make_creds(bucket_name: str) -> BucketCreds:
    return BucketCreds(
        account_id="testaccount",
        bucket_name=bucket_name,
        access_key_id="test-key",
        secret_access_key="test-secret",
    )


def fresh_bucket(admin: Any, tag: str) -> BucketCreds:
    name = f"mok-{tag}-{uuid.uuid4().hex[:8]}"
    admin.create_bucket(Bucket=name)
    return make_creds(name)


def make_fetch_fn(data_dir: Path):
    async def fetch(shard_idx: int) -> bytes:
        return (data_dir / f"shard-{shard_idx}.bin").read_bytes()

    return fetch


def make_runner(
    model: MoKTransformer,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    storage: StorageClient,
    creds: BucketCreds,
    chain: Any,
    clock: FakeClock,
    tmp: Path,
    *,
    checkpointer: Checkpointer | None = None,
    metrics: Any | None = None,
    self_leader: bool = True,
    cert_timeout_s: float = 10.0,
) -> WindowRunner:
    cfg = make_run_cfg()
    return WindowRunner(
        model,
        cfg,
        manifest,
        uid=UID,
        rank=0,
        world_size=1,
        wait_for_gate=False,   # FakeClock is static; gate timing is tested explicitly
        comm=SingleNodeComm(),
        storage=storage,
        chain=chain,
        shard_cache=ShardCache(tmp / "cache", 1 << 30, index),
        fetch_fn=make_fetch_fn(data_dir),
        compressor=make_compressor(model, cfg),
        error_feedback=ErrorFeedback(beta=cfg.compression.ef_beta),
        outer_step=make_outer_step(model, cfg),
        checkpointer=checkpointer,
        metrics=metrics,
        clock=clock,
        peer_buckets=lambda _w: {UID: creds},
        leader_bucket=lambda _w: creds,
        self_leader=self_leader,
        cert_poll_s=0.01,
        cert_timeout_s=cert_timeout_s,
    )


@pytest.fixture(scope="module")
def loopback(
    template_model: MoKTransformer,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    """One full self-leader loopback window (shared by several tests)."""
    tmp = tmp_path_factory.mktemp("loopback")
    creds = fresh_bucket(admin, "loop")
    chain = MagicMock()
    model = copy.deepcopy(template_model)
    metrics = RecordingMetrics()
    clock = FakeClock(now_ts=1000.0 * (WINDOW + 1) + 10.0)  # inside the gate
    checkpointer = Checkpointer(None, tmp / "ckpt")

    async def go() -> dict[str, Any]:
        async with StorageClient(
            creds, make_run_cfg().storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            runner = make_runner(
                model,
                manifest,
                index,
                data_dir,
                sc,
                creds,
                chain,
                clock,
                tmp,
                checkpointer=checkpointer,
                metrics=metrics,
            )
            outcome = await runner.run_window(WINDOW, RunState(0, 0, 0))
            payload_obj = await sc.get_bytes(creds, keys.payload_key(WINDOW, UID, "1"))
        return {"outcome": outcome, "payload_bytes": payload_obj}

    result = asyncio.run(go())
    result.update(
        model_after=model,
        chain=chain,
        metrics=metrics,
        checkpointer=checkpointer,
        creds=creds,
    )
    return result


# --------------------------------------------------------------------------- #
# (1) The full loopback window
# --------------------------------------------------------------------------- #


def test_loopback_window_completes_cleanly(
    loopback: dict[str, Any], template_model: MoKTransformer
) -> None:
    outcome = loopback["outcome"]
    assert not outcome.restart_required
    assert not outcome.desync
    assert not outcome.late_upload
    assert outcome.window == WINDOW

    theta_start_root = hash_named_tensors(template_model.iter_master_params())
    assert outcome.state_root_start == theta_start_root  # combined root == plain state_root
    assert outcome.theta_end_root != theta_start_root    # the window trained
    assert outcome.state_root_after not in (None, theta_start_root)  # outer step applied

    # the runner's model is exactly at the reported post-outer-step root
    assert hash_named_tensors(loopback["model_after"].iter_master_params()) == outcome.state_root_after

    assert outcome.gather_uids == (UID,)
    assert outcome.outer_report is not None and outcome.outer_report.applied_peers == 1
    assert outcome.outer_report.global_grad_l2 > 0.0
    assert outcome.sync_divergences == ()  # leader debug slices match ourselves

    assert outcome.state_after == RunState(
        global_step=1,
        global_inner_step=INNER_STEPS,
        tokens_consumed=INNER_STEPS * GRAD_ACCUM * TOKENS_PER_MB,
    )
    assert outcome.train_result is not None and outcome.train_result.final_loss > 0.0


def test_loopback_two_phase_commit_and_payload(loopback: dict[str, Any]) -> None:
    outcome = loopback["outcome"]
    chain = loopback["chain"]

    chain.commit_window.assert_called_once()
    commit = chain.commit_window.call_args.args[0]
    assert isinstance(commit, WindowCommit)
    assert commit.window == WINDOW
    assert commit.binds_payload_hash(outcome.payload_hash)   # wire v2: 128-bit prefix bound on-chain
    assert commit.state_root == outcome.state_root_start
    assert commit.theta_end_hash == outcome.theta_end_root

    # the uploaded bytes hash to the on-chain commitment
    from mok_core.determinism.hashing import hash_bytes

    assert commit.binds_payload_hash(hash_bytes(loopback["payload_bytes"]))
    assert outcome.upload_key == keys.payload_key(WINDOW, UID, "1")


def test_loopback_metrics_and_checkpoint(loopback: dict[str, Any]) -> None:
    outcome = loopback["outcome"]
    metrics: RecordingMetrics = loopback["metrics"]
    assert [kind for kind, _ in metrics.events] == ["window"]
    payload = metrics.events[0][1]
    assert payload["window"] == WINDOW and payload["applied_peers"] == 1

    # WINDOW % checkpoint_every_windows(2) == 0 -> saved; meta binds θ_start(W+1)
    assert outcome.checkpoint_saved
    ckpt: Checkpointer = loopback["checkpointer"]
    assert ckpt.local_windows() == [WINDOW]
    state, outer, meta = ckpt.load_local(WINDOW)
    assert meta.state_root == outcome.state_root_after
    assert meta.global_step == 1
    assert hash_named_tensors(state.items()) == outcome.state_root_after


# --------------------------------------------------------------------------- #
# (2) Replay reproduces the committed window bitwise
# --------------------------------------------------------------------------- #


def test_replay_matches_run_window_commit(
    loopback: dict[str, Any],
    template_model: MoKTransformer,
    manifest: RunManifest,
    data_dir: Path,
) -> None:
    outcome = loopback["outcome"]
    commit: WindowCommit = loopback["chain"].commit_window.call_args.args[0]

    replica = copy.deepcopy(template_model)  # θ_start(WINDOW), as catch_up would produce
    replayer = WindowReplayer(
        replica,
        make_run_cfg(),
        manifest,
        comm=SingleNodeComm(),
        shard_lookup_factory=make_shard_lookup_factory(data_dir),
        auditor_uid=42,
    )
    report = replayer.replay(
        ReplayTask(miner_uid=UID, window=WINDOW, commit=commit),
        global_state=RunState(0, 0, 0),
    )
    assert report.match
    assert report.replayed_theta_end == commit.theta_end_hash == outcome.theta_end_root
    assert report.divergences == []
    assert report.auditor_uid == 42 and report.miner_uid == UID
    # the auditor's replica is unchanged
    assert hash_named_tensors(replica.iter_master_params()) == commit.state_root


# --------------------------------------------------------------------------- #
# (5) Late upload
# --------------------------------------------------------------------------- #


def test_late_upload_skips_publication_without_corrupting_state(
    template_model: MoKTransformer,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path: Path,
) -> None:
    creds = fresh_bucket(admin, "late")
    chain = MagicMock()
    model = copy.deepcopy(template_model)
    clock = FakeClock(now_ts=1000.0 * (WINDOW + 1) + 90.0)  # gate closed exactly

    async def go():
        async with StorageClient(
            creds, make_run_cfg().storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            runner = make_runner(
                model, manifest, index, data_dir, sc, creds, chain, clock, tmp_path
            )
            return await runner.run_window(WINDOW, RunState(0, 0, 0))

    outcome = asyncio.run(go())
    assert outcome.late_upload
    assert not outcome.desync and not outcome.restart_required
    chain.commit_window.assert_not_called()  # neither commit phase nor upload ran
    assert outcome.upload_key is None
    assert outcome.gather_uids == ()
    assert outcome.outer_report is not None and outcome.outer_report.applied_peers == 0
    # state uncorrupted: empty certified set -> outer step is a no-op at θ_start
    assert outcome.state_root_after == outcome.state_root_start
    assert (
        hash_named_tensors(model.iter_master_params())
        == hash_named_tensors(template_model.iter_master_params())
    )
    assert outcome.state_after.global_step == 1  # the window still passed globally


# --------------------------------------------------------------------------- #
# Desync (certificate never appears)
# --------------------------------------------------------------------------- #


def test_certificate_timeout_yields_desync(
    template_model: MoKTransformer,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path: Path,
) -> None:
    creds = fresh_bucket(admin, "desync")
    chain = MagicMock()
    model = copy.deepcopy(template_model)
    clock = FakeClock(now_ts=1000.0 * (WINDOW + 1) + 10.0)
    state0 = RunState(0, 0, 0)

    async def go():
        async with StorageClient(
            creds, make_run_cfg().storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            runner = make_runner(
                model,
                manifest,
                index,
                data_dir,
                sc,
                creds,
                chain,
                clock,
                tmp_path,
                self_leader=False,     # nobody publishes a certificate
                cert_timeout_s=0.05,
            )
            return await runner.run_window(WINDOW, state0)

    outcome = asyncio.run(go())
    assert outcome.desync
    assert "certificate timeout" in outcome.reason
    assert outcome.state_after == state0  # counters unchanged — caller runs catch_up
    chain.commit_window.assert_called_once()  # its own upload did happen
    # θ was restored and no outer step ran: the model is still at θ_start
    assert (
        hash_named_tensors(model.iter_master_params())
        == hash_named_tensors(template_model.iter_master_params())
    )


# --------------------------------------------------------------------------- #
# (6) Restart on phase shape change
# --------------------------------------------------------------------------- #


def test_restart_required_on_context_extension_phase(
    template_model: MoKTransformer, index: DatasetShardIndex, data_dir: Path, tmp_path: Path
) -> None:
    manifest16k = build_manifest(
        index,
        phase_table=(
            PhaseEntry(start_window=0, name="bulk"),
            PhaseEntry(
                start_window=WINDOW,
                name="context16k",
                overrides=PhaseOverrides(seq_len=2 * SEQ_LEN, rope_theta=5e5, requires_restart=True),
            ),
        ),
    )
    chain = MagicMock()
    storage = MagicMock()
    model = copy.deepcopy(template_model)
    runner = make_runner(
        model,
        manifest16k,
        index,
        data_dir,
        storage,
        make_creds("unused"),
        chain,
        FakeClock(0.0),
        tmp_path,
    )

    outcome = asyncio.run(runner.run_window(WINDOW, RunState(0, 0, 0)))
    assert outcome.restart_required
    assert not outcome.desync and not outcome.late_upload
    assert outcome.state_after == RunState(0, 0, 0)
    chain.commit_window.assert_not_called()
    storage.put_bytes.assert_not_called()

    # the window BEFORE the phase flip does not restart
    phase_before = resolve_phase(manifest16k, make_run_cfg(), WINDOW - 1)
    assert not phase_before.requires_restart


# --------------------------------------------------------------------------- #
# run_state_at — consensus run accounting
# --------------------------------------------------------------------------- #


def test_run_state_at_no_ramp(manifest: RunManifest) -> None:
    cfg = make_run_cfg()
    assert run_state_at(cfg, manifest, 0, world_size=1) == RunState(0, 0, 0)
    got = run_state_at(cfg, manifest, WINDOW, world_size=1)
    per_window = INNER_STEPS * GRAD_ACCUM * TOKENS_PER_MB
    assert got == RunState(WINDOW, WINDOW * INNER_STEPS, WINDOW * per_window)
    # world_size scales token accounting
    got8 = run_state_at(cfg, manifest, WINDOW, world_size=8)
    assert got8.tokens_consumed == 8 * WINDOW * per_window


def test_run_state_at_skips_void_windows(index: DatasetShardIndex) -> None:
    voided = build_manifest(
        index,
        void_ranges=(VoidRange(first_window=1, last_window=2, reseed_salt_hex="ab" * 32),),
    )
    cfg = make_run_cfg()
    got = run_state_at(cfg, voided, 4, world_size=1)
    assert got.global_step == 2  # windows 0 and 3 only
    assert got.global_inner_step == 2 * INNER_STEPS


def test_run_state_at_mirrors_accum_ramp() -> None:
    cfg = make_run_cfg()
    ramped = cfg.model_copy(
        update={
            "window": WindowConfig(
                inner_steps=INNER_STEPS,
                tokens_per_rank_microbatch=TOKENS_PER_MB,
                grad_accum=2,
                accum_ramp_start=1,
                accum_ramp_tokens=3 * TOKENS_PER_MB,
            )
        }
    )
    index = build_index_stub()
    manifest = build_manifest(index)
    got = run_state_at(ramped, manifest, 2, world_size=1)
    # step-by-step: t=0 -> accum 1 (+512); t=512 -> 1+(2-1)*512//1536=1 (+512);
    # t=1024 -> 1+1024//1536=1 (+512); t=1536 -> ramp done, accum 2 (+1024)
    assert got == RunState(2, 4, 512 + 512 + 512 + 1024)


def build_index_stub() -> DatasetShardIndex:
    return DatasetShardIndex(name="bulk", seq_len=SEQ_LEN, shard_hashes=["ab" * 32] * NUM_SHARDS)


# --------------------------------------------------------------------------- #
# run_training_phase — deterministic payload bytes
# --------------------------------------------------------------------------- #


def test_training_phase_payload_bytes_deterministic(
    template_model: MoKTransformer, manifest: RunManifest, data_dir: Path
) -> None:
    cfg = make_run_cfg()
    phase = resolve_phase(manifest, cfg, WINDOW)
    factory = make_shard_lookup_factory(data_dir)

    def one_run() -> tuple[bytes, str, str]:
        model = copy.deepcopy(template_model)
        with factory(
            _plan_for(model, manifest, phase)
        ) as shard_lookup:
            artifacts = run_training_phase(
                model,
                cfg,
                manifest,
                phase,
                uid=UID,
                window=WINDOW,
                rank=0,
                world_size=1,
                comm=SingleNodeComm(),
                shard_lookup=shard_lookup,
                global_state=RunState(0, 0, 0),
                compressor=make_compressor(model, cfg),
                error_feedback=ErrorFeedback(beta=cfg.compression.ef_beta),
            )
        assert artifacts.payload_bytes is not None
        assert artifacts.theta_end_root is not None and artifacts.state_root_start is not None
        return artifacts.payload_bytes, artifacts.theta_end_root, artifacts.state_root_start

    bytes_a, end_a, start_a = one_run()
    bytes_b, end_b, start_b = one_run()
    assert bytes_a == bytes_b  # the payload BYTES are a pure function of the window
    assert end_a == end_b and start_a == start_b


def _plan_for(model: MoKTransformer, manifest: RunManifest, phase) -> Any:
    from subnet.core.window_runner import build_window_plan

    return build_window_plan(
        manifest, phase, run_seed=RUN_SEED, uid=UID, window=WINDOW, rank=0, world_size=1
    )


class _AdvancingClock:
    """boundary_ts like FakeClock, but now() advances by `step` per call —
    lets the gate-open wait loop make progress without real sleeping."""

    def __init__(self, start: float, step: float = 40.0) -> None:
        self.now_ts = float(start)
        self.step = float(step)

    def boundary_ts(self, window: int) -> float:
        return 1000.0 * window

    def now(self) -> float:
        self.now_ts += self.step
        return self.now_ts


async def test_runner_waits_for_gate_open_before_upload(
    template_model, manifest, index, data_dir, admin, moto_endpoint, tmp_path
):
    """Training that finishes before boundary(W+1) must NOT upload early — the
    payload would fail validators' in-gate check. The runner waits for the gate."""
    creds = fresh_bucket(admin, "gatewait")
    chain = MagicMock()
    chain.commit_window = MagicMock()
    clock = _AdvancingClock(start=10.0)          # window 0: gate opens at 1000
    model = copy.deepcopy(template_model)
    async with StorageClient(
        creds, make_run_cfg().storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
    ) as sc:
        runner = make_runner(model, manifest, index, data_dir, sc, creds, chain, clock, tmp_path)
        runner.wait_for_gate = True              # the behavior under test
        runner.cert_poll_s = 0.01
        outcome = await runner.run_window(0, RunState(0, 0, 0))
    assert not outcome.desync and not outcome.late_upload
    # phase-1 chain commit happened, and only after the gate opened
    assert chain.commit_window.called
    assert clock.now_ts >= 1000.0                # the wait actually advanced past the boundary
