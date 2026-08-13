"""Tests for C/miner — the miner application over the full loopback rig.

Reuses the shard/model/manifest builders from tests/unit/test_window_runner.py
(imported, not re-run) and drives `MinerApp.run()` end to end on CPU: moto S3
storage, the `ScriptedChain` harness chain, a settable `LoopbackClock`, and a
2-window mini-session in self-leader mode. Also covers the restart(3) path,
warmup null windows, the desync→catch-up recovery path, the `--local-harness`
bootstrap, and import-cleanliness without bittensor.

The builders here (`make_app_cfg`, `make_app_manifest`, `make_ctx`, ...) are
shared with test_app_validator.py / test_app_auditor.py.
"""

from __future__ import annotations

import asyncio
import copy
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
import test_window_runner as twr
import torch

from C.core.checkpoint import build_outer_inputs
from C.core.compress import ErrorFeedback
from C.core.window_runner import (
    DENSE_SUFFIX,
    RunState,
    SingleNodeComm,
    build_window_plan,
    run_training_phase,
)
from C.miner.app import RESTART_EXIT_CODE, MinerApp
from C.miner.bootstrap import (
    INIT_SEED,
    LocalSigner,
    LoopbackClock,
    MemoryStorage,
    NodeContext,
    ScriptedChain,
    bootstrap,
)
from mok_core.chain.schemas import WindowCommit
from mok_core.config import (
    InnerOptConfig,
    LRSpec,
    RunConfig,
    ScoringConfig,
    StorageConfig,
    WindowConfig,
)
from mok_core.config.manifest import PhaseEntry, PhaseOverrides, RunManifest
from mok_core.config.schemas import BucketCreds
from mok_core.data import DatasetShardIndex, ShardCache
from mok_core.determinism import hash_bytes, hash_named_tensors
from mok_core.model import MoKTransformer, build_reference_model
from mok_core.storage import ObjectMissingError, StorageClient, keys

UID = twr.UID  # 3
RUN_SEED = twr.RUN_SEED
WINDOW_S = 1000.0
TOKENS_PER_WINDOW = twr.INNER_STEPS * twr.GRAD_ACCUM * twr.TOKENS_PER_MB


# --------------------------------------------------------------------------- #
# Shared builders (imported by test_app_validator.py / test_app_auditor.py)
# --------------------------------------------------------------------------- #


def make_app_cfg(**window_kw: Any) -> RunConfig:
    window: dict[str, Any] = {
        "inner_steps": twr.INNER_STEPS,
        "tokens_per_rank_microbatch": twr.TOKENS_PER_MB,
        "grad_accum": twr.GRAD_ACCUM,
        "accum_ramp_start": twr.GRAD_ACCUM,
        "checkpoint_every_windows": 2,
        "upload_grace_s": 90,
        "warmup_null_windows": 0,
        "blocks_per_window": 225,
    }
    window.update(window_kw)
    return RunConfig(
        model=twr.make_model_cfg(),
        window=WindowConfig(**window),
        inner=InnerOptConfig(lr=LRSpec(kind="const", const_lr=0.02)),
        storage=StorageConfig(gather_timeout_s=10.0),
        scoring=ScoringConfig(windows_per_weights=1, eval_sequences=4, overlap_threshold=1.5),
    )


def make_app_manifest(index: DatasetShardIndex, template: MoKTransformer, **overrides: Any) -> RunManifest:
    root = hash_named_tensors(template.iter_master_params())
    return twr.build_manifest(index, init_checkpoint_hash=root, **overrides)


def make_clock(*, genesis: float = 0.0, now_ts: float = 10.0) -> LoopbackClock:
    return LoopbackClock(genesis=genesis, window_s=WINDOW_S, now_ts=now_ts)


def make_chain(
    clock: LoopbackClock,
    *,
    my_uid: int,
    stakes: dict[int, float],
    buckets: dict[int, BucketCreds],
) -> ScriptedChain:
    return ScriptedChain(
        clock=clock,
        start_block=100,
        blocks_per_window=225,
        my_uid=my_uid,
        stakes=stakes,
        buckets=buckets,
    )


def make_ctx(
    role: str,
    *,
    cfg: RunConfig,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    storage: Any,
    creds: BucketCreds,
    chain: Any,
    clock: LoopbackClock,
    tmp: Path,
    uid: int,
) -> NodeContext:
    state_dir = tmp / f"state-{role}-{uid}"
    state_dir.mkdir(parents=True, exist_ok=True)
    return NodeContext(
        role=role,
        cfg=cfg,
        manifest=manifest,
        uid=uid,
        signer=LocalSigner(hotkey=f"local-{uid}"),
        chain=chain,
        storage=storage,
        own_bucket=creds,
        shard_caches={"bulk": ShardCache(tmp / f"cache-{role}-{uid}", 1 << 30, index)},
        shard_indexes={"bulk": index},
        fetch_fns={"bulk": twr.make_fetch_fn(data_dir)},
        metrics=twr.RecordingMetrics(),
        comm=SingleNodeComm(),
        clock=clock,
        rank=0,
        world_size=1,
        protocol_world_size=1,
        device="cpu",
        state_dir=state_dir,
        local=True,
        dev_insecure=True,
    )


def run_peer_window(
    template: MoKTransformer,
    cfg: RunConfig,
    manifest: RunManifest,
    data_dir: Path,
    *,
    uid: int,
    window: int,
    state: RunState | None = None,
) -> Any:
    """A peer miner's full training phase → TrainingArtifacts (payload included)."""
    model = copy.deepcopy(template)
    factory = twr.make_shard_lookup_factory(data_dir)
    phase = _phase(manifest, cfg, window)
    plan = build_window_plan(
        manifest, phase, run_seed=RUN_SEED, uid=uid, window=window, rank=0, world_size=1
    )
    with factory(plan) as shard_lookup:
        return run_training_phase(
            model,
            cfg,
            manifest,
            phase,
            uid=uid,
            window=window,
            rank=0,
            world_size=1,
            comm=SingleNodeComm(),
            shard_lookup=shard_lookup,
            global_state=state if state is not None else RunState(0, 0, 0),
            compressor=twr.make_compressor(model, cfg),
            error_feedback=ErrorFeedback(beta=cfg.compression.ef_beta),
            plan=plan,
            run_seed=RUN_SEED,
        )


def _phase(manifest: RunManifest, cfg: RunConfig, window: int) -> Any:
    from C.core.phase import resolve_phase

    return resolve_phase(manifest, cfg, window)


def apply_outer_to(
    template: MoKTransformer, cfg: RunConfig, payloads: dict[int, Any]
) -> MoKTransformer:
    """Reference application of certified payloads (uid-ascending) to a copy."""
    model = copy.deepcopy(template)
    master = dict(model.iter_master_params())
    compressor = twr.make_compressor(model, cfg)
    comp_names = sorted(n for n in master if not n.endswith(DENSE_SUFFIX))
    ordered = {uid: payloads[uid] for uid in sorted(payloads)}
    sparse, dense, norms = build_outer_inputs(ordered, compressor, comp_names)
    twr.make_outer_step(model, cfg).apply(master, sparse, dense, norms)
    return model


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
    return build_reference_model(twr.make_model_cfg(), INIT_SEED)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("app-shards")
    twr.write_shard_files(root)
    return root


@pytest.fixture(scope="module")
def index(data_dir: Path) -> DatasetShardIndex:
    return twr.build_index(data_dir)


@pytest.fixture(scope="module")
def manifest(index: DatasetShardIndex, template_model: MoKTransformer) -> RunManifest:
    return make_app_manifest(index, template_model)


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


def fresh_bucket(admin: Any, tag: str) -> BucketCreds:
    name = f"mok-app-{tag}-{uuid.uuid4().hex[:8]}"
    admin.create_bucket(Bucket=name)
    return twr.make_creds(name)


# --------------------------------------------------------------------------- #
# (1) Full 2-window mini-session (self-leader loopback)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def mini_session(
    template_model: MoKTransformer,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("mini-session")
    cfg = make_app_cfg()
    creds = fresh_bucket(admin, "miner")
    clock = make_clock(now_ts=10.0)  # inside window 0
    chain = make_chain(clock, my_uid=UID, stakes={UID: 1.0}, buckets={UID: creds})
    outcomes: list[Any] = []

    def advance(outcome: Any) -> None:
        outcomes.append(outcome)
        clock.set(WINDOW_S * (outcome.window + 1) + 10.0)

    async def go() -> dict[str, Any]:
        async with StorageClient(
            creds, cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            ctx = make_ctx(
                "miner",
                cfg=cfg,
                manifest=manifest,
                index=index,
                data_dir=data_dir,
                storage=sc,
                creds=creds,
                chain=chain,
                clock=clock,
                tmp=tmp,
                uid=UID,
            )
            app = MinerApp(
                ctx,
                self_leader=True,
                max_windows=2,
                on_window=advance,
                cert_poll_s=0.01,
                cert_timeout_s=5.0,
                window_poll_s=0.01,
            )
            code = await app.run()
            payloads = {
                w: await sc.get_bytes(creds, keys.payload_key(w, UID, "1")) for w in (0, 1)
            }
            debug_keys = await sc.list_keys(creds, "debug/")
            telemetry_keys = await sc.list_keys(creds, "telemetry/")
            return {
                "code": code,
                "app": app,
                "ctx": ctx,
                "payload_bytes": payloads,
                "debug_keys": debug_keys,
                "telemetry_keys": telemetry_keys,
            }

    result = asyncio.run(go())
    result.update(chain=chain, outcomes=outcomes, tmp=tmp)
    return result


def test_mini_session_completes(mini_session: dict[str, Any]) -> None:
    assert mini_session["code"] == 0
    outcomes = mini_session["outcomes"]
    assert [o.window for o in outcomes] == [0, 1]
    for o in outcomes:
        assert not o.restart_required and not o.desync and not o.late_upload
        assert o.outer_report is not None and o.outer_report.applied_peers == 1


def test_mini_session_two_phase_commits(mini_session: dict[str, Any]) -> None:
    chain: ScriptedChain = mini_session["chain"]
    assert sorted(chain.window_commits) == [0, 1]
    for w, outcome in zip((0, 1), mini_session["outcomes"], strict=True):
        commit = chain.window_commits[w][UID]
        assert isinstance(commit, WindowCommit)
        assert commit.payload_hash == outcome.payload_hash
        assert hash_bytes(mini_session["payload_bytes"][w]) == commit.payload_hash
        assert commit.state_root == outcome.state_root_start
        assert commit.theta_end_hash == outcome.theta_end_root


def test_mini_session_state_advances(
    mini_session: dict[str, Any], template_model: MoKTransformer
) -> None:
    app: MinerApp = mini_session["app"]
    assert app.run_state == RunState(
        global_step=2,
        global_inner_step=2 * twr.INNER_STEPS,
        tokens_consumed=2 * TOKENS_PER_WINDOW,
    )
    assert app.window == 2
    # window 1 chained off window 0's post-outer state
    o0, o1 = mini_session["outcomes"]
    assert o1.state_root_start == o0.state_root_after
    assert o1.state_root_after != o0.state_root_after
    # the live model sits exactly at the last reported root
    assert hash_named_tensors(app.model.iter_master_params()) == o1.state_root_after
    assert o0.state_root_start == hash_named_tensors(template_model.iter_master_params())


def test_mini_session_checkpoint_cadence(mini_session: dict[str, Any]) -> None:
    app: MinerApp = mini_session["app"]
    o0, o1 = mini_session["outcomes"]
    assert o0.checkpoint_saved      # 0 % 2 == 0 → runner cadence save
    assert not o1.checkpoint_saved  # 1 % 2 != 0
    # the final graceful-stop checkpoint added window 1
    assert app.checkpointer.local_windows() == [0, 1]
    _state, _outer, meta = app.checkpointer.load_local(0)
    assert meta.state_root == o0.state_root_after
    _state, _outer, meta1 = app.checkpointer.load_local(1)
    assert meta1.state_root == o1.state_root_after


def test_mini_session_publications(mini_session: dict[str, Any]) -> None:
    from C.core.exchange import debug_key

    assert debug_key(0, UID) in mini_session["debug_keys"]
    assert debug_key(1, UID) in mini_session["debug_keys"]
    assert keys.telemetry_key(0, UID) in mini_session["telemetry_keys"]
    assert keys.telemetry_key(1, UID) in mini_session["telemetry_keys"]
    kinds = [k for k, _ in mini_session["ctx"].metrics.events]
    assert kinds.count("window") == 2


# --------------------------------------------------------------------------- #
# (2) restart_required → SystemExit(3)
# --------------------------------------------------------------------------- #


def test_restart_required_exits_3(
    template_model: MoKTransformer, index: DatasetShardIndex, data_dir: Path, tmp_path: Path
) -> None:
    manifest16k = make_app_manifest(
        index,
        template_model,
        phase_table=(
            PhaseEntry(
                start_window=0,
                name="context16k",
                overrides=PhaseOverrides(
                    seq_len=2 * twr.SEQ_LEN, rope_theta=5e5, requires_restart=True
                ),
            ),
        ),
    )
    cfg = make_app_cfg()
    clock = make_clock(now_ts=10.0)
    creds = twr.make_creds("unused-restart")
    chain = make_chain(clock, my_uid=UID, stakes={UID: 1.0}, buckets={UID: creds})
    storage = MemoryStorage(creds)
    ctx = make_ctx(
        "miner",
        cfg=cfg,
        manifest=manifest16k,
        index=index,
        data_dir=data_dir,
        storage=storage,
        creds=creds,
        chain=chain,
        clock=clock,
        tmp=tmp_path,
        uid=UID,
    )
    app = MinerApp(ctx, self_leader=True, max_windows=2, window_poll_s=0.01)

    with pytest.raises(SystemExit) as excinfo:
        asyncio.run(app.run())
    assert excinfo.value.code == RESTART_EXIT_CODE
    assert not chain.window_commits  # nothing was committed or uploaded
    assert ("restart_required", {"window": 0, "reason": app.last_outcome.reason}) in ctx.metrics.events


# --------------------------------------------------------------------------- #
# (3) Warmup null windows: train but never publish
# --------------------------------------------------------------------------- #


def test_warmup_window_suppresses_publication(
    template_model: MoKTransformer,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path: Path,
) -> None:
    cfg = make_app_cfg(warmup_null_windows=1)
    creds = fresh_bucket(admin, "warmup")
    clock = make_clock(now_ts=10.0)
    chain = make_chain(clock, my_uid=UID, stakes={UID: 1.0}, buckets={UID: creds})

    async def go() -> Any:
        async with StorageClient(
            creds, cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            ctx = make_ctx(
                "miner",
                cfg=cfg,
                manifest=manifest,
                index=index,
                data_dir=data_dir,
                storage=sc,
                creds=creds,
                chain=chain,
                clock=clock,
                tmp=tmp_path,
                uid=UID,
            )
            app = MinerApp(
                ctx,
                self_leader=True,
                max_windows=1,
                cert_poll_s=0.01,
                cert_timeout_s=5.0,
                window_poll_s=0.01,
            )
            code = await app.run()
            with pytest.raises(ObjectMissingError):
                await sc.get_bytes(creds, keys.payload_key(0, UID, "1"))
            return code, app

    code, app = asyncio.run(go())
    assert code == 0
    assert app.last_outcome.late_upload  # publication suppressed via the gate
    assert not chain.window_commits      # phase-1 commit never happened
    # θ unchanged: empty certified set → outer step no-op
    assert app.last_outcome.state_root_after == app.last_outcome.state_root_start
    assert hash_named_tensors(app.model.iter_master_params()) == hash_named_tensors(
        template_model.iter_master_params()
    )
    assert app.warmup_left == 0  # consumed


# --------------------------------------------------------------------------- #
# (4) Desync → catch-up recovery from the leader's aggregator mirror
# --------------------------------------------------------------------------- #


class FlakyAggregatorStorage(StorageClient):
    """Fails the FIRST fetch of selected keys (scripted mirror unavailability)."""

    def __init__(self, *args: Any, fail_once: set[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.fail_once = set(fail_once)

    async def get_bytes(self, bucket: Any, key: str, **kwargs: Any) -> bytes:
        if key in self.fail_once:
            self.fail_once.discard(key)
            raise ObjectMissingError(f"scripted first-fetch failure for {key}")
        return await super().get_bytes(bucket, key, **kwargs)


def test_desync_then_catch_up_recovery(
    template_model: MoKTransformer,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path: Path,
) -> None:
    from C.core.certificate import build_certificate
    from C.core.exchange import put_aggregator_object, put_certificate
    from C.core.window_runner import _SelfCommit

    cfg = make_app_cfg()
    peer_uid = 9
    peer = run_peer_window(template_model, cfg, manifest, data_dir, uid=peer_uid, window=0)
    assert peer.payload_bytes is not None
    commit9 = WindowCommit(
        window=0,
        payload_hash=peer.payload_hash,
        state_root=peer.state_root_start,
        theta_end_hash=peer.theta_end_root,
    )
    cert = build_certificate(
        0,
        {peer_uid: _SelfCommit(uid=peer_uid, payload_hash=peer.payload_hash, in_gate=True, valid=True)},
        {peer_uid: 1.0},
        gather_count=cfg.window.gather_peer_count,
        reserve_count=cfg.window.reserve_peer_count,
        theta_start_root=peer.state_root_start,
        leader_uid=peer_uid,
        sign=LocalSigner(hotkey=f"local-{peer_uid}").sign,
    )

    own_creds = fresh_bucket(admin, "desync-self")
    leader_creds = fresh_bucket(admin, "desync-leader")
    clock = make_clock(now_ts=10.0)
    chain = make_chain(
        clock,
        my_uid=UID,
        stakes={peer_uid: 5.0, UID: 1.0},  # peer is the leader
        buckets={UID: own_creds, peer_uid: leader_creds},
    )
    chain.window_commits.setdefault(0, {})[peer_uid] = commit9

    async def go() -> tuple[int, MinerApp, Any]:
        async with StorageClient(
            leader_creds, cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as leader_sc:
            await put_certificate(leader_sc, cert)
            await put_aggregator_object(leader_sc, 0, {peer_uid: peer.payload_bytes})
        flaky = FlakyAggregatorStorage(
            own_creds,
            cfg.storage,
            endpoint_override=moto_endpoint,
            retry_base_delay_s=0.01,
            fail_once={keys.aggregator_key(0)},
        )
        async with flaky as sc:
            ctx = make_ctx(
                "miner",
                cfg=cfg,
                manifest=manifest,
                index=index,
                data_dir=data_dir,
                storage=sc,
                creds=own_creds,
                chain=chain,
                clock=clock,
                tmp=tmp_path,
                uid=UID,
            )
            app = MinerApp(
                ctx,
                self_leader=False,
                max_windows=1,
                cert_poll_s=0.01,
                cert_timeout_s=2.0,
                catchup_retries=3,
                catchup_retry_s=0.01,
                window_poll_s=0.01,
            )
            code = await app.run()
            return code, app, ctx

    code, app, ctx = asyncio.run(go())
    assert code == 0
    assert app.last_outcome.desync
    assert "unavailable" in app.last_outcome.reason
    kinds = [k for k, _ in ctx.metrics.events]
    assert "desync" in kinds and "catch_up" in kinds

    # recovery applied EXACTLY the certified peer's outer step, bitwise
    expected = apply_outer_to(template_model, cfg, {peer_uid: peer.payload})
    assert hash_named_tensors(app.model.iter_master_params()) == hash_named_tensors(
        expected.iter_master_params()
    )
    assert app.window == 1
    assert app.run_state == RunState(1, twr.INNER_STEPS, TOKENS_PER_WINDOW)
    # the miner's own upload still happened before the desync (two-phase order)
    assert chain.window_commits[0][UID].state_root == peer.state_root_start


# --------------------------------------------------------------------------- #
# (5) --local-harness bootstrap (fallback in-memory harness)
# --------------------------------------------------------------------------- #


def test_bootstrap_local_harness(tmp_path: Path) -> None:
    import yaml

    from mok_core.config import config_hash

    cfg = make_app_cfg()
    cfg_path = tmp_path / "app-config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8")
    prev_det = torch.are_deterministic_algorithms_enabled()

    async def go() -> None:
        ctx = await bootstrap(
            "miner",
            [
                "--config",
                str(cfg_path),
                "--local-harness",
                "--uid",
                "0",
                "--device",
                "cpu",
                "--state-dir",
                str(tmp_path / "state"),
            ],
        )
        try:
            assert ctx.local and ctx.uid == 0 and ctx.role == "miner"
            assert ctx.manifest.config_hash == config_hash(ctx.cfg)
            assert "bulk" in ctx.shard_caches and "bulk" in ctx.fetch_fns
            assert ctx.world_size == 1 and ctx.rank == 0
            assert ctx.chain.get_manifest_hash(0) == ctx.manifest.manifest_hash()
            # the harness seeded verifiable shards: fetch one through the fetch_fn
            data = await ctx.fetch_fns["bulk"](0)
            assert len(data) == ctx.manifest.datasets[0].shard_bytes
        finally:
            await ctx.aclose()

    try:
        asyncio.run(go())
    finally:
        torch.use_deterministic_algorithms(prev_det)


# --------------------------------------------------------------------------- #
# (6) Apps import clean without bittensor/wandb/vllm
# --------------------------------------------------------------------------- #


def test_apps_import_clean_without_heavy_deps() -> None:
    code = (
        "import sys\n"
        "import C.miner, C.miner.main, C.validator, C.validator.main, C.auditor, C.auditor.main\n"
        "for mod in ('bittensor', 'wandb', 'vllm', 'transformers', 'aioboto3', 'boto3'):\n"
        "    assert mod not in sys.modules, mod\n"
        "print('clean')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[2]),
        check=True,
    )
    assert "clean" in out.stdout
