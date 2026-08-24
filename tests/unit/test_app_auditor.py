"""Tests for subnet/auditor — sampled bitwise replays over scripted miner artifacts.

Reuses the app-test builders from test_app_miner.py. The scripted world:
miner 3 mined window 0 (real training phase), leader validator 1 published the
certificate + aggregator mirror; auditor 7 (ρ = 1.0 so the sample always
fires) replays the miner's window bitwise, publishes the signed verdict to its
own bucket, then applies the outer step and advances. A tampered commit yields
a mismatch report with divergences; a zero wall-time budget skips replays.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import test_app_miner as tam
import test_window_runner as twr
import torch

from mok_core.chain.schemas import WindowCommit
from mok_core.chain.windows import boundary_block
from mok_core.config import RunConfig
from mok_core.config.manifest import RunManifest
from mok_core.data import DatasetShardIndex
from mok_core.determinism import hash_named_tensors
from mok_core.model import MoKTransformer, build_reference_model
from mok_core.storage import StorageClient
from subnet.auditor.app import AuditorApp
from subnet.core.certificate import build_certificate
from subnet.core.exchange import list_audit_reports, put_aggregator_object, put_certificate
from subnet.core.replay import audit_sampler, verify_report
from subnet.core.window_runner import _SelfCommit
from subnet.miner.bootstrap import AUDITOR_COMMITMENT, LocalSigner

A_UID = 7        # the auditor under test
M_UID = 3        # the audited miner
L_UID = 1        # the leader validator (highest stake)
WINDOW = 0


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def template_model(_single_thread) -> MoKTransformer:
    return build_reference_model(twr.make_model_cfg(), tam.INIT_SEED)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("auditor-shards")
    twr.write_shard_files(root)
    return root


@pytest.fixture(scope="module")
def index(data_dir: Path) -> DatasetShardIndex:
    return twr.build_index(data_dir)


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


@pytest.fixture(scope="module")
def audit_cfg() -> RunConfig:
    cfg = tam.make_app_cfg()
    return cfg.model_copy(update={"audit": cfg.audit.model_copy(update={"probability": 1.0})})


@pytest.fixture(scope="module")
def manifest(index: DatasetShardIndex, template_model: MoKTransformer) -> RunManifest:
    return tam.make_app_manifest(index, template_model)


@pytest.fixture(scope="module")
def miner_artifacts(
    template_model: MoKTransformer, audit_cfg: RunConfig, manifest: RunManifest, data_dir: Path
) -> Any:
    return tam.run_peer_window(
        template_model, audit_cfg, manifest, data_dir, uid=M_UID, window=WINDOW
    )


def make_commit(art: Any, *, tampered: bool = False) -> WindowCommit:
    theta_end = art.theta_end_root
    if tampered:
        flipped = "0" if theta_end[0] != "0" else "1"
        theta_end = flipped + theta_end[1:]
    return WindowCommit(
        window=WINDOW,
        payload_hash=art.payload_hash,
        state_root=art.state_root_start,
        theta_end_hash=theta_end,
    )


async def _run_auditor(
    *,
    audit_cfg: RunConfig,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp: Path,
    art: Any,
    commit: WindowCommit,
    wall_budget_s: float | None = None,
) -> dict[str, Any]:
    """One full auditor window against the scripted leader/miner world."""
    a_creds = tam.fresh_bucket(admin, "auditor")
    l_creds = tam.fresh_bucket(admin, "leader")
    m_creds = tam.fresh_bucket(admin, "miner")
    clock = tam.make_clock(now_ts=tam.WINDOW_S + 95.0)  # window 0's gate is closed
    chain = tam.make_chain(
        clock,
        my_uid=A_UID,
        stakes={L_UID: 10.0},
        buckets={A_UID: a_creds, L_UID: l_creds, M_UID: m_creds},
    )
    chain.window_commits.setdefault(WINDOW, {})[M_UID] = commit

    cert = build_certificate(
        WINDOW,
        {M_UID: _SelfCommit(uid=M_UID, payload_hash=art.payload_hash, in_gate=True, valid=True)},
        {M_UID: 1.0},
        gather_count=audit_cfg.window.gather_peer_count,
        reserve_count=audit_cfg.window.reserve_peer_count,
        theta_start_root=art.state_root_start,
        leader_uid=L_UID,
        sign=LocalSigner(hotkey=f"local-{L_UID}").sign,
    )

    async with StorageClient(
        l_creds, audit_cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
    ) as lsc:
        await put_certificate(lsc, cert)
        await put_aggregator_object(lsc, WINDOW, {M_UID: art.payload_bytes})

    async with StorageClient(
        a_creds, audit_cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
    ) as sc:
        ctx = tam.make_ctx(
            "auditor",
            cfg=audit_cfg,
            manifest=manifest,
            index=index,
            data_dir=data_dir,
            storage=sc,
            creds=a_creds,
            chain=chain,
            clock=clock,
            tmp=tmp,
            uid=A_UID,
        )
        kwargs: dict[str, Any] = {}
        if wall_budget_s is not None:
            kwargs["wall_budget_s"] = wall_budget_s
        app = AuditorApp(ctx, max_windows=1, catchup_retry_s=0.01, poll_s=0.01, **kwargs)
        code = await app.run()
        reports = await list_audit_reports(sc, a_creds, WINDOW)
        return {"code": code, "app": app, "ctx": ctx, "chain": chain, "reports": reports}


# --------------------------------------------------------------------------- #
# Assignment determinism
# --------------------------------------------------------------------------- #


def test_audit_assignment_deterministic() -> None:
    run_seed = twr.RUN_SEED
    block_hash = bytes(range(32))
    uids = list(range(10))
    a = audit_sampler(run_seed, block_hash, WINDOW, uids, 0.5, [A_UID, 11])
    b = audit_sampler(run_seed, block_hash, WINDOW, uids, 0.5, [11, A_UID])
    assert a == b  # auditor-set order never matters (sorted internally)
    assert audit_sampler(run_seed, block_hash, WINDOW, uids, 0.5, [A_UID, 11]) == a
    # rho = 1.0 samples everyone, round-robin over sorted auditors
    full = audit_sampler(run_seed, block_hash, WINDOW, [M_UID], 1.0, [A_UID])
    assert full == {A_UID: [(M_UID, WINDOW)]}


# --------------------------------------------------------------------------- #
# Honest miner → match report; then the replica advances via the outer step
# --------------------------------------------------------------------------- #


def test_auditor_replays_and_publishes_match(
    audit_cfg: RunConfig,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path: Path,
    template_model: MoKTransformer,
    miner_artifacts: Any,
) -> None:
    result = asyncio.run(
        _run_auditor(
            audit_cfg=audit_cfg,
            manifest=manifest,
            index=index,
            data_dir=data_dir,
            admin=admin,
            moto_endpoint=moto_endpoint,
            tmp=tmp_path,
            art=miner_artifacts,
            commit=make_commit(miner_artifacts),
        )
    )
    assert result["code"] == 0
    app: AuditorApp = result["app"]
    chain = result["chain"]

    # the signed match report landed in the auditor's own bucket
    reports = result["reports"]
    assert len(reports) == 1
    report = reports[0]
    assert report["miner_uid"] == M_UID and report["auditor_uid"] == A_UID
    assert report["match"] is True and report["divergences"] == []
    assert report["replayed_theta_end"] == miner_artifacts.theta_end_root
    hotkey = f"local-{A_UID}"
    assert verify_report(report, lambda m, s: chain.verify(hotkey, m, s))
    assert app.reports[0].match

    # the auditor advertised itself on-chain
    assert chain.get_commitment(A_UID) == AUDITOR_COMMITMENT
    # the assignment used the consensus sampler on the post-window block hash
    block_hash = chain.block_hash(
        boundary_block(WINDOW + 1, manifest.start_block, manifest.blocks_per_window)
    )
    assert audit_sampler(
        twr.RUN_SEED, block_hash, WINDOW, [M_UID], 1.0, [A_UID]
    ) == {A_UID: [(M_UID, WINDOW)]}

    # after auditing, the replica applied window 0's outer step bitwise
    expected = tam.apply_outer_to(template_model, audit_cfg, {M_UID: miner_artifacts.payload})
    assert hash_named_tensors(app.model.iter_master_params()) == hash_named_tensors(
        expected.iter_master_params()
    )
    assert app.window == WINDOW + 1


# --------------------------------------------------------------------------- #
# Tampered commit → mismatch report with divergences
# --------------------------------------------------------------------------- #


def test_auditor_reports_mismatch_on_tampered_commit(
    audit_cfg: RunConfig,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path: Path,
    miner_artifacts: Any,
) -> None:
    tampered = make_commit(miner_artifacts, tampered=True)
    result = asyncio.run(
        _run_auditor(
            audit_cfg=audit_cfg,
            manifest=manifest,
            index=index,
            data_dir=data_dir,
            admin=admin,
            moto_endpoint=moto_endpoint,
            tmp=tmp_path,
            art=miner_artifacts,
            commit=tampered,
        )
    )
    assert result["code"] == 0
    reports = result["reports"]
    assert len(reports) == 1
    report = reports[0]
    assert report["match"] is False
    assert report["divergences"]  # non-empty evidence
    assert report["divergences"][0]["name"] == "<state_root>"
    assert report["committed_theta_end"] == tampered.theta_end_hash
    assert report["replayed_theta_end"] == miner_artifacts.theta_end_root
    hotkey = f"local-{A_UID}"
    assert verify_report(report, lambda m, s: result["chain"].verify(hotkey, m, s))


# --------------------------------------------------------------------------- #
# Wall-time budget: over-budget tasks are skipped, the replica still advances
# --------------------------------------------------------------------------- #


def test_auditor_wall_budget_skips_tasks(
    audit_cfg: RunConfig,
    manifest: RunManifest,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path: Path,
    template_model: MoKTransformer,
    miner_artifacts: Any,
) -> None:
    result = asyncio.run(
        _run_auditor(
            audit_cfg=audit_cfg,
            manifest=manifest,
            index=index,
            data_dir=data_dir,
            admin=admin,
            moto_endpoint=moto_endpoint,
            tmp=tmp_path,
            art=miner_artifacts,
            commit=make_commit(miner_artifacts),
            wall_budget_s=-1.0,  # every task is over budget
        )
    )
    assert result["code"] == 0
    assert result["reports"] == []            # nothing replayed or published
    app: AuditorApp = result["app"]
    assert app.reports == []
    # ... but lockstep was still maintained
    expected = tam.apply_outer_to(template_model, audit_cfg, {M_UID: miner_artifacts.payload})
    assert hash_named_tensors(app.model.iter_master_params()) == hash_named_tensors(
        expected.iter_master_params()
    )


def test_auditor_waits_out_a_pending_certificate(monkeypatch) -> None:
    """A down/lagging leader must stall the auditor, not crash it (the same
    guarantee the miner has): pending-certificate retries are unbounded."""
    import subnet.auditor.app as app_mod
    from subnet.core.checkpoint import CertificatePendingError

    calls = {"n": 0}

    async def flaky_catch_up(ctx, model, outer, *, from_window, to_window):
        calls["n"] += 1
        if calls["n"] <= 7:                       # far beyond catchup_retries=2
            raise CertificatePendingError("window 5: 2 on-chain commits but no leader certificate")

    monkeypatch.setattr(app_mod, "catch_up_replica", flaky_catch_up)
    app = app_mod.AuditorApp.__new__(app_mod.AuditorApp)
    app.ctx = None
    app.model = None
    app.outer_step = None
    app.catchup_retries = 2
    app.catchup_retry_s = 0.01
    asyncio.run(app._catch_up_retrying(from_window=4, to_window=5))
    assert calls["n"] == 8                        # 7 waits + the success


def test_align_replica_catches_up_to_resume_window(monkeypatch) -> None:
    """Checkpoint at w=100 (replica θ_start(101)) + state resume at 102 must
    catch up 101 BEFORE any replay (the window-102 PreconditionError crash)."""
    import subnet.auditor.app as app_mod

    calls: list[tuple[int, int]] = []

    async def record(self, *, from_window, to_window):
        calls.append((from_window, to_window))

    monkeypatch.setattr(app_mod.AuditorApp, "_catch_up_retrying", record)
    app = app_mod.AuditorApp.__new__(app_mod.AuditorApp)
    app.window = 102
    asyncio.run(app._align_replica(101))       # replica is at θ_start(101)
    assert calls == [(100, 101)]
    calls.clear()
    app.window = 101
    asyncio.run(app._align_replica(101))       # already aligned -> no-op
    assert calls == []
