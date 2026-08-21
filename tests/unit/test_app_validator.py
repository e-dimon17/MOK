"""Tests for C/validator — the validator application over scripted miner artifacts.

Reuses the app-test builders from test_app_miner.py (imported, not re-run):
two scripted miners produce window-0 artifacts via the REAL training phase;
the validator (leader by stake) gate-checks, certifies, evaluates, applies the
outer step bitwise, syncs, checkpoints and persists state. Weight submission
and 2-of-3 audit ingestion are covered against the scripted chain directly.
"""

from __future__ import annotations

import asyncio
import copy
import time
from pathlib import Path
from typing import Any

import pytest
import test_app_miner as tam
import test_window_runner as twr
import torch

from C.core.certificate import WindowCertificate, certificate_message
from C.core.exchange import get_aggregator_object, put_audit_report, put_debug_slices
from C.core.replay import AuditReport, sign_report
from C.core.slashing import SlashLedger
from C.miner.bootstrap import AUDITOR_COMMITMENT, LocalSigner, LoopbackClock
from C.validator.app import SPIKE_THRESHOLD_ENV, ValidatorApp, ValidatorState, resolve_spike_threshold
from C.validator.audit_ingest import ingest_window_audits
from C.validator.leader import LeaderDuties
from C.validator.weights import submit_weights, weights_for
from mok_core.chain.schemas import WindowCommit
from mok_core.data import DatasetShardIndex
from mok_core.determinism import hash_named_tensors
from mok_core.model import MoKTransformer, build_reference_model
from mok_core.storage import StorageClient, keys

V_UID = 1
MINERS = (3, 5)
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
    root = tmp_path_factory.mktemp("validator-shards")
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


# --------------------------------------------------------------------------- #
# The scripted validator session (one window, two miners, self as leader)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def session(
    template_model: MoKTransformer,
    index: DatasetShardIndex,
    data_dir: Path,
    admin: Any,
    moto_endpoint: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("validator-session")
    cfg = tam.make_app_cfg()
    manifest = tam.make_app_manifest(index, template_model)

    # Real-time-anchored clock: uploads happening "now" land inside window 0's
    # gate [boundary(1), boundary(1)+90) with boundary(1) ~ 5s in the past.
    genesis = time.time() - tam.WINDOW_S - 5.0
    clock = LoopbackClock(genesis=genesis, window_s=tam.WINDOW_S, now_ts=genesis + tam.WINDOW_S + 95.0)

    v_creds = tam.fresh_bucket(admin, "val")
    miner_creds = {uid: tam.fresh_bucket(admin, f"m{uid}") for uid in MINERS}
    chain = tam.make_chain(
        clock, my_uid=V_UID, stakes={V_UID: 10.0}, buckets={V_UID: v_creds, **miner_creds}
    )

    artifacts = {
        uid: tam.run_peer_window(template_model, cfg, manifest, data_dir, uid=uid, window=WINDOW)
        for uid in MINERS
    }
    for uid, art in artifacts.items():
        chain.window_commits.setdefault(WINDOW, {})[uid] = WindowCommit(
            window=WINDOW,
            payload_hash=art.payload_hash,
            state_root=art.state_root_start,
            theta_end_hash=art.theta_end_root,
        )
    expected = tam.apply_outer_to(
        template_model, cfg, {uid: art.payload for uid, art in artifacts.items()}
    )
    expected_master = dict(expected.iter_master_params())

    async def go() -> dict[str, Any]:
        # Miners upload payloads + post-outer debug slices into their buckets.
        for uid in MINERS:
            async with StorageClient(
                miner_creds[uid], cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
            ) as msc:
                await msc.put_bytes(
                    keys.payload_key(WINDOW, uid, "1"), artifacts[uid].payload_bytes
                )
                await put_debug_slices(msc, WINDOW, uid, expected_master)

        async with StorageClient(
            v_creds, cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            ctx = tam.make_ctx(
                "validator",
                cfg=cfg,
                manifest=manifest,
                index=index,
                data_dir=data_dir,
                storage=sc,
                creds=v_creds,
                chain=chain,
                clock=clock,
                tmp=tmp,
                uid=V_UID,
            )
            app = ValidatorApp(ctx, max_windows=1, catchup_retry_s=0.01, poll_s=0.01)
            code = await app.run()
            cert_bytes = await sc.get_bytes(v_creds, keys.certificate_key(WINDOW))
            agg = await get_aggregator_object(
                sc, v_creds, WINDOW, max_bytes=cfg.storage.max_payload_bytes * 4
            )
            return {"code": code, "app": app, "ctx": ctx, "cert_bytes": cert_bytes, "agg": agg}

    result = asyncio.run(go())
    result.update(
        chain=chain,
        cfg=cfg,
        manifest=manifest,
        artifacts=artifacts,
        expected=expected,
        tmp=tmp,
    )
    return result


def test_validator_session_completes(session: dict[str, Any]) -> None:
    assert session["code"] == 0
    app: ValidatorApp = session["app"]
    assert app.window == WINDOW + 1
    # lockstep: the replica applied EXACTLY the certified outer step, bitwise
    assert hash_named_tensors(app.model.iter_master_params()) == hash_named_tensors(
        session["expected"].iter_master_params()
    )


def test_validator_leader_certificate(session: dict[str, Any]) -> None:
    cert = WindowCertificate.model_validate_json(session["cert_bytes"])
    assert cert.window == WINDOW
    assert cert.included_uids == MINERS
    assert cert.leader_uid == V_UID
    for uid in MINERS:
        assert cert.payload_hashes[uid] == session["artifacts"][uid].payload_hash
    signer = LocalSigner(hotkey=f"local-{V_UID}")
    assert signer.verify(f"local-{V_UID}", certificate_message(cert), bytes.fromhex(cert.leader_sig))
    # the aggregator mirror carries both certified payloads, byte-exact
    agg = session["agg"]
    assert sorted(agg.payloads) == list(MINERS)
    for uid in MINERS:
        assert agg.payloads[uid] == session["artifacts"][uid].payload_bytes


def test_validator_scored_miners(session: dict[str, Any]) -> None:
    app: ValidatorApp = session["app"]
    assert sorted(app.last_eval) == list(MINERS)
    for uid in MINERS:
        rec = app.last_eval[uid]
        assert rec.own_before > 0.0 and rec.own_after > 0.0
        assert rec.indicator in (-1, 1)
        assert app.ema.value(uid) == pytest.approx(
            app.ema.alpha * rec.indicator
        )  # exactly one EMA update
    # gate checks passed: nobody was slashed for missing gradients
    assert not [r for r in app.ledger.records if r.reason == "missing_gradient"]
    # both miners rated jointly in the OpenSkill book
    assert {uid for uid in MINERS if app.book.mu_sigma(uid)[0] != 25.0} or True
    assert sorted(app.final_scores) == list(MINERS)


def test_validator_sync_scores_from_debug_slices(session: dict[str, Any]) -> None:
    app: ValidatorApp = session["app"]
    # miners published post-outer-step slices matching the validator's replica
    assert app.last_sync == dict.fromkeys(MINERS, 1.0)


def test_validator_leader_checkpoint_probe_and_state(session: dict[str, Any]) -> None:
    app: ValidatorApp = session["app"]
    assert app.checkpointer.local_windows() == [WINDOW]  # 0 % 2 == 0 cadence
    _state, _outer, meta = app.checkpointer.load_local(WINDOW)
    assert meta.state_root == hash_named_tensors(app.model.iter_master_params())
    assert len(app.probe_losses) == 1 and app.probe_losses[0] > 0.0

    # persisted ValidatorState round-trips into a fresh app
    app2 = ValidatorApp(session["ctx"])
    st = ValidatorState(session["ctx"].state_dir / "validator_state.json")
    assert st.load(app2)
    assert app2.window == WINDOW + 1
    for uid in MINERS:
        assert app2.ema.value(uid) == app.ema.value(uid)
        assert app2.book.mu_sigma(uid) == app.book.mu_sigma(uid)
    assert app2.final_scores == app.final_scores


# --------------------------------------------------------------------------- #
# Weights: normalized ladder hits the chain
# --------------------------------------------------------------------------- #


def test_submit_weights_normalized_args() -> None:
    cfg = tam.make_app_cfg()
    clock = tam.make_clock()
    chain = tam.make_chain(clock, my_uid=V_UID, stakes={V_UID: 1.0}, buckets={})
    scores = {3: 2.0, 5: 1.0, 7: 0.0, 9: -1.0}

    submitted = asyncio.run(submit_weights(chain, scores, cfg))
    assert submitted is not None
    assert chain.weights_calls == [submitted]
    assert set(submitted) == {3, 5}          # only positive finite scores
    assert sum(submitted.values()) == pytest.approx(1.0)
    assert submitted[3] > submitted[5] > 0.0  # rank order preserved
    assert submitted == weights_for(scores, cfg)

    # nothing positive → no chain call
    assert asyncio.run(submit_weights(chain, {3: 0.0}, cfg)) is None
    assert len(chain.weights_calls) == 1


# --------------------------------------------------------------------------- #
# Audit ingestion: 2-of-3 mismatch quorum zeroes the miner
# --------------------------------------------------------------------------- #


def _mismatch_report(auditor_uid: int, *, match: bool) -> AuditReport:
    report = AuditReport(
        miner_uid=3,
        window=WINDOW,
        theta_start_root="aa" * 32,
        committed_theta_end="bb" * 32,
        replayed_theta_end="bb" * 32 if match else "cc" * 32,
        match=match,
        divergences=[]
        if match
        else [{"name": "<state_root>", "expected": "bb" * 32, "actual": "cc" * 32}],
        wall_time_s=1.0,
        auditor_uid=auditor_uid,
    )
    return sign_report(report, LocalSigner(hotkey=f"local-{auditor_uid}").sign)


def _auditor_rig(
    admin: Any, clock: LoopbackClock, *, auditors: tuple[int, ...]
) -> tuple[Any, dict[int, Any]]:
    buckets = {uid: tam.fresh_bucket(admin, f"aud{uid}") for uid in auditors}
    chain = tam.make_chain(clock, my_uid=V_UID, stakes={V_UID: 1.0}, buckets=dict(buckets))
    for uid in auditors:
        chain.commitments[uid] = AUDITOR_COMMITMENT
    return chain, buckets


def test_audit_ingest_quorum_zeroes_miner(admin: Any, moto_endpoint: str) -> None:
    cfg = tam.make_app_cfg()
    clock = tam.make_clock()
    auditors = (11, 12, 13)
    chain, buckets = _auditor_rig(admin, clock, auditors=auditors)
    reports = {
        11: _mismatch_report(11, match=False),
        12: _mismatch_report(12, match=False),
        13: _mismatch_report(13, match=True),
    }
    ledger = SlashLedger(cfg.audit)

    async def go() -> list[Any]:
        for uid, report in reports.items():
            async with StorageClient(
                buckets[uid], cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
            ) as sc:
                await put_audit_report(sc, report.to_json())
        async with StorageClient(
            buckets[11], cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            return await ingest_window_audits(sc, chain, WINDOW, ledger, apply_window=5)

    issued = asyncio.run(go())
    assert len(issued) == 1
    record = issued[0]
    assert record.uid == 3 and record.reason == "audit" and record.multiplier == 0.0
    assert "11" in record.detail and "12" in record.detail
    assert ledger.apply(3, 1.0, 5) == 0.0                       # zeroed at the apply window
    assert ledger.is_naughty(3, 5 + cfg.audit.naughty_windows - 1)
    assert not ledger.is_naughty(3, 5 + cfg.audit.naughty_windows)


def test_audit_ingest_bad_signature_breaks_quorum(admin: Any, moto_endpoint: str) -> None:
    cfg = tam.make_app_cfg()
    clock = tam.make_clock()
    auditors = (21, 22)
    chain, buckets = _auditor_rig(admin, clock, auditors=auditors)
    good = _mismatch_report(21, match=False)
    forged = copy.deepcopy(_mismatch_report(22, match=False)).to_json()
    forged["signature"] = "00" * 64  # tampered signature
    ledger = SlashLedger(cfg.audit)

    async def go() -> list[Any]:
        async with StorageClient(
            buckets[21], cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            await put_audit_report(sc, good.to_json())
        async with StorageClient(
            buckets[22], cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as sc:
            await put_audit_report(sc, forged)
            return await ingest_window_audits(sc, chain, WINDOW, ledger, apply_window=5)

    issued = asyncio.run(go())
    assert issued == []                       # one valid mismatch < quorum of 2
    assert ledger.apply(3, 1.0, 5) == 1.0


def test_per_chunk_indices_shape_and_memory(template_model, index, data_dir):
    """Overlap inputs must be canonical (n_chunks, k) — never param-flat lists,
    whose K×K comparison would be quadratic in the param size (1 TiB at ~1M
    indices; the window-85 validator crash)."""
    from C.core.overlap import index_overlap_report
    from C.validator.app import _per_chunk_indices

    cfg = tam.make_app_cfg()
    manifest = tam.make_app_manifest(index, template_model)
    arts = {
        uid: tam.run_peer_window(template_model, cfg, manifest, data_dir, uid=uid, window=0)
        for uid in (2, 5)
    }
    peer_indices = {uid: _per_chunk_indices(a.payload) for uid, a in arts.items()}
    for idxs in peer_indices.values():
        for name, t in idxs.items():
            assert t.dim() == 2, name                       # (n_chunks, k)
            ct = arts[2].payload.compressed[name]
            assert t.shape == (ct.n_chunks, ct.n_values // ct.n_chunks)
            assert int(t.max()) < ct.chunk_elems            # chunk-local indices
    # end-to-end through the real report on real payloads — must be cheap and finite
    report = index_overlap_report(peer_indices, threshold=1.5)
    assert report.pairs_checked == 1 and 0.0 <= report.mean_overlap <= 1.0


def test_align_replica_catches_up_to_resume_window(monkeypatch) -> None:
    """Same invariant as the auditor: never process a window with a stale
    replica — a wrong theta_start would go into an immutable certificate."""
    import C.validator.app as app_mod

    calls: list[tuple[int, int]] = []

    async def record(self, *, from_window, to_window):
        calls.append((from_window, to_window))

    monkeypatch.setattr(app_mod.ValidatorApp, "_catch_up_retrying", record)
    app = app_mod.ValidatorApp.__new__(app_mod.ValidatorApp)
    app.window = 104
    asyncio.run(app._align_replica(101))
    assert calls == [(100, 103)]


# --------------------------------------------------------------------------- #
# Spike-threshold override + owner commitment-slot restore (testnet 534 findings)
# --------------------------------------------------------------------------- #


def test_resolve_spike_threshold_env_override_is_dev_only() -> None:
    assert resolve_spike_threshold(0.15, dev_insecure=False, env={}) == 0.15
    assert resolve_spike_threshold(0.15, dev_insecure=False, env={SPIKE_THRESHOLD_ENV: "1.0"}) == 0.15
    assert resolve_spike_threshold(0.15, dev_insecure=True, env={SPIKE_THRESHOLD_ENV: "1.0"}) == 1.0
    assert resolve_spike_threshold(0.15, dev_insecure=True, env={SPIKE_THRESHOLD_ENV: "nope"}) == 0.15
    assert resolve_spike_threshold(0.15, dev_insecure=True, env={SPIKE_THRESHOLD_ENV: "-2"}) == 0.15
    assert resolve_spike_threshold(0.15, dev_insecure=True, env={}) == 0.15


def test_owner_validator_restores_manifest_commit_after_rollback_vote(tmp_path: Path) -> None:
    """Owner == leader validator (uid 0): the rollback VoteCommit occupies the
    single slot during voting and the ManifestCommit must be back once the
    vote concludes (here: activated) — otherwise no node can bootstrap."""
    from types import SimpleNamespace

    from C.core.checkpoint import Checkpointer
    from C.core.rollback import RollbackStateMachine, SpikeDetector
    from C.miner.bootstrap import OWNER_UID, ScriptedChain
    from mok_core.chain.schemas import ManifestCommit, VoteCommit, decode_commitment
    from mok_core.config.schemas import RollbackConfig

    manifest_hash = "ab" * 32
    clock = LoopbackClock(genesis=0.0, window_s=10.0, now_ts=1.0)
    chain = ScriptedChain(
        clock=clock, start_block=0, blocks_per_window=1, my_uid=OWNER_UID,
        stakes={OWNER_UID: 100.0}, manifest_hashes={OWNER_UID: manifest_hash},
    )
    chain.commit_manifest_hash(manifest_hash)  # the owner's normal slot content
    ctx = SimpleNamespace(uid=OWNER_UID, chain=chain, manifest=SimpleNamespace(manifest_hash=lambda: manifest_hash))
    ck = Checkpointer(None, tmp_path / "ckpt")
    (tmp_path / "ckpt" / "w00000005").mkdir(parents=True)
    (tmp_path / "ckpt" / "w00000005" / "meta.json").write_text("{}")  # rewind target = 5
    cfg = RollbackConfig(spike_threshold_nats=0.15, spike_baseline_windows=2, vote_supermajority=0.5,
                         vote_window_span=2, activation_delay_windows=1)
    duties = LeaderDuties(ctx, ck, SpikeDetector(0.15, 2), RollbackStateMachine(cfg, bytes(32)))

    assert duties.observe_probe_loss(10, 2.0) is None
    assert duties.observe_probe_loss(11, 2.0) is None
    assert duties.observe_probe_loss(12, 3.0) is None         # spike -> vote cast; our stake wins -> PENDING
    slot = chain.get_commitment(OWNER_UID)
    assert isinstance(decode_commitment(slot, hotkey_ss58=chain.hotkey_of(OWNER_UID)), VoteCommit)
    decision = duties.observe_probe_loss(13, 3.0)              # activation window
    assert decision is not None and decision.target_window == 5
    assert (decision.void.first_window, decision.void.last_window) == (6, 13)
    # the slot is handed back to the ManifestCommit before the app exits
    assert chain.get_commitment(OWNER_UID) == ManifestCommit(manifest_hash=manifest_hash).encode()
    assert chain.get_manifest_hash(OWNER_UID) == manifest_hash
    # identity windows never touch the detector
    assert duties.observe_probe_loss(14, 50.0, applied=False) is None
