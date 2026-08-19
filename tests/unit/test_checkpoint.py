"""Tests for C/core/checkpoint.py — DCP layout, build_outer_inputs, bitwise catch-up."""

from __future__ import annotations

import json
import uuid
from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from C.core.certificate import WindowCertificate
from C.core.checkpoint import (
    KIND_META,
    KIND_MODEL_TAR,
    KIND_OUTER_STATE,
    CatchUpError,
    Checkpointer,
    CheckpointError,
    CheckpointMeta,
    build_outer_inputs,
    catch_up,
    consensus_state_root,
    sparse_pairs_from_compressed,
)
from C.core.compress import (
    ChunkingTransformer,
    Quantizer,
    TopKCompressor,
    unpack_2bit_values,
    unpack_12bit_indices,
)
from C.core.exchange import AggregatorObject
from C.core.outer_opt import ReplicatedOuterStep
from C.core.payload import PayloadMeta, WindowPayload, serialize
from C.core.window_state import state_root
from mok_core.chain import WindowCommit
from mok_core.config.canonical import canonical_bytes
from mok_core.config.manifest import PRFSpec, RunManifest, VoidRange
from mok_core.config.schemas import (
    BucketCreds,
    CompressionConfig,
    OuterOptConfig,
    RunConfig,
    StorageConfig,
)
from mok_core.determinism.hashing import hash_bytes, tensor_bytes
from mok_core.storage import StorageClient, keys

TARGET_CHUNK = 4
TOPK = 3
MAX_BYTES = 1 << 20

COMP_SHAPES: dict[str, tuple[int, ...]] = {"layer.weight": (6, 10), "layer.bias": (7,)}
DENSE_SHAPES: dict[str, tuple[int, ...]] = {"router.balance_bias": (5,)}
ALL_SHAPES = {**COMP_SHAPES, **DENSE_SHAPES}
COMP_NAMES = sorted(COMP_SHAPES)

_HEX = "ab" * 32


# --------------------------------------------------------------------------- #
# moto fixtures (pattern from test_storage_client.py)
# --------------------------------------------------------------------------- #


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


def make_creds(bucket_name: str, access_key: str = "test-key") -> BucketCreds:
    return BucketCreds(
        account_id="testaccount",
        bucket_name=bucket_name,
        access_key_id=access_key,
        secret_access_key="test-secret",
    )


def fresh_bucket(admin: Any, tag: str) -> str:
    name = f"mok-{tag}-{uuid.uuid4().hex[:8]}"
    admin.create_bucket(Bucket=name)
    return name


def make_client(creds: BucketCreds, endpoint: str) -> StorageClient:
    return StorageClient(creds, StorageConfig(), endpoint_override=endpoint, retry_base_delay_s=0.01)


# --------------------------------------------------------------------------- #
# Model/compression helpers (fixture style from test_outer_opt.py)
# --------------------------------------------------------------------------- #


def _compressor() -> TopKCompressor:
    tf = ChunkingTransformer(COMP_SHAPES, target_chunk=TARGET_CHUNK)
    return TopKCompressor(tf, Quantizer(bins=4, range_sigmas=6.0), topk=TOPK)


def _fresh_params(seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {name: torch.randn(shape, generator=g) for name, shape in ALL_SHAPES.items()}


def _payload_for(uid: int, window: int) -> WindowPayload:
    comp = _compressor()
    g = torch.Generator().manual_seed(1000 * window + uid)
    compressed = {
        name: comp.compress(name, torch.randn(shape, generator=g))
        for name, shape in sorted(COMP_SHAPES.items())
    }
    dense = {
        name: torch.randn(shape, generator=g) for name, shape in sorted(DENSE_SHAPES.items())
    }
    meta = PayloadMeta(
        sample_digest="aa" * 32,
        sample_count=64,
        theta_end_hash="dd" * 32,
        state_root="cc" * 32,
        global_step=100,
        spec_version=1,
    )
    return WindowPayload(uid=uid, window=window, compressed=compressed, dense=dense, metadata=meta)


def _meta(window: int, root: str) -> CheckpointMeta:
    return CheckpointMeta(
        window=window,
        global_step=window * 500,
        tokens_consumed=window * 1_000_000,
        state_root=root,
        manifest_hash=_HEX,
        spec_version=1,
    )


def _manifest(**overrides: Any) -> RunManifest:
    m = RunManifest(
        spec_version=1,
        run_id="test-run",
        netuid=1,
        network="test",
        config_hash="00" * 32,
        container_digest="sha256:" + "0" * 64,
        mok_commit="deadbeef",
        tk_commit="cafebabe",
        attention_backend="cudnn_det",
        start_block=0,
        blocks_per_window=225,
        prf=PRFSpec(run_seed_hex="00" * 32),
        datasets=(),
        init_checkpoint_hash="00" * 32,
    )
    return m.model_copy(update=overrides) if overrides else m


def _cfg() -> RunConfig:
    return RunConfig(compression=CompressionConfig(target_chunk=TARGET_CHUNK, topk=TOPK))


# --------------------------------------------------------------------------- #
# CheckpointMeta canonicality
# --------------------------------------------------------------------------- #


def test_meta_canonical_bytes_and_round_trip():
    meta = _meta(7, _HEX)
    raw = meta.canonical()
    obj = json.loads(raw)
    assert list(obj) == sorted(obj)  # canonical JSON sorts keys
    assert CheckpointMeta.from_dict(obj) == meta
    assert CheckpointMeta.from_dict(obj).canonical() == raw


def test_meta_from_dict_is_strict():
    good = _meta(7, _HEX).to_dict()
    with pytest.raises(CheckpointError):
        CheckpointMeta.from_dict({**good, "extra": 1})
    missing = dict(good)
    del missing["state_root"]
    with pytest.raises(CheckpointError):
        CheckpointMeta.from_dict(missing)
    with pytest.raises(CheckpointError):
        CheckpointMeta.from_dict({**good, "window": True})  # bool is not an int here
    with pytest.raises(CheckpointError):
        CheckpointMeta.from_dict({**good, "spec_version": 0})


# --------------------------------------------------------------------------- #
# DCP save/load round trip
# --------------------------------------------------------------------------- #


def _params_with_bf16(seed: int = 3) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "layer.weight": torch.randn(6, 10, generator=g),
        "emb.weight": torch.randn(8, 4, generator=g).to(torch.bfloat16),
        "router.balance_bias": torch.randn(5, generator=g),
    }


def test_dcp_save_load_round_trip_bitwise(tmp_path):
    params = _params_with_bf16()
    step = ReplicatedOuterStep(OuterOptConfig(), {n: torch.Size(t.shape) for n, t in params.items()})
    g = torch.tensor([0.5, -0.5])
    step.apply(
        {"layer.weight": params["layer.weight"]},
        {"layer.weight": [(torch.tensor([0, 1]), g)]},
        {},
        {"layer.weight": torch.tensor([1.0])},
    )
    outer_state = step.state_dict()
    root = state_root(params.items())
    ck = Checkpointer(None, tmp_path / "ckpts")

    saved_dir = ck.save_local(7, params, outer_state, _meta(7, root))
    assert saved_dir == ck.window_dir(7)

    state, outer_loaded, meta = ck.load_local(7)
    assert meta == _meta(7, root)
    assert set(state) == set(params)
    for name, t in params.items():
        assert state[name].dtype == t.dtype, name
        assert tensor_bytes(state[name]) == tensor_bytes(t), name
    # state_root stability across DCP save/load — the bitwise resume guarantee.
    assert state_root(state.items()) == root == meta.state_root
    assert set(outer_loaded) == set(outer_state)
    for name in outer_state:
        assert tensor_bytes(outer_loaded[name]) == tensor_bytes(outer_state[name]), name

    # A fresh ReplicatedOuterStep resumes exactly from the loaded momentum.
    step2 = ReplicatedOuterStep(OuterOptConfig(), {n: torch.Size(t.shape) for n, t in params.items()})
    step2.load_state_dict(outer_loaded)
    for name, buf in step.state_dict().items():
        assert tensor_bytes(step2.state_dict()[name]) == tensor_bytes(buf), name


def test_checkpoint_layout_contract(tmp_path):
    params = _params_with_bf16()
    ck = Checkpointer(None, tmp_path)
    meta = _meta(3, state_root(params.items()))
    d = ck.save_local(3, params, {"layer.weight": torch.zeros(6, 10)}, meta)

    assert d.name == "w00000003"  # w{window:08d}
    assert (d / "model" / ".metadata").is_file()  # DCP directory
    assert (d / "outer_state.pt").is_file()
    assert (d / "meta.json").read_bytes() == canonical_bytes(meta.to_dict())
    saved = torch.load(d / "outer_state.pt", weights_only=True)
    assert set(saved) == {"outer"}  # torch.save({'outer': ...}) per contract


def test_save_local_rejects_mismatched_meta_and_empty_state(tmp_path):
    ck = Checkpointer(None, tmp_path)
    with pytest.raises(ValueError, match="meta.window"):
        ck.save_local(2, {"w": torch.ones(2)}, {}, _meta(3, _HEX))
    with pytest.raises(ValueError, match="empty"):
        ck.save_local(2, {}, {}, _meta(2, _HEX))


def test_prune_keeps_newest_local_checkpoints(tmp_path):
    ck = Checkpointer(None, tmp_path, keep_local=2)
    params = {"w": torch.ones(4)}
    for w in (1, 2, 3):
        ck.save_local(w, params, {}, _meta(w, _HEX))
    assert ck.local_windows() == [2, 3]
    assert not ck.window_dir(1).exists()


async def test_upload_and_remote_load_latest(admin: Any, moto_endpoint: str, tmp_path):
    params = _params_with_bf16(seed=11)
    outer_state = {"layer.weight": torch.full((6, 10), 0.25)}
    root = state_root(params.items())
    creds = make_creds(fresh_bucket(admin, "ckpt"))

    async with make_client(creds, moto_endpoint) as sc:
        ck_a = Checkpointer(sc, tmp_path / "node_a")
        await ck_a.save(9, params, outer_state, _meta(9, root))
        listed = await sc.list_keys(creds, "checkpoints/")
        assert listed == [
            keys.checkpoint_key(9, KIND_META),
            keys.checkpoint_key(9, KIND_MODEL_TAR),
            keys.checkpoint_key(9, KIND_OUTER_STATE),
        ]

        # A different node with an empty local dir restores from the bucket.
        ck_b = Checkpointer(sc, tmp_path / "node_b")
        loaded = await ck_b.load_latest(bucket=creds)
        assert loaded is not None
        state, outer_loaded, meta = loaded
        assert meta == _meta(9, root)
        assert state_root(state.items()) == root
        for name, t in params.items():
            assert tensor_bytes(state[name]) == tensor_bytes(t), name
        assert tensor_bytes(outer_loaded["layer.weight"]) == tensor_bytes(outer_state["layer.weight"])
        assert ck_b.local_windows() == [9]  # downloaded checkpoint is now local


async def test_load_latest_empty_everywhere_returns_none(tmp_path):
    ck = Checkpointer(None, tmp_path / "empty")
    assert await ck.load_latest() is None


# --------------------------------------------------------------------------- #
# build_outer_inputs — golden vs decompress and vs an independent index mapping
# --------------------------------------------------------------------------- #


def _reference_pairs(name: str, ct, comp: TopKCompressor) -> list[tuple[int, float]]:
    """Independent (pure-python) chunk->original index mapping, pad positions dropped."""
    g = comp.transformer.geometry(name)
    tc = comp.transformer.target_chunk
    n = ct.n_values
    local = unpack_12bit_indices(ct.idxs_packed, n).tolist()
    vals = comp.quantizer.dequantize(unpack_2bit_values(ct.codes_packed, n), ct.qparams).tolist()
    pairs: list[tuple[int, float]] = []
    for j in range(n):
        chunk, e = j // ct.topk, local[j]
        if g.mode == "grid":
            bpr = g.pad_cols // tc
            r = (chunk // bpr) * tc + e // tc
            c = (chunk % bpr) * tc + e % tc
            if r < g.rows and c < g.cols:
                pairs.append((r * g.cols + c, vals[j]))
        else:
            flat = chunk * g.chunk_elems + e
            if flat < g.numel:
                pairs.append((flat, vals[j]))
    return pairs


def test_sparse_pairs_match_decompress_and_reference():
    comp = _compressor()
    g = torch.Generator().manual_seed(5)
    for name, shape in COMP_SHAPES.items():
        ct = comp.compress(name, torch.randn(shape, generator=g))
        idx, vals = sparse_pairs_from_compressed(name, ct, comp)

        # Golden 1: scattering the pairs reproduces decompress() bitwise.
        dense = torch.zeros(int(torch.tensor(shape).prod()), dtype=torch.float32)
        dense[idx] = vals
        assert tensor_bytes(dense.view(shape)) == tensor_bytes(comp.decompress(name, ct)), name

        # Golden 2: an independent pure-python mapping agrees pair-for-pair.
        assert list(zip(idx.tolist(), vals.tolist(), strict=True)) == _reference_pairs(
            name, ct, comp
        ), name

        # The pre-clip norm equals the decompressed contribution's norm.
        assert float(torch.linalg.vector_norm(vals)) == pytest.approx(
            float(torch.linalg.vector_norm(comp.decompress(name, ct)))
        )


def test_sparse_pairs_drop_padding_positions():
    """A nearly-empty 1-D param forces top-k to select zero pads; they must not
    leak indices past the real element count."""
    comp = _compressor()
    t = torch.zeros(7)
    t[6] = 3.0  # single real value; chunk has 16 slots, 9 of them padding
    ct = comp.compress("layer.bias", t)
    idx, vals = sparse_pairs_from_compressed("layer.bias", ct, comp)
    assert int(idx.max()) < 7
    dense = torch.zeros(7)
    dense[idx] = vals
    assert tensor_bytes(dense) == tensor_bytes(comp.decompress("layer.bias", ct))


def test_build_outer_inputs_orders_and_norms():
    comp = _compressor()
    payloads = OrderedDict((uid, _payload_for(uid, window=4)) for uid in (1, 5, 9))
    sparse, dense, norms = build_outer_inputs(payloads, comp, COMP_NAMES)

    assert set(sparse) == set(COMP_NAMES)
    assert set(dense) == set(DENSE_SHAPES)
    for name in COMP_NAMES:
        assert len(sparse[name]) == 3  # one entry per peer, payload order
        assert norms[name].shape == (3,)
        for k, uid in enumerate((1, 5, 9)):
            ref_idx, ref_vals = sparse_pairs_from_compressed(
                name, payloads[uid].compressed[name], comp
            )
            assert torch.equal(sparse[name][k][0], ref_idx)
            assert torch.equal(sparse[name][k][1], ref_vals)
            assert norms[name][k] == pytest.approx(float(torch.linalg.vector_norm(ref_vals)))
    for name in DENSE_SHAPES:
        assert len(dense[name]) == 3
        for k, uid in enumerate((1, 5, 9)):
            assert torch.equal(dense[name][k], payloads[uid].dense[name])
            assert norms[name][k] == pytest.approx(
                float(torch.linalg.vector_norm(payloads[uid].dense[name]))
            )


def test_build_outer_inputs_validation():
    comp = _compressor()
    out_of_order = OrderedDict(((5, _payload_for(5, 4)), (1, _payload_for(1, 4))))
    with pytest.raises(ValueError, match="uid-ascending"):
        build_outer_inputs(out_of_order, comp, COMP_NAMES)
    payload = _payload_for(1, 4)
    del payload.compressed["layer.bias"]
    with pytest.raises(ValueError, match="missing compressed"):
        build_outer_inputs(OrderedDict({1: payload}), comp, COMP_NAMES)
    assert build_outer_inputs(OrderedDict(), comp, COMP_NAMES) == ({}, {}, {})


# --------------------------------------------------------------------------- #
# consensus_state_root
# --------------------------------------------------------------------------- #


def _commit(window: int, root: str, payload_hash: str = "11" * 32) -> WindowCommit:
    return WindowCommit(
        window=window, payload_hash=payload_hash, state_root=root, theta_end_hash="dd" * 32
    )


def test_consensus_state_root_majority_and_ties():
    a, b = "aa" * 32, "bb" * 32
    assert consensus_state_root({}) is None
    assert consensus_state_root({1: _commit(0, a)}) == a
    assert consensus_state_root({1: _commit(0, a), 2: _commit(0, b), 3: _commit(0, b)}) == b
    assert consensus_state_root({1: _commit(0, b), 2: _commit(0, a)}) == a  # tie -> lexicographic


# --------------------------------------------------------------------------- #
# catch_up — bitwise replay over fabricated windows
# --------------------------------------------------------------------------- #


class _Network:
    """Fabricated run history: certified windows, aggregator mirrors, chain commits."""

    def __init__(self, windows: list[int], peers: tuple[int, ...] = (1, 2), seed: int = 0):
        self.cfg = _cfg()
        self.windows = windows
        self.certs: dict[int, WindowCertificate] = {}
        self.aggs: dict[int, bytes] = {}
        self.commits: dict[int, dict[int, WindowCommit]] = {}

        comp = _compressor()
        self.params = _fresh_params(seed)
        self.init_params = {n: t.clone() for n, t in self.params.items()}
        self.outer = ReplicatedOuterStep(
            self.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()}
        )
        for w in windows:
            start_root = state_root(self.params.items())
            blobs = {uid: serialize(_payload_for(uid, w)) for uid in peers}
            self.certs[w] = WindowCertificate(
                window=w,
                included_uids=tuple(sorted(blobs)),
                payload_hashes={uid: hash_bytes(b) for uid, b in blobs.items()},
                theta_start_root=start_root,
                leader_uid=0,
            )
            self.aggs[w] = AggregatorObject(window=w, payloads=blobs).serialize()
            self.commits[w] = {
                uid: _commit(w, start_root, payload_hash=hash_bytes(blobs[uid])) for uid in peers
            }
            payloads = OrderedDict((uid, _payload_for(uid, w)) for uid in sorted(blobs))
            sparse, dense, norms = build_outer_inputs(payloads, comp, COMP_NAMES)
            self.outer.apply(self.params, sparse, dense, norms)
        head = max(windows) + 1
        self.commits[head] = {
            uid: _commit(head, state_root(self.params.items())) for uid in peers
        }

    def chain(self) -> MagicMock:
        chain = MagicMock()
        chain.get_window_commits.side_effect = lambda w: dict(self.commits.get(w, {}))
        return chain

    def publish(self, admin: Any, leader: BucketCreds) -> None:
        for w in self.windows:
            admin.put_object(
                Bucket=leader.bucket_name,
                Key=keys.certificate_key(w),
                Body=canonical_bytes(self.certs[w]),
            )
            admin.put_object(Bucket=leader.bucket_name, Key=keys.aggregator_key(w), Body=self.aggs[w])


async def test_catch_up_happy_path_two_windows(admin: Any, moto_endpoint: str):
    net = _Network(windows=[1, 2])
    leader = make_creds(fresh_bucket(admin, "culeader"))
    net.publish(admin, leader)

    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    own = make_creds(fresh_bucket(admin, "cuself"))
    async with make_client(own, moto_endpoint) as sc:
        report = await catch_up(
            params,
            outer,
            None,  # real C.core.exchange
            sc,
            net.chain(),
            _manifest(),
            net.cfg,
            0,
            2,
            leader_bucket=leader,
            max_bytes=MAX_BYTES,
        )

    assert report.applied_windows == (1, 2)
    assert report.skipped_void == () and report.unverified_windows == ()
    assert report.final_root == state_root(net.params.items())
    for name in ALL_SHAPES:  # bitwise lockstep with the fabricated network
        assert tensor_bytes(params[name]) == tensor_bytes(net.params[name]), name
    for name, buf in net.outer.state_dict().items():
        assert tensor_bytes(outer.state_dict()[name]) == tensor_bytes(buf), name


async def test_catch_up_apply_fn_replaces_default_application(admin: Any, moto_endpoint: str):
    net = _Network(windows=[1, 2], seed=7)
    leader = make_creds(fresh_bucket(admin, "cufn"))
    net.publish(admin, leader)

    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    comp = _compressor()
    seen: list[int] = []

    def apply_fn(gather) -> None:
        seen.append(len(gather.payloads))
        sparse, dense, norms = build_outer_inputs(gather.payloads, comp, COMP_NAMES)
        outer.apply(params, sparse, dense, norms)

    own = make_creds(fresh_bucket(admin, "cufnself"))
    async with make_client(own, moto_endpoint) as sc:
        report = await catch_up(
            params, outer, None, sc, net.chain(), _manifest(), net.cfg, 0, 2,
            apply_fn=apply_fn, leader_bucket=leader, max_bytes=MAX_BYTES,
        )
    assert seen == [2, 2]
    assert report.final_root == state_root(net.params.items())


async def test_catch_up_skips_void_windows(admin: Any, moto_endpoint: str):
    net = _Network(windows=[1], seed=2)  # the surviving lineage applies only window 1
    leader = make_creds(fresh_bucket(admin, "cuvoid"))
    net.publish(admin, leader)
    # Window 2 was voided by a rollback; head commits sit at window 3.
    net.commits[3] = net.commits.pop(2)
    manifest = _manifest(
        void_ranges=(VoidRange(first_window=2, last_window=2, reseed_salt_hex="ab"),)
    )

    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    own = make_creds(fresh_bucket(admin, "cuvoidself"))
    async with make_client(own, moto_endpoint) as sc:
        report = await catch_up(
            params, outer, None, sc, net.chain(), manifest, net.cfg, 0, 2,
            leader_bucket=leader, max_bytes=MAX_BYTES,
        )
    assert report.applied_windows == (1,)
    assert report.skipped_void == (2,)
    assert report.final_root == state_root(net.params.items())


async def test_catch_up_state_root_mismatch_raises_with_report(admin: Any, moto_endpoint: str):
    net = _Network(windows=[1, 2], seed=3)
    leader = make_creds(fresh_bucket(admin, "cudiv"))
    net.publish(admin, leader)
    # Poison the consensus for window 2: every miner "committed" a foreign root.
    bad_root = "ff" * 32
    net.commits[2] = {
        uid: _commit(2, bad_root, payload_hash=c.payload_hash)
        for uid, c in net.commits[2].items()
    }

    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    own = make_creds(fresh_bucket(admin, "cudivself"))
    async with make_client(own, moto_endpoint) as sc:
        with pytest.raises(CatchUpError) as excinfo:
            await catch_up(
                params, outer, None, sc, net.chain(), _manifest(), net.cfg, 0, 2,
                leader_bucket=leader, max_bytes=MAX_BYTES,
            )
    div = excinfo.value.divergence
    assert div is not None
    assert div.window == 2
    assert div.expected_root == bad_root
    assert div.actual_root == net.certs[2].theta_start_root  # true θ_start(2) after window 1
    assert "commits" in div.detail


async def test_catch_up_missing_certified_payload_is_fatal(admin: Any, moto_endpoint: str):
    net = _Network(windows=[1], seed=4)
    leader = make_creds(fresh_bucket(admin, "cumiss"))
    net.publish(admin, leader)
    # Replace the mirror with one that lost uid 2 — lockstep is impossible.
    blobs = {1: serialize(_payload_for(1, 1))}
    admin.put_object(
        Bucket=leader.bucket_name,
        Key=keys.aggregator_key(1),
        Body=AggregatorObject(window=1, payloads=blobs).serialize(),
    )

    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    own = make_creds(fresh_bucket(admin, "cumissself"))
    async with make_client(own, moto_endpoint) as sc:
        with pytest.raises(CatchUpError, match="unavailable"):
            await catch_up(
                params, outer, None, sc, net.chain(), _manifest(), net.cfg, 0, 1,
                leader_bucket=leader, max_bytes=MAX_BYTES,
            )


async def test_catch_up_skips_empty_windows_as_identity(admin: Any, moto_endpoint: str):
    """Windows with NO commits and NO certificate (pre-launch idle, fleet downtime)
    are the identity outer step: θ unchanged, window recorded as unverified."""
    net = _Network(windows=[3])                      # only window 3 was ever mined
    leader = make_creds(fresh_bucket(admin, "cu-empty-leader"))
    net.publish(admin, leader)

    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    own = make_creds(fresh_bucket(admin, "cu-empty-self"))
    async with make_client(own, moto_endpoint) as sc:
        report = await catch_up(
            params, outer, None, sc, net.chain(), _manifest(), net.cfg,
            0, 3, leader_bucket=leader, max_bytes=MAX_BYTES,
        )

    assert report.applied_windows == (3,)
    assert report.unverified_windows == (1, 2)       # empty → identity, unverified
    assert report.final_root == state_root(net.params.items())   # still bitwise lockstep


async def test_catch_up_missing_certificate_with_commits_is_an_error(
    admin: Any, moto_endpoint: str
):
    """Commits on-chain but no leader certificate is a REAL inconsistency."""
    net = _Network(windows=[1])
    leader = make_creds(fresh_bucket(admin, "cu-nocert-leader"))
    # publish nothing: certificate for window 1 deliberately absent

    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    own = make_creds(fresh_bucket(admin, "cu-nocert-self"))
    async with make_client(own, moto_endpoint) as sc:
        with pytest.raises(CatchUpError, match="no leader certificate"):
            await catch_up(
                params, outer, None, sc, net.chain(), _manifest(), net.cfg,
                0, 1, leader_bucket=leader, max_bytes=MAX_BYTES,
            )


async def test_pending_certificate_is_distinct_wait_condition(admin: Any, moto_endpoint: str):
    """Commits-without-certificate raises CertificatePendingError (a subclass) so
    apps can WAIT on a lagging leader instead of treating it as corruption."""
    from C.core.checkpoint import CertificatePendingError

    net = _Network(windows=[1])
    leader = make_creds(fresh_bucket(admin, "cu-pend-leader"))   # no cert published
    params = {n: t.clone() for n, t in net.init_params.items()}
    outer = ReplicatedOuterStep(net.cfg.outer, {n: torch.Size(s) for n, s in ALL_SHAPES.items()})
    own = make_creds(fresh_bucket(admin, "cu-pend-self"))
    async with make_client(own, moto_endpoint) as sc:
        with pytest.raises(CertificatePendingError):
            await catch_up(
                params, outer, None, sc, net.chain(), _manifest(), net.cfg,
                0, 1, leader_bucket=leader, max_bytes=MAX_BYTES,
            )
