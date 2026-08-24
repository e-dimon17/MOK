"""Tests for fleet/onboarding/init_publish.py — the seed-42 init, end to end on CPU
with moto S3, a MagicMock chain and a tiny reference-backend config."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import test_window_runner as twr
import torch

from fleet.onboarding.init_publish import (
    InitPublishError,
    build_and_publish_init,
    fetch_and_verify_init,
)
from mok_core.config import RunConfig
from mok_core.config.schemas import BucketCreds
from mok_core.determinism import hash_named_tensors
from mok_core.model import build_reference_model
from mok_core.storage import StorageClient

SEED = 42
OWNER_UID = 1


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def cfg() -> RunConfig:
    return twr.make_run_cfg()


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


def fresh_bucket(admin: Any) -> BucketCreds:
    name = f"mok-init-{uuid.uuid4().hex[:8]}"
    admin.create_bucket(Bucket=name)
    return BucketCreds(
        account_id="testaccount",
        bucket_name=name,
        access_key_id="test-key",
        secret_access_key="test-secret",
    )


@pytest.fixture(scope="module")
def published(cfg: RunConfig, admin: Any, moto_endpoint: str, tmp_path_factory) -> dict[str, Any]:
    """One full owner-side publish, shared by the verification tests."""
    tmp = tmp_path_factory.mktemp("init-owner")
    creds = fresh_bucket(admin)
    chain = MagicMock()

    async def go() -> str:
        async with StorageClient(
            creds, cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as storage:
            return await build_and_publish_init(
                cfg, storage, chain, local_dir=tmp / "ckpt", seed=SEED, backend="reference"
            )

    root = asyncio.run(go())
    return {"root": root, "creds": creds, "chain": chain, "local_dir": tmp / "ckpt"}


# --------------------------------------------------------------------------- #
# Owner side
# --------------------------------------------------------------------------- #


def test_publish_returns_the_init_state_root(published, cfg: RunConfig) -> None:
    reference = build_reference_model(cfg.model, SEED)
    assert published["root"] == hash_named_tensors(reference.iter_master_params())


def test_publish_commits_the_root_on_chain(published) -> None:
    chain = published["chain"]
    chain.commit_manifest_hash.assert_called_once_with(published["root"])


def test_publish_wrote_the_window0_checkpoint_layout(published) -> None:
    d = Path(published["local_dir"]) / "w00000000"
    assert (d / "model").is_dir()
    assert (d / "outer_state.pt").is_file()
    assert (d / "meta.json").is_file()


def test_publish_is_deterministic(cfg: RunConfig, tmp_path: Path) -> None:
    async def local_only() -> str:
        return await build_and_publish_init(
            cfg, None, None, local_dir=tmp_path / "a", seed=SEED, backend="reference"
        )

    async def local_only_b() -> str:
        return await build_and_publish_init(
            cfg, None, None, local_dir=tmp_path / "b", seed=SEED, backend="reference"
        )

    assert asyncio.run(local_only()) == asyncio.run(local_only_b())


def test_publish_different_seed_different_root(cfg: RunConfig, tmp_path: Path, published) -> None:
    async def go() -> str:
        return await build_and_publish_init(
            cfg, None, None, local_dir=tmp_path, seed=SEED + 1, backend="reference"
        )

    assert asyncio.run(go()) != published["root"]


# --------------------------------------------------------------------------- #
# Miner side
# --------------------------------------------------------------------------- #


def _fetch(cfg, moto_endpoint, published, tmp_path, *, expected=None, chain=None, owner_uid=None):
    miner_creds = published["creds"]  # any creds that can read the owner bucket

    async def go():
        async with StorageClient(
            miner_creds, cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as storage:
            return await fetch_and_verify_init(
                storage,
                chain,
                expected if expected is not None else published["root"],
                local_dir=tmp_path / "miner-ckpt",
                bucket=published["creds"],
                owner_uid=owner_uid,
            )

    return asyncio.run(go())


def test_fetch_and_verify_roundtrip(cfg, moto_endpoint, published, tmp_path) -> None:
    state, outer, meta = _fetch(cfg, moto_endpoint, published, tmp_path)
    assert meta.window == 0
    assert meta.state_root == published["root"]
    assert hash_named_tensors(state.items()) == published["root"]
    # fresh outer momentum: all zeros, one tensor per master param
    assert set(outer) == set(state)
    assert all(torch.all(t == 0) for t in outer.values())
    # bitwise identical to a locally-built reference init
    reference = dict(build_reference_model(cfg.model, SEED).iter_master_params())
    for name, tensor in state.items():
        assert torch.equal(tensor, reference[name].detach().cpu())


def test_fetch_rejects_wrong_expected_root(cfg, moto_endpoint, published, tmp_path) -> None:
    with pytest.raises(InitPublishError, match="state_root"):
        _fetch(cfg, moto_endpoint, published, tmp_path, expected="00" * 32)


def test_fetch_checks_chain_commitment_when_owner_given(
    cfg, moto_endpoint, published, tmp_path
) -> None:
    chain = MagicMock()
    chain.get_manifest_hash.return_value = published["root"]
    state, _outer, _meta = _fetch(
        cfg, moto_endpoint, published, tmp_path, chain=chain, owner_uid=OWNER_UID
    )
    chain.get_manifest_hash.assert_called_once_with(OWNER_UID)
    assert hash_named_tensors(state.items()) == published["root"]

    lying = MagicMock()
    lying.get_manifest_hash.return_value = "ff" * 32
    with pytest.raises(InitPublishError, match="on-chain"):
        _fetch(cfg, moto_endpoint, published, tmp_path, chain=lying, owner_uid=OWNER_UID)


def test_fetch_with_empty_bucket_raises(cfg, moto_endpoint, admin, tmp_path) -> None:
    empty = fresh_bucket(admin)

    async def go():
        async with StorageClient(
            empty, cfg.storage, endpoint_override=moto_endpoint, retry_base_delay_s=0.01
        ) as storage:
            return await fetch_and_verify_init(
                storage, None, "ab" * 32, local_dir=tmp_path, bucket=empty
            )

    with pytest.raises(InitPublishError, match="no init checkpoint"):
        asyncio.run(go())
