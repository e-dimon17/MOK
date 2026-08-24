"""Resumable dataset upload against a local moto S3 server (dataprep/pipeline/upload.py)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from dataprep.pipeline.build_manifest import build_dataset_manifest
from dataprep.pipeline.shard_writer import save_shard_metas, write_shards
from dataprep.pipeline.upload import dataset_files, resolve_endpoint, upload_dataset


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


def fresh_bucket(admin: Any) -> str:
    name = f"mok-stepa-{uuid.uuid4().hex[:8]}"
    admin.create_bucket(Bucket=name)
    return name


def make_dataset_dir(tmp_path: Path) -> Path:
    seqs = [np.arange(i * 16, (i + 1) * 16, dtype=np.uint16) for i in range(5)]
    metas = write_shards(seqs, tmp_path, shard_sequences=2, seq_len=16)
    save_shard_metas(metas, tmp_path / "shards.json")
    build_dataset_manifest(
        metas,
        name="uptest",
        seq_len=16,
        tokenizer_hash="11" * 32,
        out_dir=tmp_path,
        shard_sequences=2,
    )
    (tmp_path / "tokenizer.json").write_text('{"model": {"type": "BPE"}}')
    return tmp_path


def creds() -> dict[str, str]:
    return {"access_key_id": "test-key", "secret_access_key": "test-secret"}


async def test_upload_publishes_everything(moto_endpoint, admin, tmp_path):
    data_dir = make_dataset_dir(tmp_path)
    bucket = fresh_bucket(admin)
    report = await upload_dataset(
        data_dir, prefix="datasets/uptest", bucket=bucket, endpoint_url=moto_endpoint, **creds()
    )
    expected_keys = {f"datasets/uptest/{p.name}" for p in dataset_files(data_dir)}
    assert set(report.uploaded) == expected_keys
    assert report.skipped == ()
    assert report.bytes_sent == sum(p.stat().st_size for p in dataset_files(data_dir))
    listed = admin.list_objects_v2(Bucket=bucket, Prefix="datasets/uptest/")
    assert {o["Key"] for o in listed["Contents"]} == expected_keys
    # 3 shards + shard_index + manifest + shards.json + tokenizer.json
    assert len(expected_keys) == 7


async def test_second_upload_skips_matching_objects(moto_endpoint, admin, tmp_path):
    data_dir = make_dataset_dir(tmp_path)
    bucket = fresh_bucket(admin)
    kw = {"prefix": "d", "bucket": bucket, "endpoint_url": moto_endpoint, **creds()}
    first = await upload_dataset(data_dir, **kw)
    second = await upload_dataset(data_dir, **kw)
    assert second.uploaded == ()
    assert set(second.skipped) == set(first.uploaded)
    assert second.bytes_sent == 0


async def test_changed_bytes_same_size_reuploaded(moto_endpoint, admin, tmp_path):
    data_dir = make_dataset_dir(tmp_path)
    bucket = fresh_bucket(admin)
    kw = {"prefix": "d", "bucket": bucket, "endpoint_url": moto_endpoint, **creds()}
    await upload_dataset(data_dir, **kw)
    shard = sorted(data_dir.glob("shard-*.bin"))[0]
    data = bytearray(shard.read_bytes())
    data[0] ^= 0xFF  # same size, different content -> ETag mismatch
    shard.write_bytes(bytes(data))
    report = await upload_dataset(data_dir, **kw)
    assert report.uploaded == (f"d/{shard.name}",)
    assert len(report.skipped) == 6


async def test_env_credentials_and_bucket(moto_endpoint, admin, tmp_path, monkeypatch):
    data_dir = make_dataset_dir(tmp_path)
    bucket = fresh_bucket(admin)
    monkeypatch.setenv("R2_BUCKET_NAME", bucket)
    monkeypatch.setenv("R2_WRITE_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("R2_WRITE_SECRET_ACCESS_KEY", "env-secret")
    report = await upload_dataset(data_dir, prefix="env", endpoint_url=moto_endpoint)
    assert len(report.uploaded) == 7


async def test_missing_credentials_error(tmp_path, monkeypatch):
    data_dir = make_dataset_dir(tmp_path)
    monkeypatch.delenv("R2_BUCKET_NAME", raising=False)
    with pytest.raises(ValueError, match="R2_BUCKET_NAME"):
        await upload_dataset(data_dir, prefix="x", endpoint_url="http://127.0.0.1:1")


def test_resolve_endpoint(monkeypatch):
    assert resolve_endpoint("http://x:9") == "http://x:9"
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct123")
    assert resolve_endpoint() == "https://acct123.r2.cloudflarestorage.com"
    monkeypatch.delenv("R2_ACCOUNT_ID")
    with pytest.raises(ValueError, match="R2_ACCOUNT_ID"):
        resolve_endpoint()
