"""Resumable upload of a prepared dataset directory to R2/S3 — step A, method 4.

Uploads every shard plus `shard_index.json`, `manifest.json`, `shards.json`
and `tokenizer.json` via aioboto3. Endpoint/credentials default to the R2_*
environment (`.env.example`); tests inject a moto endpoint. Resumable: an
object whose remote size matches and whose ETag equals the local MD5 (or is
a multipart ETag of matching size) is skipped.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from mok_core.telemetry import get_logger

from .build_manifest import MANIFEST_FILENAME, SHARD_INDEX_FILENAME, TOKENIZER_FILENAME
from .shard_writer import SHARD_METAS_FILENAME

log = get_logger("stepA.upload")

_SIDECARS = (SHARD_INDEX_FILENAME, MANIFEST_FILENAME, SHARD_METAS_FILENAME, TOKENIZER_FILENAME)

_PROGRESS_EVERY = 25


@dataclass(frozen=True)
class UploadReport:
    uploaded: tuple[str, ...]
    skipped: tuple[str, ...]
    bytes_sent: int


def _require(value: str | None, env: str, what: str) -> str:
    got = value if value is not None else os.environ.get(env, "")
    if not got:
        raise ValueError(f"{what} not provided and ${env} is unset")
    return got


def resolve_endpoint(endpoint_url: str | None = None) -> str:
    """Explicit endpoint, or the R2 endpoint derived from $R2_ACCOUNT_ID."""
    if endpoint_url:
        return endpoint_url
    account = _require(None, "R2_ACCOUNT_ID", "endpoint_url")
    return f"https://{account}.r2.cloudflarestorage.com"


def dataset_files(data_dir: str | PathLike[str]) -> list[Path]:
    """Everything worth publishing: shards (sorted by name) then sidecars."""
    d = Path(data_dir)
    shards = sorted(p for p in d.iterdir() if p.name.startswith("shard-") and p.suffix == ".bin")
    sidecars = [d / n for n in _SIDECARS if (d / n).exists()]
    return shards + sidecars


async def upload_dataset(
    data_dir: str | PathLike[str],
    *,
    prefix: str,
    bucket: str | None = None,
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    region: str = "auto",
    concurrency: int = 4,
) -> UploadReport:
    """Upload the dataset directory to `bucket` under `prefix/` (resumable)."""
    import aioboto3
    from botocore.exceptions import ClientError

    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    bucket_name = _require(bucket, "R2_BUCKET_NAME", "bucket")
    endpoint = resolve_endpoint(endpoint_url)
    key_id = _require(access_key_id, "R2_WRITE_ACCESS_KEY_ID", "access_key_id")
    secret = _require(secret_access_key, "R2_WRITE_SECRET_ACCESS_KEY", "secret_access_key")

    files = dataset_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no dataset files found in {data_dir}")

    uploaded: list[str] = []
    skipped: list[str] = []
    bytes_sent = 0
    done = 0
    total_bytes = sum(p.stat().st_size for p in files)
    bytes_done = 0  # sent + skipped — drives the ETA
    t0 = time.monotonic()
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    log.info(
        "upload starting",
        files=len(files),
        total_gb=round(total_bytes / 1024**3, 1),
        bucket=bucket_name,
        prefix=prefix,
        concurrency=concurrency,
    )

    session = aioboto3.Session(aws_access_key_id=key_id, aws_secret_access_key=secret)
    async with session.client("s3", endpoint_url=endpoint, region_name=region) as s3:

        async def put(path: Path) -> None:
            nonlocal bytes_sent, done, bytes_done
            key = f"{prefix.rstrip('/')}/{path.name}"
            # Read AND hash inside the semaphore: resident bytes stay bounded to
            # concurrency * shard size (an eager read of all 7k+ shards OOMs).
            async with sem:
                data = await asyncio.to_thread(path.read_bytes)
                md5 = await asyncio.to_thread(
                    lambda: hashlib.md5(data).hexdigest()  # noqa: S324 — ETag comparison, not security
                )
                try:
                    head = await s3.head_object(Bucket=bucket_name, Key=key)
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey", "NotFound"):
                        raise
                    head = None
                was_skipped = False
                if head is not None and head["ContentLength"] == len(data):
                    etag = head.get("ETag", "").strip('"')
                    was_skipped = etag == md5 or "-" in etag  # multipart ETag: trust the size match
                if not was_skipped:
                    await s3.put_object(Bucket=bucket_name, Key=key, Body=data)
            async with lock:
                done += 1
                bytes_done += len(data)
                if was_skipped:
                    skipped.append(key)
                else:
                    uploaded.append(key)
                    bytes_sent += len(data)
                if done % _PROGRESS_EVERY == 0 or done == len(files):
                    elapsed = max(time.monotonic() - t0, 1e-9)
                    rate = bytes_done / elapsed
                    log.info(
                        "upload progress",
                        files_done=done,
                        files_total=len(files),
                        uploaded=len(uploaded),
                        skipped=len(skipped),
                        gb_done=round(bytes_done / 1024**3, 1),
                        mb_per_s=round(rate / 1024**2, 1),
                        eta_s=round((total_bytes - bytes_done) / rate),
                    )

        await asyncio.gather(*(put(p) for p in files))

    return UploadReport(
        uploaded=tuple(sorted(uploaded)), skipped=tuple(sorted(skipped)), bytes_sent=bytes_sent
    )


def upload_dataset_sync(data_dir: str | PathLike[str], **kwargs) -> UploadReport:
    """Blocking wrapper for the CLI."""
    return asyncio.run(upload_dataset(data_dir, **kwargs))
