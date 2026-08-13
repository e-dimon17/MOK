"""Async R2/S3 object layer — every read hash-verified, every result deterministic.

Per-credential client cache, put/get with retries, multipart upload, ranged
download, HEAD timestamps, and concurrent gather — simple and strict:

  - every ``get`` verifies blake2b-256 against the caller's expected hash
    (committed on-chain) and raises :class:`IntegrityError` on mismatch;
  - object size is HEAD-checked *before* any body bytes are downloaded, and
    streams abort mid-flight if a server lies about ContentLength;
  - :meth:`StorageClient.gather_bytes` returns uid-sorted results regardless
    of completion order and records failures instead of silently dropping.

aioboto3/aiofiles/botocore are imported lazily so CPU-only unit runs that
never touch storage do not pay for them at module load.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
import random
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mok_core.config.schemas import BucketCreds, StorageConfig
from mok_core.determinism.hashing import hash_bytes, hash_file
from mok_core.telemetry import get_logger

log = get_logger("storage.client")

T = TypeVar("T")

_S3_MIN_PART_BYTES = 5 * 1024 * 1024  # S3/R2 hard minimum for all but the last part
_S3_MAX_PARTS = 9_000  # stay under the 10k part limit with headroom
_RETRYABLE_CODES = frozenset(
    {
        "InternalError",
        "ServiceUnavailable",
        "SlowDown",
        "Throttling",
        "ThrottlingException",
        "RequestTimeout",
        "RequestTimeoutException",
        "500",
        "503",
    }
)
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class StorageError(Exception):
    """Base class for storage-layer failures."""


class ObjectMissingError(StorageError):
    """The object does not exist in the peer's bucket."""


class ObjectTooLargeError(StorageError):
    """The object exceeds the caller's size bound (rejected before/while downloading)."""


class IntegrityError(StorageError):
    """Downloaded bytes do not hash to the expected blake2b-256 digest."""


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str(response.get("Error", {}).get("Code", ""))
    return ""


def _is_missing(exc: BaseException) -> bool:
    return _error_code(exc) in _MISSING_CODES


def _is_retryable(exc: BaseException) -> bool:
    """True only for transient transport/server faults; protocol errors never retry."""
    from botocore.exceptions import (  # noqa: PLC0415
        ClientError,
        ConnectionClosedError,
        ConnectTimeoutError,
        EndpointConnectionError,
        IncompleteReadError,
        ReadTimeoutError,
        ResponseStreamingError,
    )

    if isinstance(
        exc,
        (
            ConnectionClosedError,
            ConnectTimeoutError,
            EndpointConnectionError,
            ReadTimeoutError,
            IncompleteReadError,
            ResponseStreamingError,
        ),
    ):
        return True
    if isinstance(exc, ClientError):
        code = _error_code(exc)
        if code in _MISSING_CODES:
            return False
        meta = exc.response.get("ResponseMetadata", {})
        status = meta.get("HTTPStatusCode", 0)
        return code in _RETRYABLE_CODES or (isinstance(status, int) and status >= 500)
    return False


def _failure_reason(exc: BaseException) -> str:
    """Stable, prefix-classified reason string for GatherResult.failed."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, ObjectMissingError):
        return f"missing: {exc}"
    if isinstance(exc, IntegrityError):
        return f"integrity: {exc}"
    if isinstance(exc, ObjectTooLargeError):
        return f"too_large: {exc}"
    return f"error: {type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class GatherResult:
    """Outcome of a concurrent multi-peer fetch. ``ok`` is uid-ascending always."""

    ok: OrderedDict[int, bytes]
    failed: dict[int, str] = field(default_factory=dict)

    @property
    def uids(self) -> list[int]:
        return list(self.ok)


class StorageClient:
    """One node's view of the object layer: writes to its own bucket, verified
    reads from any peer's bucket. Async context manager; caches one aiobotocore
    client per (endpoint, access key).

    ``endpoint_override`` points every bucket at a single endpoint (moto/MinIO
    in tests); production leaves it None and uses each bucket's R2 endpoint.
    """

    def __init__(
        self,
        creds: BucketCreds,
        cfg: StorageConfig,
        *,
        session_factory: Callable[[], Any] | None = None,
        endpoint_override: str | None = None,
        region_name: str = "auto",
        retry_attempts: int = 3,
        retry_base_delay_s: float = 0.5,
        part_concurrency: int = 8,
    ) -> None:
        self._creds = creds
        self._cfg = cfg
        self._session_factory = session_factory
        self._endpoint_override = endpoint_override
        self._region_name = region_name
        self._retry_attempts = max(1, retry_attempts)
        self._retry_base_delay_s = retry_base_delay_s
        self._part_concurrency = max(1, part_concurrency)
        self._session: Any = None
        self._clients: dict[tuple[str, str], Any] = {}
        self._client_lock = asyncio.Lock()

    # ----------------------------------------------------------------- #
    # Lifecycle
    # ----------------------------------------------------------------- #

    async def __aenter__(self) -> StorageClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close every cached client; safe to call more than once."""
        for cache_key, client in list(self._clients.items()):
            try:
                await client.__aexit__(None, None, None)
            except Exception as e:  # pragma: no cover - shutdown best effort
                log.warning("s3 client close failed", endpoint=cache_key[0], error=str(e))
        self._clients.clear()

    # ----------------------------------------------------------------- #
    # Client cache
    # ----------------------------------------------------------------- #

    def _endpoint(self, bucket: BucketCreds) -> str:
        return self._endpoint_override or bucket.endpoint_url

    async def _client(self, bucket: BucketCreds) -> Any:
        cache_key = (self._endpoint(bucket), bucket.access_key_id)
        client = self._clients.get(cache_key)
        if client is not None:
            return client
        async with self._client_lock:
            client = self._clients.get(cache_key)
            if client is not None:
                return client
            if self._session is None:
                if self._session_factory is not None:
                    self._session = self._session_factory()
                else:
                    import aioboto3  # noqa: PLC0415

                    self._session = aioboto3.Session()
            from aiobotocore.config import AioConfig  # noqa: PLC0415

            config = AioConfig(
                max_pool_connections=32,
                retries={"max_attempts": 1},  # this layer owns retries
                s3={"addressing_style": "path"},
                signature_version="s3v4",
            )
            ctx = self._session.client(
                "s3",
                endpoint_url=cache_key[0],
                region_name=self._region_name,
                aws_access_key_id=bucket.access_key_id,
                aws_secret_access_key=bucket.secret_access_key,
                config=config,
            )
            client = await ctx.__aenter__()
            self._clients[cache_key] = client
            return client

    async def _purge_client(self, bucket: BucketCreds) -> None:
        """Drop a (possibly broken) cached client so the next call recreates it."""
        client = self._clients.pop((self._endpoint(bucket), bucket.access_key_id), None)
        if client is not None:
            with contextlib.suppress(Exception):
                await client.__aexit__(None, None, None)

    # ----------------------------------------------------------------- #
    # Retry wrapper
    # ----------------------------------------------------------------- #

    async def _retry(
        self,
        op: str,
        fn: Callable[[], Awaitable[T]],
        *,
        bucket: BucketCreds | None = None,
    ) -> T:
        """Run ``fn`` up to retry_attempts times, backing off only on retryable
        botocore faults. Protocol errors (404, integrity, size) propagate at once."""
        last: BaseException | None = None
        for attempt in range(self._retry_attempts):
            try:
                return await fn()
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                last = exc
                if bucket is not None:
                    await self._purge_client(bucket)
                if attempt + 1 < self._retry_attempts:
                    delay = self._retry_base_delay_s * (2**attempt)
                    delay += random.random() * 0.1 * self._retry_base_delay_s
                    log.warning(
                        "retryable storage fault",
                        op=op,
                        attempt=attempt + 1,
                        delay_s=round(delay, 3),
                        error=str(exc),
                    )
                    await asyncio.sleep(delay)
        assert last is not None
        raise last

    # ----------------------------------------------------------------- #
    # HEAD helpers
    # ----------------------------------------------------------------- #

    async def _head(self, bucket: BucketCreds, key: str) -> dict[str, Any]:
        async def _do() -> dict[str, Any]:
            s3 = await self._client(bucket)
            return await s3.head_object(Bucket=bucket.bucket_name, Key=key)

        try:
            return await self._retry(f"head {key}", _do, bucket=bucket)
        except Exception as exc:
            if _is_missing(exc):
                raise ObjectMissingError(f"{bucket.bucket_name}/{key}") from exc
            raise

    async def object_exists(self, bucket: BucketCreds, key: str) -> bool:
        try:
            await self._head(bucket, key)
            return True
        except ObjectMissingError:
            return False

    async def object_timestamp(self, bucket: BucketCreds, key: str) -> float:
        """Epoch seconds of the object's LastModified — the two-phase-commit gate clock.
        Raises ObjectMissingError if absent."""
        head = await self._head(bucket, key)
        last_modified = head.get("LastModified")
        if last_modified is None:  # pragma: no cover - S3 always returns it
            raise StorageError(f"HEAD {key}: no LastModified")
        return float(last_modified.timestamp())

    # ----------------------------------------------------------------- #
    # Bytes put/get
    # ----------------------------------------------------------------- #

    async def put_bytes(self, key: str, data: bytes) -> None:
        """Upload ``data`` to this node's own bucket."""

        async def _do() -> None:
            s3 = await self._client(self._creds)
            await s3.put_object(Bucket=self._creds.bucket_name, Key=key, Body=data)

        await self._retry(f"put {key}", _do, bucket=self._creds)

    async def get_bytes(
        self,
        bucket: BucketCreds,
        key: str,
        *,
        expected_hash: str | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        """Verified download into memory.

        Size is HEAD-checked before any body bytes move; the stream aborts if it
        exceeds the bound anyway (a lying server); the result must blake2b-hash
        to ``expected_hash`` when one is supplied.
        """
        limit = max_bytes if max_bytes is not None else self._cfg.max_payload_bytes
        size = await self._head_size(bucket, key)
        if size > limit:
            raise ObjectTooLargeError(f"{bucket.bucket_name}/{key}: {size} bytes > limit {limit}")

        async def _do() -> bytes:
            s3 = await self._client(bucket)
            resp = await s3.get_object(Bucket=bucket.bucket_name, Key=key)
            body = resp["Body"]
            buf = bytearray()
            try:
                while True:
                    chunk = await body.read(self._cfg.download_chunk_bytes)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    if len(buf) > limit:
                        raise ObjectTooLargeError(
                            f"{bucket.bucket_name}/{key}: stream exceeded limit {limit}"
                        )
            finally:
                body.close()
            return bytes(buf)

        try:
            data = await self._retry(f"get {key}", _do, bucket=bucket)
        except Exception as exc:
            if _is_missing(exc):
                raise ObjectMissingError(f"{bucket.bucket_name}/{key}") from exc
            raise
        if expected_hash is not None:
            actual = hash_bytes(data)
            if actual != expected_hash.lower():
                raise IntegrityError(
                    f"{bucket.bucket_name}/{key}: blake2b {actual} != expected {expected_hash.lower()}"
                )
        return data

    async def _head_size(self, bucket: BucketCreds, key: str) -> int:
        return int((await self._head(bucket, key))["ContentLength"])

    # ----------------------------------------------------------------- #
    # File upload (multipart above threshold)
    # ----------------------------------------------------------------- #

    async def upload_file(self, key: str, path: str | os.PathLike[str]) -> None:
        """Upload a local file to this node's own bucket; multipart above the
        configured threshold."""
        path = os.fspath(path)
        size = os.path.getsize(path)
        if size <= self._cfg.multipart_threshold_bytes:
            import aiofiles  # noqa: PLC0415

            async with aiofiles.open(path, "rb") as f:
                data = await f.read()
            await self.put_bytes(key, data)
            return
        await self._multipart_upload(key, path, size)

    async def _multipart_upload(self, key: str, path: str, size: int) -> None:
        import aiofiles  # noqa: PLC0415

        part_size = max(
            self._cfg.multipart_threshold_bytes,
            _S3_MIN_PART_BYTES,
            math.ceil(size / _S3_MAX_PARTS),
        )
        total_parts = max(1, math.ceil(size / part_size))
        bucket_name = self._creds.bucket_name

        async def _create() -> str:
            s3 = await self._client(self._creds)
            resp = await s3.create_multipart_upload(Bucket=bucket_name, Key=key)
            return resp["UploadId"]

        upload_id = await self._retry(f"create-mpu {key}", _create, bucket=self._creds)
        semaphore = asyncio.Semaphore(self._part_concurrency)

        async def _upload_part(part_number: int) -> dict[str, Any]:
            offset = (part_number - 1) * part_size
            length = min(part_size, size - offset)
            async with semaphore:
                async with aiofiles.open(path, "rb") as f:
                    await f.seek(offset)
                    data = await f.read(length)
                if len(data) != length:
                    raise StorageError(f"{path}: short read at offset {offset} (file changed?)")

                async def _do() -> dict[str, Any]:
                    s3 = await self._client(self._creds)
                    resp = await s3.upload_part(
                        Bucket=bucket_name,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=data,
                    )
                    return {"PartNumber": part_number, "ETag": resp["ETag"]}

                return await self._retry(f"part {part_number}/{total_parts} {key}", _do, bucket=self._creds)

        try:
            done = await asyncio.gather(*(_upload_part(pn) for pn in range(1, total_parts + 1)))
            parts = sorted(done, key=lambda p: p["PartNumber"])

            async def _complete() -> None:
                s3 = await self._client(self._creds)
                await s3.complete_multipart_upload(
                    Bucket=bucket_name,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={"Parts": parts},
                )

            await self._retry(f"complete-mpu {key}", _complete, bucket=self._creds)
        except Exception:
            try:
                s3 = await self._client(self._creds)
                await s3.abort_multipart_upload(Bucket=bucket_name, Key=key, UploadId=upload_id)
            except Exception as abort_exc:
                log.warning("mpu abort failed", key=key, error=str(abort_exc))
            raise

    # ----------------------------------------------------------------- #
    # File download (ranged, resumable, atomic)
    # ----------------------------------------------------------------- #

    async def download_file(
        self,
        bucket: BucketCreds,
        key: str,
        path: str | os.PathLike[str],
        *,
        expected_hash: str | None = None,
        max_bytes: int | None = None,
    ) -> None:
        """Ranged download to ``path``.part (resuming any existing prefix), then
        verify and atomically rename to ``path``. The final file appears only
        complete and — when ``expected_hash`` is given — verified."""
        import aiofiles  # noqa: PLC0415

        path = os.fspath(path)
        size = await self._head_size(bucket, key)
        if max_bytes is not None and size > max_bytes:
            raise ObjectTooLargeError(f"{bucket.bucket_name}/{key}: {size} bytes > limit {max_bytes}")

        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        part_path = path + ".part"
        offset = 0
        if os.path.exists(part_path):
            offset = os.path.getsize(part_path)
            if offset > size:
                offset = 0  # stale partial larger than the object: restart

        chunk_bytes = self._cfg.download_chunk_bytes
        async with aiofiles.open(part_path, "r+b" if offset else "wb") as f:
            await f.seek(offset)
            while offset < size:
                end = min(offset + chunk_bytes, size) - 1
                byte_range = f"bytes={offset}-{end}"

                async def _do(byte_range: str = byte_range) -> bytes:
                    s3 = await self._client(bucket)
                    resp = await s3.get_object(Bucket=bucket.bucket_name, Key=key, Range=byte_range)
                    body = resp["Body"]
                    try:
                        return await body.read()
                    finally:
                        body.close()

                data = await self._retry(f"range {byte_range} {key}", _do, bucket=bucket)
                if len(data) != end - offset + 1:
                    raise StorageError(f"{key}: short range read {byte_range} -> {len(data)} bytes")
                await f.write(data)
                offset += len(data)

        actual_size = os.path.getsize(part_path)
        if actual_size != size:
            raise StorageError(f"{key}: downloaded {actual_size} bytes, HEAD said {size}")
        if expected_hash is not None:
            actual = await asyncio.to_thread(hash_file, part_path)
            if actual != expected_hash.lower():
                os.remove(part_path)  # poisoned partial must not be resumed
                raise IntegrityError(
                    f"{bucket.bucket_name}/{key}: blake2b {actual} != expected {expected_hash.lower()}"
                )
        os.replace(part_path, path)

    # ----------------------------------------------------------------- #
    # Listing
    # ----------------------------------------------------------------- #

    async def list_keys(self, bucket: BucketCreds, prefix: str) -> list[str]:
        """All keys under ``prefix``, sorted (deterministic across pagination)."""

        async def _do() -> list[str]:
            s3 = await self._client(bucket)
            keys: list[str] = []
            token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": bucket.bucket_name, "Prefix": prefix}
                if token:
                    kwargs["ContinuationToken"] = token
                resp = await s3.list_objects_v2(**kwargs)
                keys.extend(obj["Key"] for obj in resp.get("Contents", []))
                if not resp.get("IsTruncated"):
                    return sorted(keys)
                token = resp.get("NextContinuationToken")

        return await self._retry(f"list {prefix}", _do, bucket=bucket)

    # ----------------------------------------------------------------- #
    # Gather
    # ----------------------------------------------------------------- #

    async def gather_bytes(
        self,
        peers: Mapping[int, BucketCreds],
        key_fn: Callable[[int], str],
        *,
        expected_hashes: Mapping[int, str],
        deadline_s: float,
        max_bytes: int | None = None,
    ) -> GatherResult:
        """Fetch one object from every peer concurrently, each fetch bounded by
        ``deadline_s`` and verified against ``expected_hashes[uid]`` when present.

        The result is uid-ascending regardless of completion order — the caller
        (outer merge) iterates it directly, so this ordering is consensus-bearing.
        Failures are recorded per-uid, never raised and never silently dropped.
        """

        async def _fetch(uid: int) -> bytes:
            return await asyncio.wait_for(
                self.get_bytes(
                    peers[uid],
                    key_fn(uid),
                    expected_hash=expected_hashes.get(uid),
                    max_bytes=max_bytes,
                ),
                timeout=deadline_s,
            )

        uids = sorted(peers)
        results = await asyncio.gather(*(_fetch(uid) for uid in uids), return_exceptions=True)
        ok: OrderedDict[int, bytes] = OrderedDict()
        failed: dict[int, str] = {}
        for uid, result in zip(uids, results, strict=True):
            if isinstance(result, BaseException):
                failed[uid] = _failure_reason(result)
                log.warning("gather fetch failed", uid=uid, reason=failed[uid])
            else:
                ok[uid] = result
        return GatherResult(ok=ok, failed=failed)
