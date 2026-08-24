"""Local re-verification of a prepared dataset directory — the dataset release gate.

`verify_local` re-hashes shards against `shard_index.json`, rechecks the
Merkle root against `manifest.json`, validates every shard's size arithmetic
(full shards exact, last shard partial-allowed, token totals consistent) and
the tokenizer hash. `verify_sample` hashes only a deterministic sample of
shards — the cheap spot-check mode.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from mok_core.data.shards import shard_filename, shard_leaf_hash, verify_index_matches_ref
from mok_core.telemetry import get_logger

from .build_manifest import (
    MANIFEST_FILENAME,
    SHARD_INDEX_FILENAME,
    TOKENIZER_FILENAME,
    load_manifest_ref,
    load_shard_index,
)
from .tokenizer_train import tokenizer_file_hash

log = get_logger("dataprep.verify")

_PROGRESS_EVERY = 100


@dataclass(frozen=True)
class VerifyReport:
    dataset: str
    merkle_root: str
    num_shards: int
    shards_hashed: int
    tokens_total: int
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def _hash_shard(args: tuple[int, str]) -> tuple[int, str]:
    """Subprocess entry: (shard index, path) -> (shard index, leaf hash hex)."""
    i, path = args
    return i, shard_leaf_hash(path)


def verify_local(
    data_dir: str | PathLike[str],
    *,
    sample: int | None = None,
    seed: int = 0,
    workers: int | None = None,
) -> VerifyReport:
    """Verify a dataset directory against its own index + manifest.

    `sample=n` re-hashes only n deterministically chosen shards (size and
    structure checks still cover every shard); `sample=None` hashes all.
    Hashing runs on `workers` processes (default: cpu//2, capped 16 — the
    full-corpus pass is disk-bound) with periodic progress + ETA logs.
    """
    d = Path(data_dir)
    index = load_shard_index(d / SHARD_INDEX_FILENAME)
    ref = load_manifest_ref(d / MANIFEST_FILENAME)
    failures: list[str] = []

    try:
        verify_index_matches_ref(index, ref)
    except ValueError as e:
        failures.append(f"index/manifest mismatch: {e}")

    seq_bytes = ref.seq_len * 2
    total_sequences = 0
    missing: set[int] = set()
    for i in range(index.num_shards):
        path = d / shard_filename(index.leaf(i))
        if not path.exists():
            failures.append(f"shard {i}: missing file {path.name}")
            missing.add(i)
            continue
        size = path.stat().st_size
        if size == 0 or size % seq_bytes != 0:
            failures.append(f"shard {i}: size {size} is not a positive multiple of {seq_bytes}")
            continue
        if i < index.num_shards - 1 and size != ref.shard_bytes:
            failures.append(f"shard {i}: size {size} != full shard_bytes {ref.shard_bytes}")
        if i == index.num_shards - 1 and size > ref.shard_bytes:
            failures.append(f"shard {i}: final shard size {size} exceeds shard_bytes {ref.shard_bytes}")
        total_sequences += size // seq_bytes

    if ref.tokens_total % ref.seq_len != 0:
        failures.append(f"tokens_total {ref.tokens_total} is not a multiple of seq_len {ref.seq_len}")
    elif not missing and total_sequences != ref.tokens_total // ref.seq_len:
        failures.append(
            f"sequence count {total_sequences} != tokens_total/seq_len "
            f"{ref.tokens_total // ref.seq_len}"
        )

    if sample is None or sample >= index.num_shards:
        chosen = list(range(index.num_shards))
    elif sample <= 0:
        chosen = []
    else:
        chosen = sorted(random.Random(seed).sample(range(index.num_shards), sample))
    import multiprocessing  # noqa: PLC0415 — heavy path only; keeps module import light
    import os  # noqa: PLC0415

    tasks = [(i, str(d / shard_filename(index.leaf(i)))) for i in chosen if i not in missing]
    workers = workers or max(2, min(16, (os.cpu_count() or 8) // 2))
    hashed = 0
    hash_failures: list[tuple[int, str]] = []  # (shard idx, message) — sorted for determinism
    log.info("hashing shards", shards=len(tasks), workers=workers)
    t0 = time.monotonic()
    if tasks:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(workers, len(tasks))) as pool:
            for i, digest in pool.imap_unordered(_hash_shard, tasks, chunksize=4):
                hashed += 1
                if digest != index.leaf(i):
                    hash_failures.append(
                        (i, f"shard {i}: content hash does not match index leaf {index.shard_hashes[i]}")
                    )
                if hashed % _PROGRESS_EVERY == 0 or hashed == len(tasks):
                    elapsed = max(time.monotonic() - t0, 1e-9)
                    rate = hashed / elapsed
                    log.info(
                        "verify progress",
                        shards_done=hashed,
                        shards_total=len(tasks),
                        shards_per_s=round(rate, 2),
                        eta_s=round((len(tasks) - hashed) / rate),
                    )
    failures.extend(msg for _, msg in sorted(hash_failures))

    tok = d / TOKENIZER_FILENAME
    if tok.exists() and tokenizer_file_hash(tok) != ref.tokenizer_hash:
        failures.append(f"tokenizer.json hash does not match manifest tokenizer_hash {ref.tokenizer_hash}")

    return VerifyReport(
        dataset=ref.name,
        merkle_root=ref.merkle_root,
        num_shards=ref.num_shards,
        shards_hashed=hashed,
        tokens_total=ref.tokens_total,
        failures=tuple(failures),
    )


def verify_sample(data_dir: str | PathLike[str], n: int, *, seed: int = 0) -> VerifyReport:
    """Spot-check mode: hash only `n` deterministically sampled shards."""
    return verify_local(data_dir, sample=n, seed=seed)
