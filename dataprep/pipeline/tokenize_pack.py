"""Deterministic tokenization and fixed-length sequence packing.

Documents are EOS-joined into one flat token stream (deterministic source
order) and chunked into exact `seq_len` uint16 sequences; the final partial
chunk is dropped. Everything here is a pure function of its inputs; the HF
`tokenizers` import happens only inside `encode_documents`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from os import PathLike
from pathlib import Path

import numpy as np

MAX_TOKEN_ID = 65535  # uint16 storage bound == vocab 65,536
_TOKEN_DTYPE = np.dtype("<u2")


def _validated_u16(buf: list[int]) -> np.ndarray:
    arr = np.asarray(buf, dtype=np.int64)
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) > MAX_TOKEN_ID):
        raise ValueError(f"token id out of uint16 range [0, {MAX_TOKEN_ID}]")
    return arr.astype(np.uint16)


def chunk_token_stream(tokens: Iterable[int], seq_len: int) -> Iterator[np.ndarray]:
    """Split a flat token stream into exact-length uint16 sequences.

    The final partial chunk (< seq_len tokens) is dropped. Token ids are
    bounds-checked into uint16.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    buf: list[int] = []
    for t in tokens:
        buf.append(t)
        if len(buf) == seq_len:
            yield _validated_u16(buf)
            buf.clear()


def pack_documents(
    token_iters: Iterable[Iterable[int]],
    seq_len: int = 4096,
    eos_id: int = 2,
) -> Iterator[np.ndarray]:
    """EOS-join per-document token iterables and chunk into `seq_len` sequences.

    Every document contributes its tokens followed by one `eos_id`; long
    documents simply span sequence boundaries; the final partial sequence is
    dropped. Pure and deterministic given the same documents in the same order.
    """
    if not 0 <= eos_id <= MAX_TOKEN_ID:
        raise ValueError(f"eos_id must be in [0, {MAX_TOKEN_ID}], got {eos_id}")

    def joined() -> Iterator[int]:
        for doc in token_iters:
            yield from doc
            yield eos_id

    return chunk_token_stream(joined(), seq_len)


def chunk_token_arrays(arrays: Iterable[np.ndarray], seq_len: int) -> Iterator[np.ndarray]:
    """Array-block variant of `chunk_token_stream` for pre-tokenized streams."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    pending = np.empty(0, dtype=np.uint16)
    for block in arrays:
        arr = np.asarray(block)
        if arr.dtype != np.uint16:
            raise ValueError(f"token arrays must be uint16, got {arr.dtype}")
        pending = arr.astype(np.uint16, copy=False) if pending.size == 0 else np.concatenate([pending, arr])
        n_full = pending.size // seq_len
        for j in range(n_full):
            yield pending[j * seq_len : (j + 1) * seq_len].copy()
        pending = pending[n_full * seq_len :].copy()


def encode_documents(
    texts: Iterable[str],
    tokenizer_path: str | PathLike[str],
    *,
    batch_size: int = 256,
) -> Iterator[list[int]]:
    """Tokenize documents (no special tokens added) in input order (lazy import)."""
    from tokenizers import Tokenizer

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    tok = Tokenizer.from_file(str(tokenizer_path))
    it = iter(texts)
    while batch := list(islice(it, batch_size)):
        for enc in tok.encode_batch(batch, add_special_tokens=False):
            yield enc.ids


def write_token_stream(
    token_iters: Iterable[Iterable[int]],
    out_path: str | PathLike[str],
    *,
    eos_id: int = 2,
    flush_tokens: int = 1 << 22,
) -> int:
    """Write the EOS-joined flat token stream as raw little-endian uint16 bytes.

    Returns the total token count (documents + one EOS each). This is the
    intermediate format between the `tokenize` and `shard` CLI stages.
    """
    if not 0 <= eos_id <= MAX_TOKEN_ID:
        raise ValueError(f"eos_id must be in [0, {MAX_TOKEN_ID}], got {eos_id}")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    buf: list[int] = []
    with open(out, "wb") as f:
        for doc in token_iters:
            buf.extend(doc)
            buf.append(eos_id)
            if len(buf) >= flush_tokens:
                f.write(_validated_u16(buf).astype(_TOKEN_DTYPE, copy=False).tobytes())
                total += len(buf)
                buf.clear()
        if buf:
            f.write(_validated_u16(buf).astype(_TOKEN_DTYPE, copy=False).tobytes())
            total += len(buf)
    return total


def iter_token_file_arrays(
    paths: Iterable[str | PathLike[str]],
    *,
    block_tokens: int = 1 << 22,
) -> Iterator[np.ndarray]:
    """Stream flat `<u2` token files (in the given order) as uint16 blocks."""
    if block_tokens <= 0:
        raise ValueError(f"block_tokens must be positive, got {block_tokens}")
    for path in paths:
        size = Path(path).stat().st_size
        if size % _TOKEN_DTYPE.itemsize != 0:
            raise ValueError(f"{Path(path).name}: {size} bytes is not a multiple of 2 (uint16 stream)")
        with open(path, "rb") as f:
            while (block := np.fromfile(f, dtype=_TOKEN_DTYPE, count=block_tokens)).size:
                yield block.astype(np.uint16, copy=False)


# --------------------------------------------------------------------------- #
# Parallel per-unit tokenization (same output bytes as the sequential path)
# --------------------------------------------------------------------------- #


def _tokenize_unit(args: tuple[str, str, tuple[str, ...], str, str, int]) -> tuple[str, int]:
    """Subprocess entry: encode one spool unit into its own token file.

    args = (source, unit_dir, parts, tokenizer_path, out_path, rayon_threads).
    Per-doc framing (doc tokens + EOS) is independent, so concatenating unit
    files in unit order is byte-identical to the sequential single stream.
    """
    import os as _os
    import time as _time

    source, unit_dir, parts, tokenizer_path, out_path, rayon_threads = args
    _os.environ.setdefault("RAYON_NUM_THREADS", str(max(1, rayon_threads)))
    from pathlib import Path as _Path

    from mok_core.telemetry import bind, get_logger, setup_logging

    from .download import _iter_parts
    from .tokenizer_train import EOS_ID

    setup_logging(_os.environ.get("MOK_LOG_LEVEL", "INFO"))
    log = get_logger("dataprep.tokenize")
    bind(source=source, unit=_Path(out_path).stem)

    t0 = _time.monotonic()
    count = 0

    def _docs():
        nonlocal count
        for text in _iter_parts(_Path(unit_dir), parts):
            count += 1
            if count % 500_000 == 0:
                elapsed = max(_time.monotonic() - t0, 1e-9)
                log.info("tokenize progress", docs=count, docs_per_s=round(count / elapsed, 1))
            yield text

    tokens = encode_documents(_docs(), tokenizer_path)
    total = write_token_stream(tokens, out_path, eos_id=EOS_ID)
    log.info("unit tokenized", docs=count, tokens=total, secs=round(_time.monotonic() - t0, 1))
    return source, total


def tokenize_parallel(
    cfg,
    spool_dir,
    tokenizer_path,
    out_dir,
    *,
    workers: int | None = None,
    parts_per_task: int = 32,
) -> dict[str, int]:
    """Tokenize every source concurrently in balanced part-group chunks.

    Each task encodes `parts_per_task` consecutive spool parts of one unit
    (~3.2M docs at the standard 100k-doc parts), so hundreds of equal-sized
    tasks keep every core busy to the end — no slow-unit tail. Groups never
    cross unit boundaries; chunk files are numbered globally per source in
    unit-then-part order, so concatenating {source}.NNN.tokens.u16 in sorted
    order is byte-identical to the sequential single stream (tested).
    """
    import multiprocessing
    import os
    from pathlib import Path

    from .dedup_parallel import plan_units

    if parts_per_task <= 0:
        raise ValueError(f"parts_per_task must be positive, got {parts_per_task}")
    workers = workers or max(4, min(32, (os.cpu_count() or 8) // 4))
    units = plan_units(cfg, spool_dir)
    # plan_units follows dedup priority order; regroup into cfg.sources order
    # (the tokenize/pack order) while keeping units of a source in spool order.
    per_source: dict[str, list] = {s.name: [] for s in cfg.sources}
    for u in units:
        per_source[u.source].append(u)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rayon = max(1, (os.cpu_count() or 8) // workers)
    tasks = []
    for spec in cfg.sources:
        chunk_idx = 0
        for u in per_source[spec.name]:
            for start in range(0, len(u.parts), parts_per_task):
                group = u.parts[start : start + parts_per_task]
                tasks.append(
                    (
                        u.source,
                        u.unit_dir,
                        group,
                        str(tokenizer_path),
                        str(out / f"{u.source}.{chunk_idx:03d}.tokens.u16"),
                        rayon,
                    )
                )
                chunk_idx += 1
        if chunk_idx > 1000:
            raise ValueError(
                f"source {spec.name!r} would produce {chunk_idx} chunk files; raise "
                "parts_per_task (the NNN naming supports at most 1000 per source)"
            )
    totals: dict[str, int] = {s.name: 0 for s in cfg.sources}
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(tasks))) as pool:
        for source, total in pool.imap_unordered(_tokenize_unit, tasks):
            totals[source] += total
    return totals


def discover_token_files(tokens_dir, source_names) -> list:
    """Token-file paths in pack order; supports both the single-file layout
    ({source}.tokens.u16) and the parallel unit layout ({source}.NNN.tokens.u16)."""
    from pathlib import Path

    tdir = Path(tokens_dir)
    paths: list = []
    for name in source_names:
        single = tdir / f"{name}.tokens.u16"
        if single.exists():
            paths.append(single)
            continue
        units = sorted(tdir.glob(f"{name}.[0-9][0-9][0-9].tokens.u16"))
        if not units:
            raise FileNotFoundError(
                f"no token files for source {name!r} in {tdir} "
                f"(expected {name}.tokens.u16 or {name}.NNN.tokens.u16)"
            )
        paths.extend(units)
    return paths
