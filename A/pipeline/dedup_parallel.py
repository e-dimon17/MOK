"""Two-pass parallel exact-hash dedup — byte-identical output to dedup.py.

The sequential reference (`dedup.dedup_documents`) defines the semantics: docs
are examined in source-priority order (legacy parts, then worker sub-spools
w00, w01, ... — exactly `iter_source_documents` order); the first occurrence
of each normalized-text digest is kept; whitespace-only docs are dropped.

This module computes the SAME result using every core:

  pass 1   hash every spool part concurrently (per-part tasks, process pool);
           each worker reads its part file directly — no text crosses IPC.
  resolve  concatenate digests in the global examine order and take
           np.unique(..., return_index=True) first occurrences — a vectorized
           transcription of "digest in seen" over the identical order.
  pass 2   per input sub-spool, stream docs, apply the keep mask, and write
           the output sub-spool — all units concurrently. Outputs use the
           worker layout (out/<source>/workers/wNN or legacy root), which
           `iter_source_documents` reads back in the same global order, so
           downstream stages see exactly the sequential result.

Equivalence is enforced by test (tests/unit/test_stepA_dedup_parallel.py):
sequential and parallel runs must agree byte-for-byte per source and on stats.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

from mok_core.telemetry import get_logger

from .dedup import DedupStats, doc_digest
from .download import (
    WORKERS_DIRNAME,
    CorpusConfig,
    _iter_parts,  # noqa: PLC2701 — same-package reuse of the part reader
    load_spool_state,
    spool_documents,
    worker_names,
)

log = get_logger("stepA.dedup")

_EMPTY = np.uint8(1)


@dataclass(frozen=True)
class SpoolUnit:
    """One contiguously-ordered slice of a source: its legacy stream or one worker."""

    source: str
    unit_dir: str                 # directory holding parts + state.json
    worker: str | None            # None = legacy stream, else "wNN"
    parts: tuple[str, ...]
    docs: int
    offset: int                   # global examine-order offset of this unit's first doc


def plan_units(cfg: CorpusConfig, spool_root: str | PathLike[str]) -> list[SpoolUnit]:
    """Units in the exact global examine order of the sequential implementation."""
    units: list[SpoolUnit] = []
    offset = 0
    for spec in cfg.dedup_sequence():
        sdir = Path(spool_root) / spec.name
        legacy = load_spool_state(spool_root, spec.name)
        if legacy.parts:
            units.append(
                SpoolUnit(spec.name, str(sdir), None, legacy.parts, legacy.docs, offset)
            )
            offset += legacy.docs
        for wname in worker_names(spool_root, spec.name):
            wroot = sdir / WORKERS_DIRNAME
            wstate = load_spool_state(wroot, wname)
            units.append(
                SpoolUnit(spec.name, str(wroot / wname), wname, wstate.parts, wstate.docs, offset)
            )
            offset += wstate.docs
    return units


# --------------------------------------------------------------------------- #
# Pass 1 — parallel hashing, one task per part file
# --------------------------------------------------------------------------- #


def _hash_part(part_path: str) -> tuple[bytes, bytes]:
    """(uint64 digests, uint8 empty-flags) for one part file, in document order."""
    digests: list[int] = []
    empties: list[int] = []
    for text in _iter_parts(Path(part_path).parent, [Path(part_path).name]):
        d = doc_digest(text)
        if d is None:
            digests.append(0)
            empties.append(1)
        else:
            digests.append(d)
            empties.append(0)
    return (
        np.asarray(digests, dtype=np.uint64).tobytes(),
        np.asarray(empties, dtype=np.uint8).tobytes(),
    )


def hash_all_parts(
    units: list[SpoolUnit], *, workers: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Digests + empty flags for the whole corpus in global order, plus per-unit
    document counts (validated against spool states)."""
    import multiprocessing  # noqa: PLC0415

    tasks = [str(Path(u.unit_dir) / p) for u in units for p in u.parts]
    t0 = time.monotonic()
    results: list[tuple[bytes, bytes]] = []
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap(_hash_part, tasks, chunksize=1)):
            results.append(res)
            if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                elapsed = max(time.monotonic() - t0, 1e-9)
                log.info(
                    "pass1 hashing progress",
                    parts_done=i + 1,
                    parts_total=len(tasks),
                    parts_per_s=round((i + 1) / elapsed, 2),
                    eta_s=round((len(tasks) - i - 1) / ((i + 1) / elapsed)),
                )
    digests = np.frombuffer(b"".join(r[0] for r in results), dtype=np.uint64)
    empties = np.frombuffer(b"".join(r[1] for r in results), dtype=np.uint8)
    unit_counts: list[int] = []
    cursor = 0
    part_lens = [len(r[1]) for r in results]
    pi = 0
    for u in units:
        n = 0
        for _ in u.parts:
            n += part_lens[pi]
            pi += 1
        unit_counts.append(n)
        cursor += n
        if n != u.docs:
            raise RuntimeError(
                f"unit {u.source}/{u.worker or 'legacy'}: state.json says {u.docs} docs, "
                f"parts contain {n} — spool is inconsistent, refusing to dedup"
            )
    if cursor != len(digests):
        raise RuntimeError("global digest count mismatch")  # pragma: no cover
    return digests, empties, unit_counts


# --------------------------------------------------------------------------- #
# Resolve — vectorized first-occurrence keep mask (identical to `in seen`)
# --------------------------------------------------------------------------- #


def resolve_keep_mask(digests: np.ndarray, empties: np.ndarray) -> np.ndarray:
    """Boolean keep mask over the global order: True iff the doc is non-empty
    and its digest has not occurred at any earlier global position."""
    keep = np.zeros(len(digests), dtype=bool)
    non_empty = empties == 0
    # First global occurrence per unique digest, restricted to non-empty docs.
    ne_positions = np.flatnonzero(non_empty)
    _, first_idx = np.unique(digests[ne_positions], return_index=True)
    keep[ne_positions[first_idx]] = True
    return keep


# --------------------------------------------------------------------------- #
# Pass 2 — parallel filtered rewrite, one task per unit
# --------------------------------------------------------------------------- #


def _write_unit(args: tuple[str, str, str, str | None, tuple[str, ...], str, int, int]) -> int:
    """Stream one unit's docs, apply its keep-mask slice, spool the survivors.

    Returns the number of docs written. args is a plain tuple for pickling:
    (out_root, source, unit_dir, worker, parts, keepmask_path, offset, count).
    """
    out_root, source, unit_dir, worker, parts, keep_path, offset, count = args
    keep = np.load(keep_path, mmap_mode="r")[offset : offset + count]
    udir = Path(unit_dir)

    def survivors():
        for i, text in enumerate(_iter_parts(udir, parts)):
            if keep[i]:
                yield text

    if worker is None:
        state = spool_documents(survivors(), out_root, source)
    else:
        wroot = Path(out_root) / source / WORKERS_DIRNAME
        wroot.mkdir(parents=True, exist_ok=True)
        meta = wroot / "meta.json"
        if not meta.exists():  # informational; layout is discovered via dir names
            meta.write_text(json.dumps({"workers": -1}) + "\n", "utf-8")
        state = spool_documents(survivors(), wroot, worker)
    return state.docs


def dedup_parallel(
    cfg: CorpusConfig,
    spool_root: str | PathLike[str],
    out_root: str | PathLike[str],
    *,
    hash_workers: int | None = None,
    write_workers: int | None = None,
    stats: DedupStats | None = None,
) -> DedupStats:
    """Run the full two-pass parallel dedup. Output at `out_root` is
    byte-identical (per source, in iter_source_documents order) to piping
    `dedup.dedup_documents` into per-source spools."""
    import multiprocessing  # noqa: PLC0415

    cpus = os.cpu_count() or 8
    hash_workers = hash_workers or max(4, min(64, cpus - 8))
    write_workers = write_workers or max(4, min(32, cpus // 4))
    stats = stats if stats is not None else DedupStats()

    units = plan_units(cfg, spool_root)
    total_docs = sum(u.docs for u in units)
    log.info(
        "parallel dedup starting",
        units=len(units),
        total_docs=total_docs,
        hash_workers=hash_workers,
        write_workers=write_workers,
    )

    digests, empties, unit_counts = hash_all_parts(units, workers=hash_workers)
    t0 = time.monotonic()
    keep = resolve_keep_mask(digests, empties)
    log.info(
        "keep mask resolved",
        examined=int(len(digests)),
        unique=int(keep.sum()),
        dropped=int((~keep & (empties == 0)).sum()),
        empty=int((empties == 1).sum()),
        resolve_s=round(time.monotonic() - t0, 1),
    )

    # Per-source stats, from the same arrays the writers will apply.
    cursor = 0
    for u, n in zip(units, unit_counts, strict=True):
        sl = slice(cursor, cursor + n)
        stats.kept[u.source] = stats.kept.get(u.source, 0) + int(keep[sl].sum())
        stats.empty[u.source] = stats.empty.get(u.source, 0) + int((empties[sl] == 1).sum())
        stats.dropped[u.source] = stats.dropped.get(u.source, 0) + int(
            (~keep[sl] & (empties[sl] == 0)).sum()
        )
        cursor += n

    with tempfile.TemporaryDirectory(prefix="mok-dedup-") as tmp:
        keep_path = str(Path(tmp) / "keep.npy")
        np.save(keep_path, keep)
        tasks = [
            (str(out_root), u.source, u.unit_dir, u.worker, u.parts, keep_path, u.offset, n)
            for u, n in zip(units, unit_counts, strict=True)
        ]
        t0 = time.monotonic()
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=min(write_workers, len(tasks))) as pool:
            written = list(pool.imap_unordered(_write_unit, tasks))
        log.info(
            "pass2 rewrite complete",
            units=len(tasks),
            docs_written=sum(written),
            write_s=round(time.monotonic() - t0, 1),
        )
    if sum(written) != stats.total_kept:
        raise RuntimeError(
            f"written docs ({sum(written)}) != kept per stats ({stats.total_kept})"
        )  # pragma: no cover
    return stats
