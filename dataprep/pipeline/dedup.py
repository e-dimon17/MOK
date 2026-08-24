"""Cross-source exact-duplicate removal over normalized text.

Documents are whitespace-normalized and lowercased, hashed with xxh3-64, and
dropped when the digest was already seen. Sources are processed in the
caller's priority order (smallest first per the playbook): the first document
with a given digest wins, within or across sources. Output preserves input
order — the result is deterministic, including under parallel hashing
(`workers > 0` uses an order-preserving process pool for the normalize+hash
step only; membership decisions stay in the parent).

Why exact-hash and not MinHash: at this corpus's scale (~1.9B documents,
9.2T chars) a Python MinHash-LSH index needs multi-TB RAM and months of CPU.
Every source we ship is already near-deduplicated internally by its publisher;
the cross-source risk worth removing is exact/boilerplate duplication, which
normalized hashing catches at ~5 orders of magnitude lower cost. 64-bit
digests over ~2e9 docs give an expected accidental-collision count of ~0.1
across the whole corpus — negligible, and a collision only drops one document.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from mok_core.telemetry import get_logger

log = get_logger("dataprep.dedup")

_LOG_EVERY = 2_000_000        # progress cadence (documents examined)
_CHUNKSIZE = 512              # docs per worker batch (order-preserving imap)


def normalize(text: str) -> str:
    """Whitespace-collapse + lowercase — the equivalence class for dedup."""
    return " ".join(text.split()).lower()


def doc_digest(text: str) -> int | None:
    """xxh3-64 of the normalized text; None for empty/whitespace-only docs.

    Consensus-adjacent constant: digests decide which documents enter the
    corpus. xxh3-64 with seed 0 is a stable, specified function.
    """
    import xxhash  # noqa: PLC0415

    norm = normalize(text)
    if not norm:
        return None
    return xxhash.xxh3_64_intdigest(norm.encode("utf-8"))


@dataclass
class DedupStats:
    kept: dict[str, int] = field(default_factory=dict)
    dropped: dict[str, int] = field(default_factory=dict)
    empty: dict[str, int] = field(default_factory=dict)

    def _bump(self, bucket: dict[str, int], source: str) -> None:
        bucket[source] = bucket.get(source, 0) + 1

    @property
    def total_kept(self) -> int:
        return sum(self.kept.values())

    @property
    def total_dropped(self) -> int:
        return sum(self.dropped.values())


def _digest_stream(texts: Iterable[str], workers: int) -> Iterator[int | None]:
    """Digests of `texts`, in order. workers > 0 offloads normalize+hash to a
    process pool via order-preserving imap; 0 computes inline."""
    if workers <= 0:
        for t in texts:
            yield doc_digest(t)
        return
    import multiprocessing  # noqa: PLC0415

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        yield from pool.imap(doc_digest, texts, chunksize=_CHUNKSIZE)


def dedup_documents(
    sources: Iterable[tuple[str, Iterable[str]]],
    *,
    stats: DedupStats | None = None,
    workers: int = 0,
) -> Iterator[tuple[str, str]]:
    """Yield (source_name, text) for every document that survives dedup.

    `sources` is an ordered iterable of (name, documents); earlier sources win
    collisions. Whitespace-only documents are dropped outright. `workers`
    parallelizes hashing only — output order and keep/drop decisions are
    identical for every worker count (tested).
    """
    seen: set[int] = set()
    examined = 0
    t0 = time.monotonic()
    for source_name, docs in sources:
        originals, for_hashing = itertools.tee(docs, 2)
        digests = _digest_stream(for_hashing, workers)
        for text, digest in zip(originals, digests, strict=True):
            examined += 1
            if digest is None:
                if stats is not None:
                    stats._bump(stats.empty, source_name)
            elif digest in seen:
                if stats is not None:
                    stats._bump(stats.dropped, source_name)
            else:
                seen.add(digest)
                if stats is not None:
                    stats._bump(stats.kept, source_name)
                yield source_name, text
            if examined % _LOG_EVERY == 0:
                elapsed = max(time.monotonic() - t0, 1e-9)
                log.info(
                    "dedup progress",
                    source=source_name,
                    examined=examined,
                    unique=len(seen),
                    dropped=examined - len(seen),
                    docs_per_s=round(examined / elapsed, 1),
                )
        log.info(
            "dedup source finished",
            source=source_name,
            examined=examined,
            unique=len(seen),
        )
