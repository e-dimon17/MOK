"""Per-source streaming download into resumable jsonl.zst spools.

The HuggingFace hub is touched only inside `_hf_documents` (lazy `datasets`
import); everything else is pure file plumbing, so tests inject their own
document iterators. A spool is a per-source directory of zstd-compressed
JSONL part files plus a `state.json` recording how many documents (in the
deterministic stream order) the completed parts hold. Resuming skips exactly
that many documents from a fresh stream and appends new parts; a crash can
only lose the in-flight `.tmp` part, never corrupt a committed one.
"""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import re
import shutil
import time
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from os import PathLike
from pathlib import Path

import zstandard
from pydantic import model_validator

from mok_core.config.loader import load_yaml
from mok_core.config.schemas import FrozenModel
from mok_core.telemetry import get_logger

log = get_logger("dataprep.download")

_NAME_RE = re.compile(r"[a-z0-9_]+")
STATE_FILENAME = "state.json"
_DEAD_STREAM_ROWS = 100_000  # rows without one usable doc => config error, not a slow stream
_SCAN_LOG_EVERY = 250_000    # row-scan heartbeat (min_score sources commit parts rarely)
_SKIP_LOG_EVERY = 2_000_000  # resume fast-forward progress cadence (docs)
WORKERS_DIRNAME = "workers"
WORKERS_META_FILENAME = "meta.json"
_WORKER_RE = re.compile(r"w\d{2}")


class SpoolLayoutError(RuntimeError):
    """Existing spool layout conflicts with the requested download mode."""


class SourceSpec(FrozenModel):
    """One corpus source: a HuggingFace dataset slice with a token budget."""

    name: str
    hf_path: str
    hf_name: str | None = None
    split: str = "train"
    text_column: str = "text"
    weight: float
    max_tokens: int
    score_column: str | None = None      # optional quality gate (e.g. fineweb-edu "score")
    min_score: float | None = None
    # Project the parquet read to these (string-typed) columns. Required for
    # sources whose subdirectories have heterogeneous schemas (starcoderdata):
    # extra columns then never reach the schema cast. Incompatible with
    # min_score (projection is string-only).
    hf_columns: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _check(self) -> SourceSpec:
        if self.hf_columns is not None:
            if self.text_column not in self.hf_columns:
                raise ValueError("hf_columns must include text_column")
            if self.min_score is not None:
                raise ValueError("hf_columns projection is string-only; incompatible with min_score")
        if not _NAME_RE.fullmatch(self.name):
            raise ValueError(f"source name must match [a-z0-9_]+, got {self.name!r}")
        if self.weight <= 0:
            raise ValueError(f"source {self.name}: weight must be positive")
        if self.max_tokens <= 0:
            raise ValueError(f"source {self.name}: max_tokens must be positive")
        if (self.score_column is None) != (self.min_score is None):
            raise ValueError(f"source {self.name}: score_column and min_score go together")
        return self


class CorpusConfig(FrozenModel):
    """A corpus build: ordered sources plus packing/dedup parameters.

    `sources` order IS the deterministic tokenize/pack order; `dedup_order`
    (a permutation of source names, smallest sources first per the playbook)
    controls which documents win when near-duplicates cross sources.
    """

    name: str
    seq_len: int = 4096
    chars_per_token: float = 4.0         # budget estimate used before the tokenizer exists
    dedup_order: tuple[str, ...] = ()
    sources: tuple[SourceSpec, ...]

    @model_validator(mode="after")
    def _check(self) -> CorpusConfig:
        if not self.sources:
            raise ValueError("corpus needs at least one source")
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError("source names must be unique")
        if self.dedup_order and sorted(self.dedup_order) != sorted(names):
            raise ValueError("dedup_order must be a permutation of the source names")
        if self.seq_len <= 0 or self.chars_per_token <= 0:
            raise ValueError("seq_len and chars_per_token must be positive")
        return self

    def source(self, name: str) -> SourceSpec:
        for s in self.sources:
            if s.name == name:
                return s
        raise KeyError(f"source {name!r} not in corpus {self.name!r}")

    def dedup_sequence(self) -> tuple[SourceSpec, ...]:
        """Sources in dedup priority order (earlier = kept on collision)."""
        if not self.dedup_order:
            return self.sources
        return tuple(self.source(n) for n in self.dedup_order)

    def char_budget(self, spec: SourceSpec) -> int:
        return int(spec.max_tokens * self.chars_per_token)


def load_corpus_config(path: str | PathLike[str]) -> CorpusConfig:
    return CorpusConfig.model_validate(load_yaml(path))


# --------------------------------------------------------------------------- #
# Spool files
# --------------------------------------------------------------------------- #


class SpoolState(FrozenModel):
    """Progress record for one source spool (state.json)."""

    source: str
    docs: int = 0
    chars: int = 0
    parts: tuple[str, ...] = ()
    complete: bool = False


def _source_dir(spool_root: str | PathLike[str], source: str) -> Path:
    return Path(spool_root) / source


def load_spool_state(spool_root: str | PathLike[str], source: str) -> SpoolState:
    path = _source_dir(spool_root, source) / STATE_FILENAME
    if not path.exists():
        return SpoolState(source=source)
    with open(path, encoding="utf-8") as f:
        return SpoolState.model_validate(json.load(f))


def _write_state(sdir: Path, state: SpoolState) -> None:
    tmp = sdir / (STATE_FILENAME + ".tmp")
    tmp.write_text(json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", "utf-8")
    os.replace(tmp, sdir / STATE_FILENAME)


def spool_documents(
    docs: Iterable[str],
    spool_root: str | PathLike[str],
    source: str,
    *,
    char_budget: int | None = None,
    part_docs: int = 100_000,
) -> SpoolState:
    """Append `docs` to the source spool until exhaustion or the char budget.

    Resumable: documents already recorded in state.json are skipped from the
    front of `docs` (streams are deterministic), and each committed part file
    is atomic (`.tmp` + rename). Returns the final state (`complete=True`).
    """
    if part_docs <= 0:
        raise ValueError(f"part_docs must be positive, got {part_docs}")
    sdir = _source_dir(spool_root, source)
    sdir.mkdir(parents=True, exist_ok=True)
    for stale in sdir.glob("*.tmp"):
        stale.unlink()

    state = load_spool_state(spool_root, source)
    if state.complete:
        log.info("spool already complete", source=source, docs=state.docs, chars=state.chars)
        return state

    if state.docs:
        log.info("resuming spool — re-skipping committed docs", source=source, skip_docs=state.docs)
    it = iter(docs)
    skip_t0 = time.monotonic()
    for skipped in range(state.docs):  # re-skip what earlier sessions committed
        try:
            next(it)
        except StopIteration:
            state = state.model_copy(update={"complete": True})
            _write_state(sdir, state)
            return state
        if (skipped + 1) % _SKIP_LOG_EVERY == 0:
            elapsed = max(time.monotonic() - skip_t0, 1e-9)
            rate = (skipped + 1) / elapsed
            log.info(
                "re-skip progress",
                source=source,
                skipped=skipped + 1,
                skip_total=state.docs,
                skip_pct=round(100.0 * (skipped + 1) / state.docs, 1),
                docs_per_s=round(rate, 1),
                eta_s=round((state.docs - skipped - 1) / rate),
            )

    docs_total, chars_total, parts = state.docs, state.chars, list(state.parts)
    cctx = zstandard.ZstdCompressor()
    exhausted = False
    session_t0 = time.monotonic()
    session_docs0, session_chars0, session_bytes = docs_total, chars_total, 0

    def budget_spent() -> bool:
        return char_budget is not None and chars_total >= char_budget

    while not exhausted and not budget_spent():
        part_name = f"part-{len(parts):05d}.jsonl.zst"
        tmp = sdir / (part_name + ".tmp")
        wrote = 0
        with open(tmp, "wb") as fh, cctx.stream_writer(fh) as zw:
            while wrote < part_docs and not budget_spent():
                try:
                    text = next(it)
                except StopIteration:
                    exhausted = True
                    break
                zw.write((json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
                wrote += 1
                chars_total += len(text)
        if wrote == 0:
            tmp.unlink()
            break
        os.replace(tmp, sdir / part_name)
        parts.append(part_name)
        docs_total += wrote
        _write_state(
            sdir,
            SpoolState(source=source, docs=docs_total, chars=chars_total, parts=tuple(parts)),
        )
        session_bytes += (sdir / part_name).stat().st_size
        elapsed = max(time.monotonic() - session_t0, 1e-9)
        budget_pct = (
            round(100.0 * min(chars_total, char_budget) / char_budget, 1) if char_budget else None
        )
        log.info(
            "spool part committed",
            source=source,
            part=part_name,
            parts=len(parts),
            docs=docs_total,
            chars=chars_total,
            spool_mb=round(session_bytes / 1e6, 1),
            docs_per_s=round((docs_total - session_docs0) / elapsed, 1),
            raw_mb_per_s=round((chars_total - session_chars0) / elapsed / 1e6, 2),
            budget_pct=budget_pct,
        )

    state = SpoolState(
        source=source, docs=docs_total, chars=chars_total, parts=tuple(parts), complete=True
    )
    _write_state(sdir, state)
    log.info(
        "spool complete",
        source=source,
        docs=docs_total,
        chars=chars_total,
        parts=len(parts),
        reason="stream exhausted" if exhausted else "char budget reached",
    )
    return state


def _iter_parts(sdir: Path, parts: Iterable[str]) -> Iterator[str]:
    dctx = zstandard.ZstdDecompressor()
    for part in parts:
        with open(sdir / part, "rb") as fh, dctx.stream_reader(fh) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                yield json.loads(line)["text"]


def worker_names(spool_root: str | PathLike[str], source: str) -> list[str]:
    """Sorted worker sub-spool names (w00, w01, ...) of a parallel download, if any."""
    wroot = _source_dir(spool_root, source) / WORKERS_DIRNAME
    if not wroot.is_dir():
        return []
    return sorted(p.name for p in wroot.iterdir() if p.is_dir() and _WORKER_RE.fullmatch(p.name))


def iter_source_documents(spool_root: str | PathLike[str], source: str) -> Iterator[str]:
    """Documents of one spooled source, in the exact order they were written.

    Deterministic across single-stream and parallel layouts: legacy parts
    first, then each worker sub-spool's parts in worker order.
    """
    state = load_spool_state(spool_root, source)
    sdir = _source_dir(spool_root, source)
    yield from _iter_parts(sdir, state.parts)
    wroot = sdir / WORKERS_DIRNAME
    for wname in worker_names(spool_root, source):
        wstate = load_spool_state(wroot, wname)
        yield from _iter_parts(wroot / wname, wstate.parts)


# --------------------------------------------------------------------------- #
# HuggingFace streaming
# --------------------------------------------------------------------------- #


def _hf_documents(
    spec: SourceSpec, *, num_shards: int | None = None, shard_index: int | None = None
) -> Iterator[str]:
    """Stream the source from the HF hub in deterministic order (lazy import).

    With `num_shards`/`shard_index`, streams the deterministic file-level shard
    `shard_index` of `num_shards` (datasets.IterableDataset.shard) — the basis
    of `--workers` parallel downloads.
    """
    from datasets import load_dataset

    kwargs: dict = {}
    if spec.hf_columns is not None:
        from datasets import Features, Value  # noqa: PLC0415

        kwargs["columns"] = list(spec.hf_columns)
        kwargs["features"] = Features(dict.fromkeys(spec.hf_columns, Value("string")))
    ds = load_dataset(
        spec.hf_path, name=spec.hf_name, split=spec.split, streaming=True, **kwargs
    )
    if num_shards is not None:
        if shard_index is None or not 0 <= shard_index < num_shards:
            raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
        ds = ds.shard(num_shards=num_shards, index=shard_index)
    scanned = yielded = 0
    for row in ds:
        scanned += 1
        # Filtered sources (min_score) discard most rows, so parts commit
        # rarely — log the scan itself or the stream looks hung for its
        # first part (score>=4 keeps ~10%: ~1M rows scanned per 100k part).
        if scanned % _SCAN_LOG_EVERY == 0:
            log.info(
                "scan progress",
                source=spec.name,
                rows_scanned=scanned,
                docs_kept=yielded,
                keep_pct=round(100.0 * yielded / scanned, 1),
            )
        text = row.get(spec.text_column)
        if isinstance(text, str) and text:
            if spec.min_score is not None:
                score = row.get(spec.score_column)
                if score is None or float(score) < spec.min_score:
                    continue
            yielded += 1
            yield text
        # A source whose rows never carry the text column is a config error
        # (e.g. an IDs-only dataset) — fail loudly, never spin silently.
        if yielded == 0 and scanned >= _DEAD_STREAM_ROWS:
            raise RuntimeError(
                f"source {spec.name!r}: scanned {scanned} rows of {spec.hf_path} without a single "
                f"usable {spec.text_column!r} value — wrong text_column, an IDs-only dataset, "
                "or a min_score filter that rejects everything. Fix the corpus config."
            )


def _spool_worker(
    spec_data: dict,
    spool_root: str,
    num_workers: int,
    worker_index: int,
    char_budget_share: int | None,
    part_docs: int,
) -> dict:
    """Subprocess entry: spool one deterministic file-shard of the source."""
    from mok_core.telemetry import bind, setup_logging  # noqa: PLC0415

    setup_logging(os.environ.get("MOK_LOG_LEVEL", "INFO"))
    spec = SourceSpec.model_validate(spec_data)
    bind(parent_source=spec.name, worker=worker_index)
    docs = _hf_documents(spec, num_shards=num_workers, shard_index=worker_index)
    wroot = Path(spool_root) / spec.name / WORKERS_DIRNAME
    state = spool_documents(
        docs, wroot, f"w{worker_index:02d}", char_budget=char_budget_share, part_docs=part_docs
    )
    return state.model_dump(mode="json")


def _prepare_worker_root(
    spool_root: str | PathLike[str], spec: SourceSpec, num_workers: int, discard_legacy: bool
) -> Path:
    """Validate/initialize the parallel layout; guards against mixing layouts."""
    sdir = _source_dir(spool_root, spec.name)
    legacy = load_spool_state(spool_root, spec.name)
    if legacy.docs or legacy.parts:
        if not discard_legacy:
            raise SpoolLayoutError(
                f"source {spec.name!r} has a single-stream spool with {legacy.docs} docs; "
                "worker shards would duplicate it. Rerun with --discard-legacy to delete it "
                "(usually net-faster), or continue without --workers."
            )
        shutil.rmtree(sdir)
        log.info("discarded legacy single-stream spool", source=spec.name, docs=legacy.docs)
    wroot = sdir / WORKERS_DIRNAME
    wroot.mkdir(parents=True, exist_ok=True)
    meta_path = wroot / WORKERS_META_FILENAME
    if meta_path.exists():
        existing = json.loads(meta_path.read_text("utf-8"))["workers"]
        if existing != num_workers:
            raise SpoolLayoutError(
                f"source {spec.name!r} was started with --workers {existing}; resume with the "
                f"same value (got {num_workers}) — shard membership depends on it."
            )
    else:
        meta_path.write_text(json.dumps({"workers": num_workers}) + "\n", "utf-8")
    return wroot


def _merge_worker_states(spec: SourceSpec, states: Sequence[SpoolState]) -> SpoolState:
    return SpoolState(
        source=spec.name,
        docs=sum(s.docs for s in states),
        chars=sum(s.chars for s in states),
        parts=(),  # parts live in the worker sub-spools; iterate via iter_source_documents
        complete=all(s.complete for s in states),
    )


def download_source(
    cfg: CorpusConfig,
    spec: SourceSpec,
    spool_root: str | PathLike[str],
    *,
    doc_iter: Iterable[str] | None = None,
    part_docs: int = 100_000,
    workers: int = 1,
    discard_legacy: bool = False,
    worker_doc_iters: Sequence[Iterable[str]] | None = None,
) -> SpoolState:
    """Spool one source up to its char budget.

    `workers > 1` shards the source's data files across that many subprocesses
    (deterministic per worker; resumable per worker; the char budget is split
    evenly). `worker_doc_iters` runs the same layout in-process — tests only.
    `doc_iter` overrides the HF stream in the single-stream path.
    """
    if workers <= 1 and worker_doc_iters is None:
        docs = doc_iter if doc_iter is not None else _hf_documents(spec)
        return spool_documents(
            docs, spool_root, spec.name, char_budget=cfg.char_budget(spec), part_docs=part_docs
        )

    n = len(worker_doc_iters) if worker_doc_iters is not None else workers
    if n < 1:
        raise ValueError("workers must be >= 1")
    wroot = _prepare_worker_root(spool_root, spec, n, discard_legacy)
    budget = cfg.char_budget(spec)
    share = None if budget is None else max(1, budget // n)

    if worker_doc_iters is not None:
        states = [
            spool_documents(it, wroot, f"w{i:02d}", char_budget=share, part_docs=part_docs)
            for i, it in enumerate(worker_doc_iters)
        ]
        return _merge_worker_states(spec, states)

    log.info("parallel download starting", source=spec.name, workers=n, char_budget_per_worker=share)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _spool_worker, spec.model_dump(mode="json"), str(spool_root), n, i, share, part_docs
            )
            for i in range(n)
        ]
        states = []
        for i, f in enumerate(futures):
            try:
                states.append(SpoolState.model_validate(f.result()))
            except Exception as e:
                raise RuntimeError(
                    f"download worker {i} of source {spec.name!r} failed: {type(e).__name__}: {e}. "
                    "Committed parts are safe — fix the cause and rerun with the same --workers value."
                ) from e
    merged = _merge_worker_states(spec, states)
    log.info(
        "parallel download complete",
        source=spec.name,
        workers=n,
        docs=merged.docs,
        chars=merged.chars,
    )
    return merged


def download_corpus(
    cfg: CorpusConfig,
    spool_root: str | PathLike[str],
    *,
    only: str | None = None,
    part_docs: int = 100_000,
    workers: int = 1,
    discard_legacy: bool = False,
) -> dict[str, SpoolState]:
    """Spool every source (or just `only`) in config order; returns final states."""
    specs = [cfg.source(only)] if only is not None else list(cfg.sources)
    return {
        s.name: download_source(
            cfg, s, spool_root, part_docs=part_docs, workers=workers, discard_legacy=discard_legacy
        )
        for s in specs
    }
