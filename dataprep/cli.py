"""`mok-data` — the data-preparation application.

Subcommands mirror the playbook method order: tokenizer, download, dedup,
tokenize, shard, manifest, upload, verify. `--smoke` wires a tiny end-to-end
run from local text files (no network): sample corpus -> tokenizer ->
tokenize -> pack -> shards -> manifest -> verify, printing the Merkle root.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterator
from itertools import groupby
from pathlib import Path

from dataprep.pipeline.build_manifest import (
    MANIFEST_FILENAME,
    TOKENIZER_FILENAME,
    build_dataset_manifest,
    load_manifest_ref,
)
from dataprep.pipeline.dedup import DedupStats, dedup_documents
from dataprep.pipeline.download import (
    CorpusConfig,
    download_corpus,
    iter_source_documents,
    load_corpus_config,
    spool_documents,
)
from dataprep.pipeline.shard_writer import (
    FULL_SHARD_SEQUENCES,
    SHARD_METAS_FILENAME,
    load_shard_metas,
    save_shard_metas,
    write_shards,
)
from dataprep.pipeline.tokenize_pack import (
    chunk_token_arrays,
    encode_documents,
    iter_token_file_arrays,
    pack_documents,
    write_token_stream,
)
from dataprep.pipeline.tokenizer_train import (
    EOS_ID,
    TokenizerConfig,
    load_tokenizer_config,
    tokenizer_file_hash,
    train_tokenizer,
)
from dataprep.pipeline.upload import upload_dataset_sync
from dataprep.pipeline.verify import verify_local

_PARAGRAPH_RE = re.compile(r"\n\s*\n")

# Smoke-profile defaults: small enough for CI, large enough to produce shards.
_SMOKE_SEQ_LEN = 128
_SMOKE_SHARD_SEQUENCES = 32
_SMOKE_VOCAB = 512


def _iter_docs_from_files(paths: list[str]) -> Iterator[str]:
    """Local text files split into blank-line-separated paragraph documents."""
    for p in paths:
        for doc in _PARAGRAPH_RE.split(Path(p).read_text("utf-8")):
            doc = doc.strip()
            if doc:
                yield doc


def _sample_texts(cfg: CorpusConfig, spool_dir: Path, total_chars: int) -> Iterator[str]:
    """Weight-proportional tokenizer-training sample, deterministic source order."""
    weight_sum = sum(s.weight for s in cfg.sources)
    for spec in cfg.sources:
        budget = int(total_chars * spec.weight / weight_sum)
        seen = 0
        for text in iter_source_documents(spool_dir, spec.name):
            if seen >= budget:
                break
            yield text
            seen += len(text)


# --------------------------------------------------------------------------- #
# Subcommand handlers
# --------------------------------------------------------------------------- #


def _cmd_tokenizer(args: argparse.Namespace) -> int:
    cfg = load_corpus_config(args.corpus_config)
    tcfg = load_tokenizer_config(args.tokenizer_config) if args.tokenizer_config else TokenizerConfig()
    texts = _sample_texts(cfg, Path(args.spool_dir), tcfg.sample_chars_total)
    trained = train_tokenizer(texts, args.out, tcfg)
    print(f"tokenizer={trained.path} tokenizer_hash={trained.tokenizer_hash}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    cfg = load_corpus_config(args.corpus_config)
    states = download_corpus(
        cfg,
        args.spool_dir,
        only=args.source,
        part_docs=args.part_docs,
        workers=args.workers,
        discard_legacy=args.discard_legacy,
    )
    for name, state in states.items():
        print(f"source={name} docs={state.docs} chars={state.chars} complete={state.complete}")
    return 0


def _cmd_dedup(args: argparse.Namespace) -> int:
    cfg = load_corpus_config(args.corpus_config)
    if args.parallel:
        from dataprep.pipeline.dedup_parallel import dedup_parallel  # noqa: PLC0415

        stats = dedup_parallel(
            cfg, args.spool_dir, args.out_dir, hash_workers=args.workers or None
        )
        for spec in cfg.dedup_sequence():
            n = spec.name
            print(f"source={n} kept={stats.kept.get(n, 0)} dropped={stats.dropped.get(n, 0)}")
        return 0
    ordered = [
        (spec.name, iter_source_documents(args.spool_dir, spec.name)) for spec in cfg.dedup_sequence()
    ]
    stats = DedupStats()
    kept = dedup_documents(ordered, stats=stats, workers=args.workers)
    for source_name, group in groupby(kept, key=lambda kv: kv[0]):
        spool_documents((text for _, text in group), args.out_dir, source_name)
    for spec in cfg.dedup_sequence():
        n = spec.name
        print(f"source={n} kept={stats.kept.get(n, 0)} dropped={stats.dropped.get(n, 0)}")
    return 0


def _cmd_tokenize(args: argparse.Namespace) -> int:
    cfg = load_corpus_config(args.corpus_config)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.workers:
        from dataprep.pipeline.tokenize_pack import tokenize_parallel  # noqa: PLC0415

        totals = tokenize_parallel(
            cfg,
            args.spool_dir,
            args.tokenizer,
            out,
            workers=args.workers,
            parts_per_task=args.parts_per_task,
        )
        for spec in cfg.sources:
            print(f"source={spec.name} tokens={totals[spec.name]}")
        return 0
    for spec in cfg.sources:  # deterministic source order
        docs = iter_source_documents(args.spool_dir, spec.name)
        tokens = encode_documents(docs, args.tokenizer)
        total = write_token_stream(tokens, out / f"{spec.name}.tokens.u16", eos_id=EOS_ID)
        print(f"source={spec.name} tokens={total}")
    return 0


def _cmd_shard(args: argparse.Namespace) -> int:
    cfg = load_corpus_config(args.corpus_config)
    from dataprep.pipeline.tokenize_pack import discover_token_files  # noqa: PLC0415

    paths = discover_token_files(args.tokens_dir, [spec.name for spec in cfg.sources])
    seqs = chunk_token_arrays(iter_token_file_arrays(paths), cfg.seq_len)
    metas = write_shards(seqs, args.out_dir, shard_sequences=args.shard_sequences, seq_len=cfg.seq_len)
    if not metas:
        print("error: corpus produced zero full sequences — nothing to shard", file=sys.stderr)
        return 1
    save_shard_metas(metas, Path(args.out_dir) / SHARD_METAS_FILENAME)
    print(f"shards={len(metas)} sequences={sum(m.num_sequences for m in metas)}")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    cfg = load_corpus_config(args.corpus_config)
    metas = load_shard_metas(Path(args.out_dir) / SHARD_METAS_FILENAME)
    _, ref = build_dataset_manifest(
        metas,
        name=cfg.name,
        seq_len=cfg.seq_len,
        tokenizer_hash=tokenizer_file_hash(args.tokenizer),
        out_dir=args.out_dir,
        shard_sequences=args.shard_sequences,
    )
    print(f"dataset={ref.name} shards={ref.num_shards} tokens_total={ref.tokens_total}")
    print(f"merkle_root={ref.merkle_root}")
    return 0


def _cmd_upload(args: argparse.Namespace) -> int:
    prefix = args.prefix
    if prefix is None:
        manifest = Path(args.data_dir) / MANIFEST_FILENAME
        if not manifest.exists():
            print("error: --prefix not given and no manifest.json to derive it from", file=sys.stderr)
            return 1
        prefix = f"datasets/{load_manifest_ref(manifest).name}"
    report = upload_dataset_sync(
        args.data_dir,
        prefix=prefix,
        bucket=args.bucket,
        endpoint_url=args.endpoint,
        access_key_id=args.access_key_id,
        secret_access_key=args.secret_access_key,
        concurrency=args.concurrency,
    )
    print(
        f"uploaded={len(report.uploaded)} skipped={len(report.skipped)} bytes_sent={report.bytes_sent}"
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify_local(args.data_dir, sample=args.sample, seed=args.seed, workers=args.workers)
    for failure in report.failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    print(
        f"dataset={report.dataset} shards_hashed={report.shards_hashed}/{report.num_shards} "
        f"tokens_total={report.tokens_total} ok={report.ok}"
    )
    print(f"merkle_root={report.merkle_root}")
    return 0 if report.ok else 1


# --------------------------------------------------------------------------- #
# Smoke profile
# --------------------------------------------------------------------------- #


def _run_smoke(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs = list(_iter_docs_from_files(args.smoke))
    if not docs:
        print("error: smoke corpus files contain no documents", file=sys.stderr)
        return 1
    seq_len = args.seq_len or _SMOKE_SEQ_LEN
    shard_sequences = args.shard_sequences or _SMOKE_SHARD_SEQUENCES

    tcfg = TokenizerConfig(vocab_size=args.vocab_size or _SMOKE_VOCAB, min_frequency=2)
    trained = train_tokenizer(iter(docs), out / TOKENIZER_FILENAME, tcfg)

    token_iters = encode_documents(iter(docs), trained.path)
    seqs = pack_documents(token_iters, seq_len=seq_len, eos_id=EOS_ID)
    metas = write_shards(seqs, out, shard_sequences=shard_sequences, seq_len=seq_len)
    if not metas:
        print(f"error: corpus too small for even one {seq_len}-token sequence", file=sys.stderr)
        return 1
    save_shard_metas(metas, out / SHARD_METAS_FILENAME)

    _, ref = build_dataset_manifest(
        metas,
        name="smoke",
        seq_len=seq_len,
        tokenizer_hash=trained.tokenizer_hash,
        out_dir=out,
        shard_sequences=shard_sequences,
    )
    report = verify_local(out)
    if not report.ok:
        for failure in report.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print(
        f"dataset=smoke docs={len(docs)} shards={ref.num_shards} "
        f"tokens_total={ref.tokens_total} tokenizer_hash={ref.tokenizer_hash}"
    )
    print(f"merkle_root={ref.merkle_root}")
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mok-data", description="MoK subnet data preparation")
    p.add_argument(
        "--smoke",
        nargs="+",
        metavar="TEXT_FILE",
        help="tiny end-to-end run from local text files (no network); prints the merkle root",
    )
    p.add_argument("--out", default="mok-data-smoke", help="output directory for --smoke")
    p.add_argument("--seq-len", type=int, default=None, help="--smoke: tokens per packed sequence")
    p.add_argument("--shard-sequences", type=int, default=None, help="--smoke: sequences per shard")
    p.add_argument("--vocab-size", type=int, default=None, help="--smoke: tokenizer vocab size")

    sub = p.add_subparsers(dest="cmd")

    tk = sub.add_parser("tokenizer", help="train the frozen 65k BPE tokenizer from spooled text")
    tk.add_argument("--corpus-config", required=True)
    tk.add_argument("--tokenizer-config", default=None)
    tk.add_argument("--spool-dir", required=True)
    tk.add_argument("--out", required=True, help="path to write tokenizer.json")
    tk.set_defaults(func=_cmd_tokenizer)

    dl = sub.add_parser("download", help="stream sources from the HF hub into resumable spools")
    dl.add_argument("--corpus-config", required=True)
    dl.add_argument("--spool-dir", required=True)
    dl.add_argument("--source", default=None, help="only this source")
    dl.add_argument("--part-docs", type=int, default=100_000)
    dl.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel download processes per source (deterministic file shards); resume with the SAME value",
    )
    dl.add_argument(
        "--discard-legacy",
        action="store_true",
        help="with --workers: delete an existing single-stream spool instead of refusing to mix layouts",
    )
    dl.set_defaults(func=_cmd_download)

    dd = sub.add_parser("dedup", help="cross-source exact-hash dedup into a new spool dir")
    dd.add_argument("--corpus-config", required=True)
    dd.add_argument("--spool-dir", required=True)
    dd.add_argument("--out-dir", required=True)
    dd.add_argument("--workers", type=int, default=0, help="hashing worker processes (0 = inline)")
    dd.add_argument(
        "--parallel",
        action="store_true",
        help="two-pass fully-parallel dedup (byte-identical output; uses all cores)",
    )
    dd.set_defaults(func=_cmd_dedup)

    tz = sub.add_parser("tokenize", help="tokenize spooled sources into flat uint16 token streams")
    tz.add_argument("--corpus-config", required=True)
    tz.add_argument("--spool-dir", required=True, help="deduped spool directory")
    tz.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    tz.add_argument("--out-dir", required=True)
    tz.add_argument("--workers", type=int, default=0, help="parallel unit tokenizers (0 = sequential)")
    tz.add_argument("--parts-per-task", type=int, default=32, help="spool parts per chunk task (balance knob)")
    tz.set_defaults(func=_cmd_tokenize)

    sh = sub.add_parser("shard", help="pack token streams into content-addressed shards")
    sh.add_argument("--corpus-config", required=True)
    sh.add_argument("--tokens-dir", required=True)
    sh.add_argument("--out-dir", required=True)
    sh.add_argument("--shard-sequences", type=int, default=FULL_SHARD_SEQUENCES)
    sh.set_defaults(func=_cmd_shard)

    mf = sub.add_parser("manifest", help="build shard_index.json + manifest.json (Merkle root)")
    mf.add_argument("--corpus-config", required=True)
    mf.add_argument("--out-dir", required=True, help="shard directory (holds shards.json)")
    mf.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    mf.add_argument("--shard-sequences", type=int, default=FULL_SHARD_SEQUENCES)
    mf.set_defaults(func=_cmd_manifest)

    up = sub.add_parser("upload", help="upload shards + index + tokenizer to R2/S3 (resumable)")
    up.add_argument("--data-dir", required=True)
    up.add_argument("--prefix", default=None, help="key prefix (default datasets/<name>)")
    up.add_argument("--bucket", default=None, help="default $R2_BUCKET_NAME")
    up.add_argument("--endpoint", default=None, help="default https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com")
    up.add_argument("--access-key-id", default=None, help="default $R2_WRITE_ACCESS_KEY_ID")
    up.add_argument("--secret-access-key", default=None, help="default $R2_WRITE_SECRET_ACCESS_KEY")
    up.add_argument("--concurrency", type=int, default=4)
    up.set_defaults(func=_cmd_upload)

    vf = sub.add_parser("verify", help="re-hash shards vs index and recheck the merkle root")
    vf.add_argument("--data-dir", required=True)
    vf.add_argument("--sample", type=int, default=None, help="hash only N sampled shards")
    vf.add_argument("--seed", type=int, default=0)
    vf.add_argument("--workers", type=int, default=None, help="parallel hash processes (default cpu//2, cap 16)")
    vf.set_defaults(func=_cmd_verify)

    return p


def main(argv: list[str] | None = None) -> int:
    import os  # noqa: PLC0415

    from mok_core.telemetry import setup_logging  # noqa: PLC0415

    setup_logging(os.environ.get("MOK_LOG_LEVEL", "INFO"))
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _run_smoke(args)
    if args.cmd is None:
        parser.print_help(sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
