"""End-to-end `mok-data` CLI runs on fixture text — no network (dataprep/cli.py)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from dataprep.cli import main
from dataprep.pipeline.build_manifest import MANIFEST_FILENAME, SHARD_INDEX_FILENAME, load_manifest_ref
from dataprep.pipeline.download import spool_documents
from dataprep.pipeline.verify import verify_local

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_corpus.txt"


def run_smoke(out_dir: Path, capsys) -> tuple[int, str]:
    rc = main(["--smoke", str(FIXTURE), "--out", str(out_dir)])
    return rc, capsys.readouterr().out


def extract_root(stdout: str) -> str:
    m = re.search(r"^merkle_root=([0-9a-f]{64})$", stdout, re.M)
    assert m, f"no merkle root printed in:\n{stdout}"
    return m.group(1)


def test_smoke_end_to_end(tmp_path, capsys):
    out = tmp_path / "smoke"
    rc, stdout = run_smoke(out, capsys)
    assert rc == 0
    root = extract_root(stdout)
    assert (out / "tokenizer.json").exists()
    assert (out / SHARD_INDEX_FILENAME).exists()
    assert list(out.glob("shard-*.bin"))
    ref = load_manifest_ref(out / MANIFEST_FILENAME)
    assert ref.name == "smoke"
    assert ref.merkle_root == root
    assert ref.seq_len == 128
    assert ref.tokens_total > 0
    report = verify_local(out)
    assert report.ok


def test_smoke_is_bit_deterministic(tmp_path, capsys):
    rc1, out1 = run_smoke(tmp_path / "a", capsys)
    rc2, out2 = run_smoke(tmp_path / "b", capsys)
    assert rc1 == rc2 == 0
    assert extract_root(out1) == extract_root(out2)
    ref_a = load_manifest_ref(tmp_path / "a" / MANIFEST_FILENAME)
    ref_b = load_manifest_ref(tmp_path / "b" / MANIFEST_FILENAME)
    assert ref_a == ref_b


def test_verify_subcommand_detects_tamper(tmp_path, capsys):
    out = tmp_path / "smoke"
    rc, _ = run_smoke(out, capsys)
    assert rc == 0
    assert main(["verify", "--data-dir", str(out)]) == 0
    shard = sorted(out.glob("shard-*.bin"))[0]
    data = bytearray(shard.read_bytes())
    data[3] ^= 0x01
    shard.write_bytes(bytes(data))
    assert main(["verify", "--data-dir", str(out)]) == 1


def test_no_subcommand_prints_help(capsys):
    assert main([]) == 2


def paragraphs() -> list[str]:
    return [d.strip() for d in re.split(r"\n\s*\n", FIXTURE.read_text("utf-8")) if d.strip()]


def test_full_stage_chain(tmp_path, capsys):
    """dedup -> tokenizer -> tokenize -> shard -> manifest -> verify via subcommands."""
    docs = paragraphs()
    spool = tmp_path / "spool"
    spool_documents(docs[:10], spool, "src_a")
    spool_documents(docs[10:] + [docs[0]], spool, "src_b")  # cross-source duplicate of src_a's first doc

    cfg = {
        "name": "mini",
        "seq_len": 64,
        "dedup_order": ["src_b", "src_a"],
        "sources": [
            {"name": "src_a", "hf_path": "dummy/a", "weight": 0.7, "max_tokens": 100000},
            {"name": "src_b", "hf_path": "dummy/b", "weight": 0.3, "max_tokens": 100000},
        ],
    }
    corpus_yaml = tmp_path / "corpus.yaml"
    corpus_yaml.write_text(yaml.safe_dump(cfg))
    tok_yaml = tmp_path / "tok.yaml"
    tok_yaml.write_text(yaml.safe_dump({"vocab_size": 400, "min_frequency": 2}))

    deduped = tmp_path / "deduped"
    assert (
        main(
            [
                "dedup",
                "--corpus-config", str(corpus_yaml),
                "--spool-dir", str(spool),
                "--out-dir", str(deduped),
            ]
        )
        == 0
    )
    dedup_out = capsys.readouterr().out
    assert "dropped=1" in dedup_out  # src_a's copy of docs[0] lost to src_b (higher priority)

    tok_path = tmp_path / "tokenizer.json"
    assert (
        main(
            [
                "tokenizer",
                "--corpus-config", str(corpus_yaml),
                "--tokenizer-config", str(tok_yaml),
                "--spool-dir", str(deduped),
                "--out", str(tok_path),
            ]
        )
        == 0
    )
    assert "tokenizer_hash=" in capsys.readouterr().out

    tokens_dir = tmp_path / "tokens"
    assert (
        main(
            [
                "tokenize",
                "--corpus-config", str(corpus_yaml),
                "--spool-dir", str(deduped),
                "--tokenizer", str(tok_path),
                "--out-dir", str(tokens_dir),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (tokens_dir / "src_a.tokens.u16").exists()
    assert (tokens_dir / "src_b.tokens.u16").exists()

    shards_dir = tmp_path / "shards"
    assert (
        main(
            [
                "shard",
                "--corpus-config", str(corpus_yaml),
                "--tokens-dir", str(tokens_dir),
                "--out-dir", str(shards_dir),
                "--shard-sequences", "4",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "manifest",
                "--corpus-config", str(corpus_yaml),
                "--out-dir", str(shards_dir),
                "--tokenizer", str(tok_path),
                "--shard-sequences", "4",
            ]
        )
        == 0
    )
    root = extract_root(capsys.readouterr().out)

    assert main(["verify", "--data-dir", str(shards_dir)]) == 0
    assert extract_root(capsys.readouterr().out) == root
    ref = load_manifest_ref(shards_dir / MANIFEST_FILENAME)
    assert ref.name == "mini"
    assert ref.seq_len == 64
