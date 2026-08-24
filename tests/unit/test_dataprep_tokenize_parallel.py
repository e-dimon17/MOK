"""Parallel tokenize must produce byte-identical streams to sequential."""

from __future__ import annotations

import pytest

from dataprep.pipeline.download import CorpusConfig, iter_source_documents, spool_documents
from dataprep.pipeline.tokenize_pack import (
    discover_token_files,
    encode_documents,
    tokenize_parallel,
    write_token_stream,
)
from dataprep.pipeline.tokenizer_train import EOS_ID, TokenizerConfig, train_tokenizer


def _cfg() -> CorpusConfig:
    return CorpusConfig.model_validate(
        {
            "name": "t",
            "sources": [
                {"name": "alpha", "hf_path": "x/a", "weight": 0.5, "max_tokens": 1000},
                {"name": "beta", "hf_path": "x/b", "weight": 0.5, "max_tokens": 1000},
            ],
        }
    )


@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    docs = [f"sample document {i} with words to learn from" for i in range(400)]
    out = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    train_tokenizer(iter(docs), out, TokenizerConfig(vocab_size=300, sample_chars_total=10**6))
    return out


def _build_spools(root):
    a_docs = [f"alpha doc {i} some words here" for i in range(60)]
    b0 = [f"beta w0 doc {i} other words" for i in range(40)]
    b1 = [f"beta w1 doc {i} more words" for i in range(40)]
    spool_documents(iter(a_docs), root, "alpha", part_docs=25)          # legacy layout
    spool_documents(iter(b0), root / "beta" / "workers", "w00", part_docs=25)
    spool_documents(iter(b1), root / "beta" / "workers", "w01", part_docs=25)


def test_parallel_tokenize_bytes_equal_sequential(tmp_path, tiny_tokenizer):
    root = tmp_path / "spool"
    _build_spools(root)
    cfg = _cfg()

    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()
    for spec in cfg.sources:
        docs = iter_source_documents(root, spec.name)
        tokens = encode_documents(docs, tiny_tokenizer)
        write_token_stream(tokens, seq_dir / f"{spec.name}.tokens.u16", eos_id=EOS_ID)

    par_dir = tmp_path / "par"
    # parts_per_task=1 forces one chunk per spool part — the maximum-split case
    totals = tokenize_parallel(cfg, root, tiny_tokenizer, par_dir, workers=2, parts_per_task=1)
    assert set(totals) == {"alpha", "beta"} and all(v > 0 for v in totals.values())
    # alpha: 60 docs / 25-doc parts = 3 chunks; beta: 2 units x 2 parts = 4 chunks
    assert len(discover_token_files(par_dir, ["alpha"])) == 3
    assert len(discover_token_files(par_dir, ["beta"])) == 4

    for spec in cfg.sources:
        seq_bytes = (seq_dir / f"{spec.name}.tokens.u16").read_bytes()
        par_paths = discover_token_files(par_dir, [spec.name])
        par_bytes = b"".join(p.read_bytes() for p in par_paths)
        assert par_bytes == seq_bytes, f"{spec.name}: parallel token stream diverged"


def test_discover_token_files_layouts(tmp_path):
    (tmp_path / "one.tokens.u16").write_bytes(b"\x01\x00")
    (tmp_path / "two.000.tokens.u16").write_bytes(b"\x02\x00")
    (tmp_path / "two.001.tokens.u16").write_bytes(b"\x03\x00")
    paths = discover_token_files(tmp_path, ["one", "two"])
    assert [p.name for p in paths] == ["one.tokens.u16", "two.000.tokens.u16", "two.001.tokens.u16"]
    with pytest.raises(FileNotFoundError, match="no token files"):
        discover_token_files(tmp_path, ["missing"])
