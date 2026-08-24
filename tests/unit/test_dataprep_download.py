"""Spool write/read/resume and corpus config loading (dataprep/pipeline/download.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataprep.pipeline.download import (
    CorpusConfig,
    SourceSpec,
    download_corpus,
    download_source,
    iter_source_documents,
    load_corpus_config,
    load_spool_state,
    spool_documents,
)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "dataprep" / "configs"

DOCS = [f"document number {i} — " + "lorem ipsum " * (i + 1) for i in range(10)]


def make_cfg(**source_overrides) -> CorpusConfig:
    base = {
        "name": "src_a",
        "hf_path": "dummy/a",
        "weight": 1.0,
        "max_tokens": 1_000_000,
    }
    return CorpusConfig(name="mini", sources=(SourceSpec(**{**base, **source_overrides}),))


def test_spool_round_trip_preserves_order_and_unicode(tmp_path):
    docs = [*DOCS, "unicode: café — 中文\nsecond line"]
    state = spool_documents(docs, tmp_path, "src_a", part_docs=3)
    assert state.complete
    assert state.docs == len(docs)
    assert state.chars == sum(len(d) for d in docs)
    assert len(state.parts) == 4  # ceil(11/3)
    assert list(iter_source_documents(tmp_path, "src_a")) == docs


def test_spool_char_budget_stops_early(tmp_path):
    docs = ["x" * 100 for _ in range(10)]
    state = spool_documents(docs, tmp_path, "src_a", char_budget=250, part_docs=100)
    assert state.complete
    assert state.docs == 3  # budget crossed after the third document
    assert state.chars == 300
    assert list(iter_source_documents(tmp_path, "src_a")) == docs[:3]


def test_spool_resume_skips_committed_docs(tmp_path):
    spool_documents(DOCS[:5], tmp_path, "src_a", part_docs=5)
    # simulate an interrupted session: mark incomplete, then resume with the full stream
    sdir = tmp_path / "src_a"
    state = load_spool_state(tmp_path, "src_a")
    doc = state.model_copy(update={"complete": False}).model_dump(mode="json")
    (sdir / "state.json").write_text(json.dumps(doc))
    state = spool_documents(DOCS, tmp_path, "src_a", part_docs=3)
    assert state.complete
    assert state.docs == len(DOCS)
    assert list(iter_source_documents(tmp_path, "src_a")) == DOCS


def test_spool_complete_state_is_a_no_op(tmp_path):
    first = spool_documents(DOCS, tmp_path, "src_a", part_docs=4)
    sentinel = iter(())  # would raise if consumed past completion check
    again = spool_documents(sentinel, tmp_path, "src_a", part_docs=4)
    assert again == first


def test_stale_tmp_parts_are_cleaned(tmp_path):
    sdir = tmp_path / "src_a"
    sdir.mkdir(parents=True)
    (sdir / "part-00000.jsonl.zst.tmp").write_bytes(b"garbage")
    spool_documents(DOCS[:2], tmp_path, "src_a")
    assert not list(sdir.glob("*.tmp"))
    assert list(iter_source_documents(tmp_path, "src_a")) == DOCS[:2]


def test_download_source_uses_injected_iterator_and_budget(tmp_path):
    cfg = CorpusConfig(
        name="mini",
        chars_per_token=2.0,
        sources=(SourceSpec(name="src_a", hf_path="dummy/a", weight=1.0, max_tokens=100),),
    )
    docs = ["a" * 50 for _ in range(10)]
    state = download_source(cfg, cfg.sources[0], tmp_path, doc_iter=docs)
    assert state.complete
    assert state.chars >= 200  # budget = 100 tokens * 2 chars/token
    assert state.docs == 4


def test_download_corpus_only_filter(tmp_path):
    cfg = CorpusConfig(
        name="mini",
        sources=(
            SourceSpec(name="src_a", hf_path="dummy/a", weight=1.0, max_tokens=10),
            SourceSpec(name="src_b", hf_path="dummy/b", weight=1.0, max_tokens=10),
        ),
    )
    with pytest.raises(KeyError):
        download_corpus(cfg, tmp_path, only="nope")


def test_source_spec_validation():
    with pytest.raises(ValueError, match="weight"):
        SourceSpec(name="a", hf_path="x", weight=0, max_tokens=1)
    with pytest.raises(ValueError, match="name"):
        SourceSpec(name="Bad Name", hf_path="x", weight=1.0, max_tokens=1)
    with pytest.raises(ValueError, match="score"):
        SourceSpec(name="a", hf_path="x", weight=1.0, max_tokens=1, min_score=2.0)


def test_corpus_config_validation():
    src = {"hf_path": "x", "weight": 1.0, "max_tokens": 1}
    with pytest.raises(ValueError, match="permutation"):
        CorpusConfig(
            name="c",
            dedup_order=("a",),
            sources=(SourceSpec(name="a", **src), SourceSpec(name="b", **src)),
        )
    with pytest.raises(ValueError, match="unique"):
        CorpusConfig(name="c", sources=(SourceSpec(name="a", **src), SourceSpec(name="a", **src)))


def test_shipped_corpus_configs_load():
    for fname, expected_name in (("corpus_bulk.yaml", "bulk"), ("corpus_anneal.yaml", "anneal")):
        cfg = load_corpus_config(CONFIGS_DIR / fname)
        assert cfg.name == expected_name
        assert cfg.seq_len == 4096
        assert len(cfg.dedup_order) == len(cfg.sources)

    # bulk: dedup order runs smallest sources first
    bulk = load_corpus_config(CONFIGS_DIR / "corpus_bulk.yaml")
    budgets = [bulk.source(n).max_tokens for n in bulk.dedup_order]
    assert budgets == sorted(budgets)

    # anneal: subsets dedup BEFORE their supersets so premium tiers keep their
    # docs and each superset nets only the remainder (see corpus_anneal.yaml).
    anneal = load_corpus_config(CONFIGS_DIR / "corpus_anneal.yaml")
    order = {name: i for i, name in enumerate(anneal.dedup_order)}
    assert order["finemath_4plus"] < order["finemath_3plus"]
    assert order["infiwebmath_4plus"] < order["infiwebmath_3plus"]
    assert order["fineweb_edu_top"] < order["fineweb_edu_high"] < order["fineweb_edu_mid"]


def test_shipped_bulk_weights_match_playbook():
    cfg = load_corpus_config(CONFIGS_DIR / "corpus_bulk.yaml")
    weights = {s.name: s.weight for s in cfg.sources}
    assert weights == {
        "fineweb_edu": 0.34,
        "dclm_baseline": 0.14,
        "starcoder_code": 0.06,
        "finemath": 0.03,
        "infiwebmath": 0.03,
        "fineweb_edu_score2": 0.40,
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-9
