"""Parallel dedup must be byte-identical to the sequential reference."""

from __future__ import annotations

from itertools import groupby

import numpy as np
import pytest

from A.pipeline.dedup import DedupStats, dedup_documents
from A.pipeline.dedup_parallel import dedup_parallel, plan_units, resolve_keep_mask
from A.pipeline.download import (
    CorpusConfig,
    iter_source_documents,
    spool_documents,
)


def _cfg() -> CorpusConfig:
    return CorpusConfig.model_validate(
        {
            "name": "t",
            "dedup_order": ["small", "big"],
            "sources": [
                {"name": "small", "hf_path": "x/s", "weight": 0.5, "max_tokens": 1000},
                {"name": "big", "hf_path": "x/b", "weight": 0.5, "max_tokens": 1000},
            ],
        }
    )


def _build_spools(root):
    """small: legacy layout. big: 2-worker layout. Dups within/across + empties."""
    shared = [f"shared doc {i} with plenty of text to hash" for i in range(40)]
    small = shared[:25] + ["unique small alpha", "  ", "unique small beta", shared[0]]
    big_w0 = shared[10:30] + ["unique big gamma", "", "UNIQUE   small ALPHA".lower()]
    big_w1 = ["unique big delta"] + shared[30:] + ["unique big gamma", "\t\n"]
    spool_documents(iter(small), root, "small", part_docs=7)
    spool_documents(iter(big_w0), root / "big" / "workers", "w00", part_docs=7)
    spool_documents(iter(big_w1), root / "big" / "workers", "w01", part_docs=7)


def _sequential(root, out):
    cfg = _cfg()
    stats = DedupStats()
    ordered = [(s.name, iter_source_documents(root, s.name)) for s in cfg.dedup_sequence()]
    kept = dedup_documents(ordered, stats=stats)
    for source_name, group in groupby(kept, key=lambda kv: kv[0]):
        spool_documents((t for _, t in group), out, source_name)
    return stats


def test_parallel_equals_sequential(tmp_path):
    root = tmp_path / "spool"
    _build_spools(root)
    out_seq, out_par = tmp_path / "seq", tmp_path / "par"
    s_stats = _sequential(root, out_seq)
    p_stats = dedup_parallel(_cfg(), root, out_par, hash_workers=2, write_workers=2)

    for src in ("small", "big"):
        seq_docs = list(iter_source_documents(out_seq, src))
        par_docs = list(iter_source_documents(out_par, src))
        assert seq_docs == par_docs, f"{src}: parallel output diverged"
    assert s_stats.kept == p_stats.kept
    assert s_stats.dropped == p_stats.dropped
    assert s_stats.empty == p_stats.empty


def test_plan_units_order_and_offsets(tmp_path):
    root = tmp_path / "spool"
    _build_spools(root)
    units = plan_units(_cfg(), root)
    assert [(u.source, u.worker) for u in units] == [
        ("small", None),
        ("big", "w00"),
        ("big", "w01"),
    ]
    offs = [u.offset for u in units]
    assert offs == [0, units[0].docs, units[0].docs + units[1].docs]


def test_resolve_keep_mask_semantics():
    digests = np.array([5, 7, 5, 9, 7, 5, 0], dtype=np.uint64)
    empties = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.uint8)
    keep = resolve_keep_mask(digests, empties)
    assert keep.tolist() == [True, True, False, True, False, False, False]


def test_inconsistent_spool_refused(tmp_path):
    import json

    root = tmp_path / "spool"
    _build_spools(root)
    sp = root / "small" / "state.json"
    st = json.loads(sp.read_text())
    st["docs"] += 5
    sp.write_text(json.dumps(st))
    with pytest.raises(RuntimeError, match="spool is inconsistent"):
        dedup_parallel(_cfg(), root, tmp_path / "out", hash_workers=2, write_workers=2)
