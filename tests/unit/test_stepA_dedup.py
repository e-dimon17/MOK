"""Cross-source exact-hash dedup behavior (A/pipeline/dedup.py)."""

from __future__ import annotations

from A.pipeline.dedup import DedupStats, dedup_documents, doc_digest, normalize

BASE = (
    "Error feedback keeps the part of the gradient that compression discarded and adds it "
    "back before the next compression step, so every coordinate is eventually transmitted "
    "and the compressed optimizer provably tracks its dense counterpart over training."
)
NEAR_DUP = BASE.replace("provably", "demonstrably")
WHITESPACE_CASE_VARIANT = "  " + BASE.upper().replace(" ", "   ") + "\n\n"
UNIQUE_A = (
    "The lighthouse began its patient argument with the dark while fishing boats knocked "
    "against the pilings with a sound like slow applause all through the evening."
)
UNIQUE_B = (
    "Integration by parts reduces the polynomial degree at each application, which is why "
    "the antiderivative of x squared times e to the x terminates after two rounds."
)


def _run(sources, workers=0):
    stats = DedupStats()
    out = list(dedup_documents(sources, stats=stats, workers=workers))
    return out, stats


def test_exact_duplicate_across_sources_dropped():
    out, stats = _run([("small", [BASE, UNIQUE_A]), ("big", [BASE, UNIQUE_B])])
    assert out == [("small", BASE), ("small", UNIQUE_A), ("big", UNIQUE_B)]
    assert stats.kept == {"small": 2, "big": 1}
    assert stats.dropped == {"big": 1}


def test_normalized_variant_dropped_priority_order_wins():
    # Whitespace/case differences are the same document; earlier source wins.
    out, stats = _run([("small", [BASE]), ("big", [WHITESPACE_CASE_VARIANT, UNIQUE_B])])
    assert out == [("small", BASE), ("big", UNIQUE_B)]
    assert stats.dropped == {"big": 1}
    out_rev, _ = _run([("big", [WHITESPACE_CASE_VARIANT, UNIQUE_B]), ("small", [BASE])])
    assert out_rev == [("big", WHITESPACE_CASE_VARIANT), ("big", UNIQUE_B)]


def test_near_duplicate_now_kept_by_policy():
    # Exact-hash policy: a one-word edit is a different document. Deliberate
    # trade against MinHash at 1.9B-doc scale (see module docstring).
    out, stats = _run([("a", [BASE]), ("b", [NEAR_DUP])])
    assert out == [("a", BASE), ("b", NEAR_DUP)]
    assert stats.total_dropped == 0


def test_within_source_duplicate_dropped():
    out, _ = _run([("only", [BASE, BASE, UNIQUE_A])])
    assert out == [("only", BASE), ("only", UNIQUE_A)]


def test_distinct_documents_all_kept():
    docs = [BASE, UNIQUE_A, UNIQUE_B]
    out, stats = _run([("s", docs)])
    assert [t for _, t in out] == docs and stats.total_dropped == 0


def test_whitespace_only_documents_dropped():
    out, stats = _run([("s", ["   \n\t  ", "", UNIQUE_A])])
    assert out == [("s", UNIQUE_A)]
    assert stats.empty == {"s": 2}


def test_deterministic_over_repeat_runs():
    sources = lambda: [("small", [BASE, UNIQUE_A]), ("big", [NEAR_DUP, UNIQUE_B, BASE])]  # noqa: E731
    assert list(dedup_documents(sources())) == list(dedup_documents(sources()))


def test_workers_do_not_change_output():
    docs_a = [f"document alpha number {i} with shared boilerplate text" for i in range(300)]
    docs_b = docs_a[:50] + [f"document beta number {i} entirely different" for i in range(300)]
    serial, s0 = _run([("a", docs_a), ("b", docs_b)], workers=0)
    parallel, s1 = _run([("a", docs_a), ("b", docs_b)], workers=2)
    assert serial == parallel
    assert s0.kept == s1.kept and s0.dropped == s1.dropped and s0.empty == s1.empty


def test_digest_and_normalize_properties():
    assert doc_digest("   \n ") is None
    assert doc_digest(BASE) == doc_digest(WHITESPACE_CASE_VARIANT)
    assert doc_digest(BASE) != doc_digest(NEAR_DUP)
    assert normalize("  A\tB\nC  ") == "a b c"
    assert isinstance(doc_digest("mok"), int)  # stable xxh3-64; pinned below
    import xxhash

    assert doc_digest("MOK  subnet") == xxhash.xxh3_64_intdigest(b"mok subnet")
