"""Parallel (--workers) download layout: determinism, resume, guards, merge."""

from __future__ import annotations

import json
import sys

import pytest

from A.pipeline.download import (
    WORKERS_DIRNAME,
    WORKERS_META_FILENAME,
    SourceSpec,
    SpoolLayoutError,
    download_source,
    iter_source_documents,
    load_spool_state,
    spool_documents,
    worker_names,
)


class _FakeCorpus:
    """Just enough of CorpusConfig for download_source: a char budget."""

    def __init__(self, budget: int | None) -> None:
        self._budget = budget

    def char_budget(self, spec: SourceSpec) -> int | None:
        return self._budget


def _spec(name: str = "src_a") -> SourceSpec:
    return SourceSpec(name=name, hf_path="fake/ds", weight=1.0, max_tokens=1_000_000)


def _docs(prefix: str, n: int) -> list[str]:
    return [f"{prefix} document {i:04d}" for i in range(n)]


def test_worker_layout_and_iteration_order(tmp_path):
    w0, w1 = _docs("alpha", 25), _docs("beta", 25)
    state = download_source(
        _FakeCorpus(None), _spec(), tmp_path, part_docs=10, worker_doc_iters=[iter(w0), iter(w1)]
    )
    assert state.complete and state.docs == 50 and state.parts == ()
    assert worker_names(tmp_path, "src_a") == ["w00", "w01"]
    meta = json.loads((tmp_path / "src_a" / WORKERS_DIRNAME / WORKERS_META_FILENAME).read_text())
    assert meta == {"workers": 2}
    # Iteration is w00's stream fully, then w01's — deterministic merge order.
    assert list(iter_source_documents(tmp_path, "src_a")) == w0 + w1


def test_worker_resume_skips_committed(tmp_path):
    w0, w1 = _docs("alpha", 30), _docs("beta", 30)

    def _dying_stream(docs: list[str], die_after: int):
        # A killed session: parts up to `die_after` are committed, then the
        # process vanishes mid-stream — state.json stays complete=False.
        yield from docs[:die_after]
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        download_source(
            _FakeCorpus(None),
            _spec(),
            tmp_path,
            part_docs=10,
            worker_doc_iters=[_dying_stream(w0, 20), iter(w1)],
        )
    assert load_spool_state(tmp_path / "src_a" / WORKERS_DIRNAME, "w00").docs == 20
    state = download_source(
        _FakeCorpus(None), _spec(), tmp_path, part_docs=10, worker_doc_iters=[iter(w0), iter(w1)]
    )
    assert state.docs == 60 and state.complete
    assert list(iter_source_documents(tmp_path, "src_a")) == w0 + w1  # no duplicates


def test_worker_count_mismatch_rejected(tmp_path):
    download_source(
        _FakeCorpus(None), _spec(), tmp_path, part_docs=10, worker_doc_iters=[iter(_docs("a", 5))]
    )
    with pytest.raises(SpoolLayoutError, match="--workers 1"):
        download_source(
            _FakeCorpus(None),
            _spec(),
            tmp_path,
            part_docs=10,
            worker_doc_iters=[iter([]), iter([])],
        )


def test_legacy_spool_guard_and_discard(tmp_path):
    spool_documents(iter(_docs("legacy", 12)), tmp_path, "src_a", part_docs=10)
    with pytest.raises(SpoolLayoutError, match="single-stream spool"):
        download_source(
            _FakeCorpus(None), _spec(), tmp_path, worker_doc_iters=[iter([]), iter([])]
        )
    w0, w1 = _docs("alpha", 8), _docs("beta", 8)
    download_source(
        _FakeCorpus(None),
        _spec(),
        tmp_path,
        discard_legacy=True,
        part_docs=10,
        worker_doc_iters=[iter(w0), iter(w1)],
    )
    assert load_spool_state(tmp_path, "src_a").docs == 0  # legacy gone
    assert list(iter_source_documents(tmp_path, "src_a")) == w0 + w1


def test_budget_split_across_workers(tmp_path):
    # 2 workers, budget 400 chars -> 200/worker; docs are 20 chars each -> 10 docs/worker.
    doc = "x" * 20
    state = download_source(
        _FakeCorpus(400),
        _spec(),
        tmp_path,
        part_docs=4,
        worker_doc_iters=[iter([doc] * 100), iter([doc] * 100)],
    )
    assert state.docs == 20 and state.chars == 400 and state.complete


def test_legacy_layout_unaffected_without_workers(tmp_path):
    docs = _docs("solo", 15)
    state = download_source(
        _FakeCorpus(None), _spec(), tmp_path, doc_iter=iter(docs), part_docs=10
    )
    assert state.complete and state.parts != ()
    assert worker_names(tmp_path, "src_a") == []
    assert list(iter_source_documents(tmp_path, "src_a")) == docs


def test_dead_stream_raises_config_error(monkeypatch, tmp_path):
    """An IDs-only dataset (no text column) must fail loudly, not spin forever."""
    import sys
    import types

    import A.pipeline.download as dl

    fake = types.ModuleType("datasets")

    def load_dataset(*a, **k):
        class _DS:
            def shard(self, num_shards, index):
                return self

            def __iter__(self):
                return ({"blob_id": str(i)} for i in range(10_000_000))

        return _DS()

    fake.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(dl, "_DEAD_STREAM_ROWS", 5_000)
    with pytest.raises(RuntimeError, match="without a single usable 'content'"):
        list(dl._hf_documents(_spec_ids_only()))


def _spec_ids_only() -> SourceSpec:
    return SourceSpec(
        name="ids_only",
        hf_path="fake/ids-only",
        text_column="content",
        weight=1.0,
        max_tokens=1_000,
    )


def test_hf_columns_projection_kwargs(monkeypatch):
    """hf_columns must reach load_dataset as columns= + string Features."""
    import sys
    import types

    import A.pipeline.download as dl

    captured: dict = {}
    fake = types.ModuleType("datasets")

    class _Features(dict):
        pass

    class _Value:
        def __init__(self, dtype):
            self.dtype = dtype

        def __eq__(self, other):
            return isinstance(other, _Value) and other.dtype == self.dtype

    def load_dataset(path, **kwargs):
        captured.update(kwargs)

        class _DS:
            def __iter__(self):
                return iter([{"content": "x = 1"}])

        return _DS()

    fake.load_dataset = load_dataset
    fake.Features = _Features
    fake.Value = _Value
    monkeypatch.setitem(sys.modules, "datasets", fake)

    spec = SourceSpec(
        name="code",
        hf_path="fake/hetero",
        text_column="content",
        hf_columns=("content",),
        weight=1.0,
        max_tokens=1_000,
    )
    assert list(dl._hf_documents(spec)) == ["x = 1"]
    assert captured["columns"] == ["content"]
    assert captured["features"] == _Features({"content": _Value("string")})


def test_hf_columns_validators():
    with pytest.raises(ValueError, match="must include text_column"):
        SourceSpec(
            name="bad", hf_path="x/y", text_column="content",
            hf_columns=("id",), weight=1.0, max_tokens=10,
        )
    with pytest.raises(ValueError, match="incompatible with min_score"):
        SourceSpec(
            name="bad2", hf_path="x/y", text_column="content", hf_columns=("content", "score"),
            score_column="score", min_score=0.5, weight=1.0, max_tokens=10,
        )


def test_datasets_projection_survives_heterogeneous_schemas(tmp_path):
    """Pins the upstream behavior our starcoderdata fix relies on: projecting
    columns via load_dataset(columns=..., features=...) must stream parquet
    files whose schemas differ in extra columns. If a datasets upgrade breaks
    this, this test fails before production does."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    try:
        from datasets import Features, Value, load_dataset

        pq.write_table(pa.table({"id": ["a"], "content": ["print(1)"]}), str(tmp_path / "a.parquet"))
        pq.write_table(
            pa.table({"id": ["b"], "content": ["int main(){}"], "hexsha": ["ff"], "size": [9]}),
            str(tmp_path / "b.parquet"),
        )
        ds = load_dataset(
            "parquet",
            data_files=[str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")],
            split="train",
            streaming=True,
            columns=["content"],
            features=Features({"content": Value("string")}),
        )
        assert [r["content"] for r in ds] == ["print(1)", "int main(){}"]
    finally:
        # Unload the real `datasets` so later "heavy stack not imported" asserts
        # (F/G lazy-import tests) see a clean sys.modules.
        for mod in [m for m in sys.modules if m == "datasets" or m.startswith("datasets.")]:
            sys.modules.pop(mod, None)


def test_resume_skip_logs_progress(monkeypatch, tmp_path, caplog):
    """The resume fast-forward must emit periodic progress (not look like a hang)."""
    import logging

    import A.pipeline.download as dl

    docs = _docs("p", 50)

    def _dying():
        yield from docs[:30]
        raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        spool_documents(_dying(), tmp_path, "src_a", part_docs=10)
    monkeypatch.setattr(dl, "_SKIP_LOG_EVERY", 10)
    with caplog.at_level(logging.INFO, logger="mok.stepA.download"):
        spool_documents(iter(docs), tmp_path, "src_a", part_docs=10)
    progress = [r for r in caplog.records if r.getMessage() == "re-skip progress"]
    assert len(progress) == 3  # at 10, 20, 30 of 30 skipped
    fields = progress[-1].fields  # type: ignore[attr-defined]
    assert fields["skipped"] == 30 and fields["skip_pct"] == 100.0
