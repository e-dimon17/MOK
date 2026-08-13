"""Packing determinism and boundary math (A/pipeline/tokenize_pack.py)."""

from __future__ import annotations

import numpy as np
import pytest

from A.pipeline.tokenize_pack import (
    MAX_TOKEN_ID,
    chunk_token_arrays,
    chunk_token_stream,
    iter_token_file_arrays,
    pack_documents,
    write_token_stream,
)


class ToyTokenizer:
    """Char-code tokenizer: deterministic, no external deps."""

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), 1000) for c in text]


TOY_DOCS = ["hello world", "abc", "x" * 40, "the quick brown fox", ""]


def toy_token_iters() -> list[list[int]]:
    tok = ToyTokenizer()
    return [tok.encode(d) for d in TOY_DOCS]


def test_pack_eos_join_and_split():
    seqs = list(pack_documents([[1, 2, 3], [4, 5]], seq_len=3, eos_id=0))
    assert [s.tolist() for s in seqs] == [[1, 2, 3], [0, 4, 5]]  # trailing [0] dropped
    assert all(s.dtype == np.uint16 for s in seqs)


def test_pack_boundary_math():
    docs = toy_token_iters()
    seq_len = 16
    total_stream = sum(len(d) for d in docs) + len(docs)  # one EOS per doc
    seqs = list(pack_documents(docs, seq_len=seq_len, eos_id=2))
    assert len(seqs) == total_stream // seq_len
    assert all(s.shape == (seq_len,) for s in seqs)


def test_pack_deterministic():
    a = list(pack_documents(toy_token_iters(), seq_len=8, eos_id=2))
    b = list(pack_documents(toy_token_iters(), seq_len=8, eos_id=2))
    assert len(a) == len(b) > 0
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


def test_pack_long_doc_spans_boundaries():
    long_doc = list(range(100))
    seqs = list(pack_documents([long_doc], seq_len=32, eos_id=2))
    assert len(seqs) == (100 + 1) // 32
    joined = np.concatenate(seqs)
    assert joined.tolist() == (long_doc + [2])[: len(joined)]


def test_chunk_rejects_out_of_range_tokens():
    with pytest.raises(ValueError, match="uint16"):
        list(chunk_token_stream([1, 2, MAX_TOKEN_ID + 1], seq_len=3))
    with pytest.raises(ValueError, match="uint16"):
        list(chunk_token_stream([-1, 2, 3], seq_len=3))


def test_pack_rejects_bad_eos():
    with pytest.raises(ValueError, match="eos_id"):
        pack_documents([[1]], seq_len=4, eos_id=MAX_TOKEN_ID + 1)


def test_chunk_token_arrays_matches_stream_chunker():
    rng = np.random.default_rng(7)
    flat = rng.integers(0, 65536, size=1000, dtype=np.uint16)
    blocks = [flat[:130], flat[130:131], flat[131:700], flat[700:]]
    via_arrays = list(chunk_token_arrays(blocks, seq_len=64))
    via_stream = list(chunk_token_stream((int(t) for t in flat), seq_len=64))
    assert len(via_arrays) == len(via_stream) == 1000 // 64
    for x, y in zip(via_arrays, via_stream, strict=True):
        assert np.array_equal(x, y)


def test_chunk_token_arrays_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="uint16"):
        list(chunk_token_arrays([np.zeros(4, dtype=np.int32)], seq_len=2))


def test_token_stream_file_round_trip(tmp_path):
    docs = toy_token_iters()
    path = tmp_path / "src.tokens.u16"
    total = write_token_stream(docs, path, eos_id=2, flush_tokens=8)  # force multiple flushes
    expected: list[int] = []
    for d in docs:
        expected.extend(d)
        expected.append(2)
    assert total == len(expected)
    assert path.stat().st_size == 2 * total
    read_back = np.concatenate(list(iter_token_file_arrays([path], block_tokens=16)))
    assert read_back.tolist() == expected
    # chunking the file equals packing the documents directly
    from_file = list(chunk_token_arrays(iter_token_file_arrays([path]), seq_len=8))
    direct = list(pack_documents(toy_token_iters(), seq_len=8, eos_id=2))
    for x, y in zip(from_file, direct, strict=True):
        assert np.array_equal(x, y)
