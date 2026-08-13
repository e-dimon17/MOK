"""Tokenizer training determinism and consensus special ids
(A/pipeline/tokenizer_train.py)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from A.pipeline.tokenize_pack import encode_documents
from A.pipeline.tokenizer_train import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    TokenizerConfig,
    load_tokenizer_config,
    tokenizer_file_hash,
    train_tokenizer,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_corpus.txt"
CONFIGS_DIR = Path(__file__).resolve().parents[2] / "A" / "configs"


def fixture_docs() -> list[str]:
    return [d.strip() for d in re.split(r"\n\s*\n", FIXTURE.read_text("utf-8")) if d.strip()]


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    cfg = TokenizerConfig(vocab_size=400, min_frequency=2)
    return train_tokenizer(iter(fixture_docs()), out, cfg)


def test_special_ids_are_consensus_constants(trained):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained.path))
    assert tok.token_to_id("<|pad|>") == PAD_ID == 0
    assert tok.token_to_id("<|bos|>") == BOS_ID == 1
    assert tok.token_to_id("<|eos|>") == EOS_ID == 2


def test_training_is_deterministic(trained, tmp_path):
    again = train_tokenizer(
        iter(fixture_docs()), tmp_path / "tokenizer.json", TokenizerConfig(vocab_size=400)
    )
    assert again.tokenizer_hash == trained.tokenizer_hash
    assert trained.tokenizer_hash == tokenizer_file_hash(trained.path)


def test_encode_round_trip_lossless(trained):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(trained.path))
    docs = fixture_docs()[:4]
    for doc, ids in zip(docs, encode_documents(docs, trained.path), strict=True):
        assert ids  # byte-level BPE never produces empty encodings for non-empty text
        assert max(ids) < 400
        assert tok.decode(ids, skip_special_tokens=False) == doc


def test_sample_char_cap_changes_training_input(tmp_path):
    docs = fixture_docs()
    capped = train_tokenizer(
        iter(docs), tmp_path / "capped.json", TokenizerConfig(vocab_size=400, sample_chars_total=200)
    )
    full = train_tokenizer(iter(docs), tmp_path / "full.json", TokenizerConfig(vocab_size=400))
    assert capped.tokenizer_hash != full.tokenizer_hash


def test_config_validation_and_yaml_load():
    with pytest.raises(ValueError, match="vocab_size"):
        TokenizerConfig(vocab_size=100)
    with pytest.raises(ValueError, match="vocab_size"):
        TokenizerConfig(vocab_size=70_000)
    with pytest.raises(ValueError, match="distinct"):
        TokenizerConfig(bos_token="<|pad|>")
    cfg = load_tokenizer_config(CONFIGS_DIR / "tokenizer.yaml")
    assert cfg.vocab_size == 65536
    assert (cfg.pad_token, cfg.bos_token, cfg.eos_token) == ("<|pad|>", "<|bos|>", "<|eos|>")
