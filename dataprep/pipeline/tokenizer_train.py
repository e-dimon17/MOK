"""65,536-vocab ByteLevel-BPE tokenizer training.

The tokenizer is trained once on a weighted corpus sample and then frozen
forever: the blake2b-256 hash of its `tokenizer.json` goes into every
`DatasetManifestRef` and from there into the on-chain run manifest. Special
token ids are consensus constants: `<|pad|>`=0, `<|bos|>`=1, `<|eos|>`=2.
Training is deterministic given the same input sample in the same order.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

from pydantic import model_validator

from mok_core.config.loader import load_yaml
from mok_core.config.schemas import FrozenModel

PAD_TOKEN, PAD_ID = "<|pad|>", 0
BOS_TOKEN, BOS_ID = "<|bos|>", 1
EOS_TOKEN, EOS_ID = "<|eos|>", 2

_BYTE_ALPHABET_SIZE = 256
_DIGEST_SIZE = 32


class TokenizerConfig(FrozenModel):
    """Knobs for `train_tokenizer`, loadable from dataprep/configs/tokenizer.yaml."""

    vocab_size: int = 65536
    min_frequency: int = 2
    sample_chars_total: int = 400_000_000_000  # ~100B tokens at ~4 chars/token
    pad_token: str = PAD_TOKEN
    bos_token: str = BOS_TOKEN
    eos_token: str = EOS_TOKEN

    @model_validator(mode="after")
    def _check(self) -> TokenizerConfig:
        floor = _BYTE_ALPHABET_SIZE + 3  # byte alphabet + the three specials
        if not floor <= self.vocab_size <= 65536:
            raise ValueError(f"vocab_size must be in [{floor}, 65536] (uint16 ids), got {self.vocab_size}")
        if self.min_frequency < 1:
            raise ValueError(f"min_frequency must be >= 1, got {self.min_frequency}")
        if self.sample_chars_total <= 0:
            raise ValueError("sample_chars_total must be positive")
        if len({self.pad_token, self.bos_token, self.eos_token}) != 3:
            raise ValueError("pad/bos/eos tokens must be distinct")
        return self


def load_tokenizer_config(path: str | PathLike[str]) -> TokenizerConfig:
    return TokenizerConfig.model_validate(load_yaml(path))


def tokenizer_file_hash(path: str | PathLike[str]) -> str:
    """Hex blake2b-256 of the tokenizer.json bytes — the manifest `tokenizer_hash`."""
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class TrainedTokenizer:
    path: Path
    tokenizer_hash: str


def _capped(texts: Iterable[str], max_chars: int) -> Iterator[str]:
    """Pass texts through until the cumulative character budget is spent."""
    seen = 0
    for text in texts:
        if seen >= max_chars:
            return
        yield text
        seen += len(text)


def train_tokenizer(
    texts: Iterable[str],
    out_path: str | PathLike[str],
    cfg: TokenizerConfig | None = None,
) -> TrainedTokenizer:
    """Train a ByteLevel BPE tokenizer over `texts` and save `tokenizer.json`.

    Special tokens are registered first so their ids are exactly 0/1/2; the
    result is verified before saving. Returns the saved path plus its hash.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    cfg = cfg if cfg is not None else TokenizerConfig()
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        min_frequency=cfg.min_frequency,
        special_tokens=[cfg.pad_token, cfg.bos_token, cfg.eos_token],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_capped(texts, cfg.sample_chars_total), trainer=trainer)

    for token, expected in ((cfg.pad_token, PAD_ID), (cfg.bos_token, BOS_ID), (cfg.eos_token, EOS_ID)):
        got = tok.token_to_id(token)
        if got != expected:
            raise RuntimeError(f"special token {token!r} trained to id {got}, expected {expected}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    return TrainedTokenizer(path=out, tokenizer_hash=tokenizer_file_hash(out))
