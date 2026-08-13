"""SFT dataset preparation: loaders, chat rendering, decontamination, packing.

Dataset spec (playbook step F): Tulu-3 SFT mixture backbone + OpenHermes-2.5 +
reasoning traces, all normalized to `{"messages": [{"role", "content"}, ...]}`,
rendered in the ChatML-style template below with loss masked to assistant
turns, 13-gram decontaminated against the eval suites, packed to seq 16384.

`datasets` (HuggingFace) is imported lazily inside the loaders only; every
loader takes an injectable `rows` iterable so tests and offline runs never
touch the Hub.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import torch

# --------------------------------------------------------------------------- #
# Chat template (ChatML-style)
# --------------------------------------------------------------------------- #

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

# Jinja template written into converted model dirs (tokenizer_config.json +
# chat_template.jinja). `render_chat` below produces EXACTLY this rendering,
# segment by segment, so training-time masking and inference-time prompting
# agree byte-for-byte.
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '" + IM_START + "' + message['role'] + '\\n' + message['content'] + '" + IM_END + "' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '" + IM_START + "assistant\\n' }}{% endif %}"
)

VALID_ROLES = ("system", "user", "assistant", "tool")

LABEL_IGNORE = -100


class TokenizerLike(Protocol):
    """Anything that can encode plain text to ids (HF tokenizer or raw
    `tokenizers.Tokenizer`)."""

    def encode(self, text: str, *args: Any, **kwargs: Any) -> Any: ...


def _encode(tokenizer: TokenizerLike, text: str) -> list[int]:
    """Encode WITHOUT special tokens; supports both HF and raw-tokenizers APIs."""
    try:
        ids = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # raw tokenizers.Tokenizer: encode(text) -> Encoding
        ids = tokenizer.encode(text)
    if hasattr(ids, "ids"):
        ids = ids.ids
    return list(ids)


def render_chat(
    messages: Sequence[Mapping[str, str]], tokenizer: TokenizerLike
) -> tuple[list[int], list[int]]:
    """Render a conversation to (input_ids, labels), loss-masked to assistant turns.

    Per message the template contributes two segments:
      header  = "<|im_start|>{role}\\n"           -> labels always -100
      body    = "{content}<|im_end|>\\n"          -> labels = ids iff role == assistant
    A BOS token (if the tokenizer has one) is prepended with label -100.
    Segments are tokenized independently, so multi-token ChatML markers need no
    dedicated vocab entries (the 65k step-A tokenizer has none to spare).
    """
    input_ids: list[int] = []
    labels: list[int] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        input_ids.append(int(bos))
        labels.append(LABEL_IGNORE)
    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role {role!r}; expected one of {VALID_ROLES}")
        if not isinstance(content, str):
            raise ValueError(f"message content must be str, got {type(content).__name__}")
        header_ids = _encode(tokenizer, f"{IM_START}{role}\n")
        body_ids = _encode(tokenizer, f"{content}{IM_END}\n")
        input_ids.extend(header_ids)
        labels.extend([LABEL_IGNORE] * len(header_ids))
        input_ids.extend(body_ids)
        if role == "assistant":
            labels.extend(body_ids)
        else:
            labels.extend([LABEL_IGNORE] * len(body_ids))
    return input_ids, labels


# --------------------------------------------------------------------------- #
# Loaders — all normalized to {"messages": [{"role", "content"}, ...]}
# --------------------------------------------------------------------------- #

TULU3_DATASET = "allenai/tulu-3-sft-mixture"
OPENHERMES_DATASET = "teknium/OpenHermes-2.5"

_SHAREGPT_ROLES = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}


def _clean_messages(raw: Iterable[Mapping[str, Any]]) -> list[dict[str, str]] | None:
    """Validate/coerce a message list; None if the example is unusable."""
    out: list[dict[str, str]] = []
    for m in raw:
        role = m.get("role")
        content = m.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            return None
        out.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


def tulu3(*, rows: Iterable[Mapping[str, Any]] | None = None, split: str = "train") -> Iterator[dict]:
    """Tulu-3 SFT mixture, already in `messages` form; validated + normalized."""
    if rows is None:
        from datasets import load_dataset  # noqa: PLC0415 — lazy, [post] extra

        rows = load_dataset(TULU3_DATASET, split=split)
    for row in rows:
        msgs = _clean_messages(row.get("messages") or [])
        if msgs:
            yield {"messages": msgs}


def openhermes(
    *, rows: Iterable[Mapping[str, Any]] | None = None, split: str = "train"
) -> Iterator[dict]:
    """OpenHermes-2.5 (ShareGPT `conversations` with from/value keys) -> messages."""
    if rows is None:
        from datasets import load_dataset  # noqa: PLC0415

        rows = load_dataset(OPENHERMES_DATASET, split=split)
    for row in rows:
        conv = row.get("conversations") or []
        raw = [
            {"role": _SHAREGPT_ROLES.get(turn.get("from", ""), ""), "content": turn.get("value")}
            for turn in conv
        ]
        msgs = _clean_messages(raw)
        if msgs:
            yield {"messages": msgs}


def reasoning_traces(
    path: str | Path, *, rows: Iterable[Mapping[str, Any]] | None = None
) -> Iterator[dict]:
    """Local JSONL reasoning traces (OpenThoughts-class). Accepted shapes per line:
    {"messages": [...]}, or {"prompt"/"question"/"instruction": str,
    "response"/"answer"/"completion": str} (+ optional "system")."""
    if rows is None:
        rows = _iter_jsonl(Path(path))
    for row in rows:
        if "messages" in row:
            msgs = _clean_messages(row["messages"] or [])
        else:
            prompt = next(
                (row[k] for k in ("prompt", "question", "instruction") if isinstance(row.get(k), str)),
                None,
            )
            response = next(
                (row[k] for k in ("response", "answer", "completion") if isinstance(row.get(k), str)),
                None,
            )
            if prompt is None or response is None:
                continue
            raw: list[dict[str, Any]] = []
            if isinstance(row.get("system"), str) and row["system"].strip():
                raw.append({"role": "system", "content": row["system"]})
            raw.append({"role": "user", "content": prompt})
            raw.append({"role": "assistant", "content": response})
            msgs = _clean_messages(raw)
        if msgs:
            yield {"messages": msgs}


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------- #
# Decontamination — 13-gram overlap against eval suites
# --------------------------------------------------------------------------- #


def text_ngrams(text: str, n: int = 13) -> set[str]:
    """Lowercased whitespace-token n-grams as space-joined strings."""
    tokens = text.lower().split()
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def build_eval_ngrams(texts: Iterable[str], n: int = 13) -> set[str]:
    """N-gram bank of the held-out eval suites (feed every prompt AND target)."""
    bank: set[str] = set()
    for text in texts:
        bank |= text_ngrams(text, n)
    return bank


def decontaminate(
    examples: Iterable[Mapping[str, Any]], eval_ngrams: set[str], n: int = 13
) -> list[dict]:
    """Drop any example sharing >= 1 n-gram (over any message content) with the
    eval bank. Returns the kept examples in input order."""
    kept: list[dict] = []
    for ex in examples:
        contaminated = False
        for message in ex.get("messages", []):
            if text_ngrams(message.get("content", ""), n) & eval_ngrams:
                contaminated = True
                break
        if not contaminated:
            kept.append(dict(ex))
    return kept


# --------------------------------------------------------------------------- #
# Packing collator (seq 16384)
# --------------------------------------------------------------------------- #


def pack_examples(
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    seq_len: int = 16384,
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Concatenate (input_ids, labels) pairs into a token stream and cut into
    `seq_len` rows (examples may span row boundaries — standard SFT packing).
    The final row is right-padded: pad_id / label -100 / attention_mask 0."""
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    stream_ids: list[int] = []
    stream_labels: list[int] = []
    for ids, labels in pairs:
        if len(ids) != len(labels):
            raise ValueError(f"input_ids ({len(ids)}) and labels ({len(labels)}) length mismatch")
        stream_ids.extend(int(i) for i in ids)
        stream_labels.extend(int(v) for v in labels)
    if not stream_ids:
        raise ValueError("pack_examples received no tokens")
    total = len(stream_ids)
    rows = (total + seq_len - 1) // seq_len
    pad = rows * seq_len - total
    stream_ids.extend([pad_id] * pad)
    stream_labels.extend([LABEL_IGNORE] * pad)
    mask = [1] * total + [0] * pad
    return {
        "input_ids": torch.tensor(stream_ids, dtype=torch.int64).view(rows, seq_len),
        "labels": torch.tensor(stream_labels, dtype=torch.int64).view(rows, seq_len),
        "attention_mask": torch.tensor(mask, dtype=torch.int64).view(rows, seq_len),
    }


class SFTPackCollator:
    """Collator: rendered examples -> packed fixed-length batches (default 16384).

    Accepts features as mappings with "input_ids"/"labels" or as (ids, labels)
    tuples — i.e. `render_chat` outputs, directly or wrapped by `datasets`.
    """

    def __init__(self, seq_len: int = 16384, pad_id: int = 0) -> None:
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __call__(
        self, features: Sequence[Mapping[str, Sequence[int]] | tuple[Sequence[int], Sequence[int]]]
    ) -> dict[str, torch.Tensor]:
        pairs: list[tuple[Sequence[int], Sequence[int]]] = []
        for feat in features:
            if isinstance(feat, Mapping):
                pairs.append((feat["input_ids"], feat["labels"]))
            else:
                ids, labels = feat
                pairs.append((ids, labels))
        return pack_examples(pairs, seq_len=self.seq_len, pad_id=self.pad_id)


def pack_for_sft(
    pairs: Sequence[tuple[Sequence[int], Sequence[int]]],
    seq_len: int = 16384,
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Functional alias of the collator (spec name)."""
    return pack_examples(pairs, seq_len=seq_len, pad_id=pad_id)


__all__ = [
    "CHAT_TEMPLATE",
    "IM_END",
    "IM_START",
    "LABEL_IGNORE",
    "SFTPackCollator",
    "build_eval_ngrams",
    "decontaminate",
    "openhermes",
    "pack_examples",
    "pack_for_sft",
    "reasoning_traces",
    "render_chat",
    "text_ngrams",
    "tulu3",
]
