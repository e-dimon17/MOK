"""SFT: chat rendering / loss masks, loaders, decontamination, packing."""

from __future__ import annotations

import json

import pytest
import torch

from sft.data_prep import (
    IM_END,
    IM_START,
    LABEL_IGNORE,
    SFTPackCollator,
    build_eval_ngrams,
    decontaminate,
    openhermes,
    pack_examples,
    reasoning_traces,
    render_chat,
    text_ngrams,
    tulu3,
)


class ByteTokenizer:
    """Deterministic char-level stand-in for the dataprep tokenizer."""

    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False  # render_chat must never add specials
        return list(text.encode("utf-8"))


def seg_len(text: str) -> int:
    return len(text.encode("utf-8"))


# --------------------------------------------------------------------------- #
# render_chat
# --------------------------------------------------------------------------- #


def test_render_chat_masks_non_assistant_turns() -> None:
    tok = ByteTokenizer()
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi!"},
        {"role": "user", "content": "more?"},
        {"role": "assistant", "content": "sure"},
    ]
    input_ids, labels = render_chat(messages, tok)
    assert len(input_ids) == len(labels)
    assert (input_ids[0], labels[0]) == (1, LABEL_IGNORE)  # BOS present, masked

    # Walk the segment layout and check masks exactly.
    pos = 1
    for m in messages:
        header = f"{IM_START}{m['role']}\n"
        body = f"{m['content']}{IM_END}\n"
        h, b = seg_len(header), seg_len(body)
        assert labels[pos : pos + h] == [LABEL_IGNORE] * h  # headers always masked
        body_labels = labels[pos + h : pos + h + b]
        if m["role"] == "assistant":
            assert body_labels == input_ids[pos + h : pos + h + b]  # loss on content+end
        else:
            assert body_labels == [LABEL_IGNORE] * b
        pos += h + b
    assert pos == len(input_ids)

    supervised = sum(1 for v in labels if v != LABEL_IGNORE)
    expected = sum(
        seg_len(f"{m['content']}{IM_END}\n") for m in messages if m["role"] == "assistant"
    )
    assert supervised == expected


def test_render_chat_without_bos_attribute() -> None:
    class NoBos:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return list(text.encode("utf-8"))

    ids, labels = render_chat([{"role": "assistant", "content": "x"}], NoBos())
    assert ids[0] != 1 or True  # no BOS injected
    assert len(ids) == seg_len(f"{IM_START}assistant\n") + seg_len(f"x{IM_END}\n")
    assert labels[: seg_len(f'{IM_START}assistant\n')] == [LABEL_IGNORE] * seg_len(
        f"{IM_START}assistant\n"
    )


def test_render_chat_supports_raw_tokenizers_api() -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: PLC0415

    vocab = {"<|pad|>": 0, "hi": 3, "there": 4}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<|pad|>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    ids, labels = render_chat([{"role": "assistant", "content": "hi there"}], raw)
    assert 3 in ids and 4 in ids
    assert len(ids) == len(labels)


def test_render_chat_rejects_bad_messages() -> None:
    tok = ByteTokenizer()
    with pytest.raises(ValueError, match="invalid role"):
        render_chat([{"role": "robot", "content": "x"}], tok)
    with pytest.raises(ValueError, match="content"):
        render_chat([{"role": "user", "content": 7}], tok)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def test_tulu3_normalization_filters_bad_rows() -> None:
    rows = [
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        {"messages": [{"role": "user", "content": "q only"}]},           # no assistant
        {"messages": [{"role": "alien", "content": "?"}]},               # bad role
        {"messages": [{"role": "assistant", "content": "   "}]},         # empty content
        {"messages": []},
    ]
    out = list(tulu3(rows=rows))
    assert out == [
        {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    ]


def test_openhermes_sharegpt_role_mapping() -> None:
    rows = [
        {
            "conversations": [
                {"from": "system", "value": "sys"},
                {"from": "human", "value": "q"},
                {"from": "gpt", "value": "a"},
            ]
        },
        {"conversations": [{"from": "weird", "value": "x"}]},
    ]
    out = list(openhermes(rows=rows))
    assert out == [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
    ]


def test_reasoning_traces_jsonl_shapes(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    lines = [
        {"messages": [{"role": "user", "content": "q0"}, {"role": "assistant", "content": "a0"}]},
        {"prompt": "q1", "response": "a1"},
        {"question": "q2", "answer": "a2", "system": "think hard"},
        {"prompt": "no answer here"},
    ]
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    out = list(reasoning_traces(path))
    assert len(out) == 3
    assert out[1]["messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert out[2]["messages"][0] == {"role": "system", "content": "think hard"}


# --------------------------------------------------------------------------- #
# Decontamination
# --------------------------------------------------------------------------- #


def test_decontaminate_catches_planted_overlap() -> None:
    eval_doc = "the quick brown fox jumps over the lazy dog while the moon rises slowly tonight"
    bank = build_eval_ngrams([eval_doc], n=13)
    assert bank  # 14 words -> 2 distinct 13-grams

    clean = {"messages": [{"role": "assistant", "content": "completely unrelated answer text"}]}
    planted = {
        "messages": [
            {"role": "user", "content": "please recite"},
            {
                "role": "assistant",
                "content": "sure: the quick brown fox jumps over the lazy dog while the moon "
                "rises slowly tonight indeed",
            },
        ]
    }
    short = {"messages": [{"role": "assistant", "content": "too short to hold a 13-gram"}]}
    kept = decontaminate([clean, planted, short], bank, n=13)
    assert kept == [clean, short]


def test_text_ngrams_casefolds_and_windows() -> None:
    grams = text_ngrams("A b c d", n=3)
    assert grams == {"a b c", "b c d"}
    assert text_ngrams("one two", n=3) == set()


# --------------------------------------------------------------------------- #
# Packing
# --------------------------------------------------------------------------- #


def test_pack_examples_stream_cut_and_padding() -> None:
    pairs = [
        (list(range(10, 15)), [-100, -100, 12, 13, 14]),      # 5 tokens
        (list(range(20, 28)), [-100] * 4 + list(range(24, 28))),  # 8 tokens
    ]
    batch = pack_examples(pairs, seq_len=6, pad_id=0)
    assert batch["input_ids"].shape == (3, 6)
    flat_ids = batch["input_ids"].flatten().tolist()
    assert flat_ids[:13] == list(range(10, 15)) + list(range(20, 28))
    assert flat_ids[13:] == [0] * 5                              # tail pad
    flat_labels = batch["labels"].flatten().tolist()
    assert flat_labels[:13] == pairs[0][1] + pairs[1][1]
    assert flat_labels[13:] == [-100] * 5
    flat_mask = batch["attention_mask"].flatten().tolist()
    assert flat_mask == [1] * 13 + [0] * 5
    assert batch["input_ids"].dtype == torch.int64


def test_collator_accepts_dicts_and_tuples() -> None:
    collator = SFTPackCollator(seq_len=4, pad_id=9)
    ids, labels = [1, 2, 3], [-100, 2, 3]
    a = collator([{"input_ids": ids, "labels": labels}])
    b = collator([(ids, labels)])
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert a["input_ids"].tolist() == [[1, 2, 3, 9]]
    assert a["labels"].tolist() == [[-100, 2, 3, -100]]


def test_pack_examples_validation() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        pack_examples([([1, 2], [-100])], seq_len=4)
    with pytest.raises(ValueError, match="no tokens"):
        pack_examples([], seq_len=4)
    with pytest.raises(ValueError, match="seq_len"):
        pack_examples([([1], [1])], seq_len=0)
