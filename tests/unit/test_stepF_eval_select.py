"""Step F: IFEval-style probe judges on canned outputs + checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from F.eval_select import (
    PROBES,
    judge_response,
    list_checkpoints,
    pick_checkpoint,
    score_checkpoint,
    score_outputs,
)

PROBE_BY_NAME = {p.name: p for p in PROBES}

# (probe name, passing canned output, failing canned output)
CANNED = [
    ("three_bullets", "- cardio\n- strength\n- mood", "- one\n- two"),
    ("json_keys_ab", 'Here you go: {"a": 1, "b": 2}', '{"a": 1, "b": 2, "c": 3}'),
    ("single_word", "Paris", "It is Paris"),
    ("all_caps", "THE SKY IS BLUE TODAY.", "The sky is blue."),
    ("lowercase_only", "tea is warm and calm.", "Tea is warm."),
    ("no_letter_e", "cats nap all day.", "cats sleep all day."),
    ("ends_that_is_all", "Sunny with light wind. That is all.", "Sunny today. That's it."),
    ("under_25_words", "Two lovers die; their feuding families reconcile.", "word " * 30),
    (
        "ocean_thrice",
        "The ocean is vast. I love the ocean because the ocean sings.",
        "The ocean is vast and blue.",
    ),
    ("numbered_1_to_4", "1. bread\n2. toaster\n3. wait\n4. eat", "1. bread\n2. toaster\n3. eat"),
    ("starts_certainly", "Certainly: a compiler translates source code.", "A compiler compiles."),
    ("no_commas", "I love rice with beans. It is warm and filling.", "Rice, beans, and eggs."),
    ("two_paragraphs", "Mountains are tall.\n\nThey hold snow.", "Mountains are tall and snowy."),
    ("json_list_of_five", '["apple", "pear", "fig", "plum", "kiwi"]', '["apple", "pear"]'),
    ("three_lines", "leaves fall gently\ncold wind whispers through the trees\nautumn says goodbye", "one line only"),
    ("yes_or_no", "yes", "Yes, it is prime."),
]


def test_battery_covers_all_probes() -> None:
    assert len(PROBES) >= 15
    assert {name for name, _, _ in CANNED} == set(PROBE_BY_NAME)


@pytest.mark.parametrize(("name", "good", "bad"), CANNED, ids=[c[0] for c in CANNED])
def test_probe_judges_canned_outputs(name: str, good: str, bad: str) -> None:
    probe = PROBE_BY_NAME[name]
    assert judge_response(probe, good), f"{name} should pass: {good!r}"
    assert not judge_response(probe, bad), f"{name} should fail: {bad!r}"


def test_score_outputs_fraction_and_missing_fail() -> None:
    all_good = {name: good for name, good, _ in CANNED}
    assert score_outputs(all_good) == 1.0
    assert score_outputs({}) == 0.0
    half = {name: good for name, good, _ in CANNED[:8]}
    assert score_outputs(half) == pytest.approx(8 / len(PROBES))


# --------------------------------------------------------------------------- #
# Checkpoint selection with an injected generator
# --------------------------------------------------------------------------- #


def _mk_checkpoint(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": "mok_moe"}))
    return d


def make_generate_fn(quality: dict[str, int]):
    """quality[ckpt_name] = number of probes answered correctly."""
    good = {name: g for name, g, _ in CANNED}
    bad = {name: b for name, _, b in CANNED}

    def generate(checkpoint: Path, prompts):
        n_good = quality[checkpoint.name]
        out = []
        for i, probe in enumerate(PROBES):
            out.append(good[probe.name] if i < n_good else bad[probe.name])
        assert len(prompts) == len(PROBES)
        return out

    return generate


def test_pick_checkpoint_selects_best(tmp_path) -> None:
    for name in ("checkpoint-200", "checkpoint-1000", "checkpoint-600"):
        _mk_checkpoint(tmp_path, name)
    generate = make_generate_fn({"checkpoint-200": 4, "checkpoint-600": 12, "checkpoint-1000": 7})
    best, scores = pick_checkpoint(tmp_path, generate_fn=generate)
    assert best.name == "checkpoint-600"
    assert scores[str(best)] == pytest.approx(12 / len(PROBES))
    assert len(scores) == 3


def test_pick_checkpoint_tie_goes_to_later(tmp_path) -> None:
    for name in ("checkpoint-100", "checkpoint-300"):
        _mk_checkpoint(tmp_path, name)
    generate = make_generate_fn({"checkpoint-100": 6, "checkpoint-300": 6})
    best, _ = pick_checkpoint(tmp_path, generate_fn=generate)
    assert best.name == "checkpoint-300"


def test_list_checkpoints_numeric_order_and_final(tmp_path) -> None:
    for name in ("checkpoint-1000", "checkpoint-200", "checkpoint-80"):
        _mk_checkpoint(tmp_path, name)
    (tmp_path / "checkpoint-999-noconfig").mkdir()  # no config.json -> ignored
    _mk_checkpoint(tmp_path, "final")
    found = [p.name for p in list_checkpoints(tmp_path)]
    assert found == ["checkpoint-80", "checkpoint-200", "checkpoint-1000", "final"]


def test_pick_checkpoint_empty_dir_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        pick_checkpoint(tmp_path, generate_fn=lambda c, p: [])


def test_score_checkpoint_validates_response_count(tmp_path) -> None:
    ckpt = _mk_checkpoint(tmp_path, "checkpoint-1")
    with pytest.raises(ValueError, match="responses"):
        score_checkpoint(ckpt, lambda c, p: ["only one"])
