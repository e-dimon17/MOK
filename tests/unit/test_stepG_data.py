"""Step G: RLVR dataset builders — fixtures injected AND `datasets` mocked."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from G.data.rlvr_code import (
    MBPP_DATASET,
    build_code_prompts,
    format_code_prompt,
    load_selfgen_jsonl,
    mbpp_item,
)
from G.data.rlvr_math import (
    GSM8K_CONFIG,
    GSM8K_DATASET,
    MATH_CONFIGS,
    MATH_DATASET,
    MATH_PROMPT_SUFFIX,
    build_math_prompts,
    deterministic_subsample,
    gsm8k_reference,
    math_reference,
)

# --------------------------------------------------------------------------- #
# Reference extraction + subsampling helpers
# --------------------------------------------------------------------------- #


def test_gsm8k_reference() -> None:
    assert gsm8k_reference("reasoning...\n#### 72") == "72"
    assert gsm8k_reference("#### $1,234.") == "1234"
    assert gsm8k_reference("no marker") is None
    assert gsm8k_reference("#### ") is None


def test_math_reference() -> None:
    assert math_reference("Thus $x = \\boxed{\\frac{1}{2}}$.") == "\\frac{1}{2}"
    assert math_reference("no boxed answer") is None


def test_deterministic_subsample() -> None:
    items = list(range(100))
    a = deterministic_subsample(items, 10, seed=0)
    assert a == deterministic_subsample(items, 10, seed=0)   # reproducible
    assert a != deterministic_subsample(items, 10, seed=1)   # seed-sensitive
    assert a == sorted(a)                                     # original order kept
    assert len(set(a)) == 10                                  # without replacement
    assert deterministic_subsample(items, None, seed=0) == items
    assert deterministic_subsample(items, 1000, seed=0) == items
    assert deterministic_subsample(items, 0, seed=0) == []
    with pytest.raises(ValueError):
        deterministic_subsample(items, -1, seed=0)


# --------------------------------------------------------------------------- #
# Builders with injected fixture rows
# --------------------------------------------------------------------------- #


def test_build_math_prompts_injected_rows() -> None:
    gsm8k = [
        {"question": "Two apples plus two?", "answer": "2+2=4\n#### 4"},
        {"question": "Broken row", "answer": "no marker"},   # dropped
        {"question": "", "answer": "#### 5"},                # dropped
    ]
    math = [
        {"problem": "Compute $1+1$.", "solution": "It is $\\boxed{2}$."},
        {"problem": "No answer", "solution": "unfinished"},  # dropped
    ]
    items = build_math_prompts(gsm8k_rows=gsm8k, math_rows=math)
    assert len(items) == 2
    assert all(item["tag"] == "math" for item in items)
    assert all(item["prompt"].endswith(MATH_PROMPT_SUFFIX) for item in items)
    assert items[0]["reference_answer"] == "4"
    assert items[1]["reference_answer"] == "2"  # boxed content, unwrapped


def test_mbpp_item_and_prompt_format() -> None:
    row = {
        "text": "Write a function to add two numbers.",
        "test_list": ["assert add(1, 2) == 3"],
        "test_setup_code": "",
    }
    item = mbpp_item(row)
    assert item is not None
    assert item["tag"] == "code"
    assert item["tests"] == ["assert add(1, 2) == 3"]
    assert item["prompt"] == format_code_prompt(row["text"], item["tests"])
    assert "```python" in item["prompt"]
    # setup code is prepended to each test (tests run in isolation)
    with_setup = mbpp_item({**row, "test_setup_code": "import math"})
    assert with_setup is not None
    assert with_setup["tests"] == ["import math\nassert add(1, 2) == 3"]
    assert mbpp_item({"text": "no tests", "test_list": []}) is None
    assert mbpp_item({"test_list": ["assert True"]}) is None


def test_load_selfgen_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "selfgen.jsonl"
    rows = [
        {"prompt": "Write f.", "tests": ["assert f() == 1"]},
        {"prompt": "Write g.", "test": "assert g() == 2"},   # single-string form
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    items = load_selfgen_jsonl(path)
    assert [item["tests"] for item in items] == [["assert f() == 1"], ["assert g() == 2"]]
    assert all(item["tag"] == "code" for item in items)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"prompt": "no tests"}) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_selfgen_jsonl(bad)


def test_build_code_prompts_merges_selfgen(tmp_path: Path) -> None:
    path = tmp_path / "selfgen.jsonl"
    path.write_text(json.dumps({"prompt": "Write h.", "tests": ["assert h() == 3"]}) + "\n")
    mbpp = [{"text": "Write add.", "test_list": ["assert add(1, 1) == 2"]}]
    items = build_code_prompts(mbpp_rows=mbpp, selfgen_path=path)
    assert len(items) == 2
    subsampled = build_code_prompts(mbpp_rows=mbpp, selfgen_path=path, n=1, seed=0)
    assert len(subsampled) == 1


# --------------------------------------------------------------------------- #
# Default path with `datasets` mocked (load_dataset monkeypatched)
# --------------------------------------------------------------------------- #

_GSM8K_FIXTURE = [{"question": "Q?", "answer": "#### 9"}]
_MATH_FIXTURE = {
    config: [{"problem": f"P-{config}", "solution": f"\\boxed{{{i}}}"}]
    for i, config in enumerate(MATH_CONFIGS)
}
_MBPP_FIXTURE = [{"text": "Write add.", "test_list": ["assert add(1, 1) == 2"]}]


@pytest.fixture
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    calls: list[tuple] = []

    def load_dataset(path: str, name: str | None = None, *, split: str, **kwargs):
        calls.append((path, name, split))
        if path == GSM8K_DATASET:
            assert name == GSM8K_CONFIG
            return list(_GSM8K_FIXTURE)
        if path == MATH_DATASET:
            return list(_MATH_FIXTURE[name])
        if path == MBPP_DATASET:
            assert name is None
            return list(_MBPP_FIXTURE)
        raise AssertionError(f"unexpected dataset {path!r}")

    module = types.ModuleType("datasets")
    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)
    return calls


def test_build_math_prompts_mocked_hub(fake_datasets: list[tuple]) -> None:
    items = build_math_prompts(n=5, seed=0)
    assert len(items) == 5  # 1 GSM8K + 7 MATH configs -> 8, subsampled to 5
    assert (GSM8K_DATASET, GSM8K_CONFIG, "train") in fake_datasets
    for config in MATH_CONFIGS:
        assert (MATH_DATASET, config, "train") in fake_datasets


def test_build_code_prompts_mocked_hub(fake_datasets: list[tuple]) -> None:
    items = build_code_prompts()
    assert len(items) == 1
    assert fake_datasets == [(MBPP_DATASET, None, "train")]
