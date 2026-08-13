"""RLVR code prompt sets: MBPP train + self-generated JSONL (step G).

`build_code_prompts()` yields execution-checkable items

    {"prompt": <problem statement + tests shown + code-block instruction>,
     "tests": [<assert ...>, ...],
     "tag": "code"}

MBPP rows come from `datasets` (lazy import, injectable for tests); the
self-generated loader reads HumanEval-train-style JSONL lines
{"prompt": str, "tests": list[str]} (a single "test" string is also accepted)
produced by earlier model generations that passed their own tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from G.data.rlvr_math import deterministic_subsample

MBPP_DATASET = "google-research-datasets/mbpp"

CODE_PROMPT_TEMPLATE = (
    "Write a Python function to solve the following problem.\n\n"
    "{problem}\n\n"
    "Your solution must pass these tests:\n\n"
    "{tests}\n\n"
    "Answer with the complete solution in a single ```python code block."
)


def format_code_prompt(problem: str, tests: Sequence[str]) -> str:
    return CODE_PROMPT_TEMPLATE.format(problem=problem.strip(), tests="\n".join(tests))


def _clean_tests(raw: Any, setup: str = "") -> list[str] | None:
    """Coerce a tests field into a non-empty list of runnable snippets; the
    optional MBPP `test_setup_code` is prepended to each (tests run isolated)."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return None
    tests: list[str] = []
    prefix = f"{setup.strip()}\n" if isinstance(setup, str) and setup.strip() else ""
    for test in raw:
        if not isinstance(test, str) or not test.strip():
            return None
        tests.append(prefix + test.strip())
    return tests or None


def mbpp_item(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """One MBPP row -> a code item (None if unusable). Uses `text` (the task
    description) + `test_list` (+ `test_setup_code`)."""
    problem = row.get("text") or row.get("prompt")
    tests = _clean_tests(row.get("test_list"), str(row.get("test_setup_code") or ""))
    if not isinstance(problem, str) or not problem.strip() or not tests:
        return None
    return {"prompt": format_code_prompt(problem, tests), "tests": tests, "tag": "code"}


def load_selfgen_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Self-generated problems: JSONL of {"prompt": str, "tests": list[str]}
    (or "test": str). Malformed lines raise — self-generated data is ours, so
    corruption is a bug, not noise."""
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            tests = _clean_tests(row.get("tests", row.get("test")))
            if not isinstance(prompt, str) or not prompt.strip() or not tests:
                raise ValueError(f"{path}:{line_no}: expected {{'prompt': str, 'tests': list[str]}}")
            items.append({"prompt": prompt.strip(), "tests": tests, "tag": "code"})
    return items


def _load_default_mbpp() -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415 — lazy, [post] extra

    return load_dataset(MBPP_DATASET, split="train")


def build_code_prompts(
    *,
    n: int | None = None,
    seed: int = 0,
    mbpp_rows: Iterable[Mapping[str, Any]] | None = None,
    selfgen_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """MBPP train items plus optional self-generated JSONL, deterministically
    subsampled to `n` (same helper/order semantics as the math builder)."""
    if mbpp_rows is None:
        mbpp_rows = _load_default_mbpp()
    items = [item for item in (mbpp_item(row) for row in mbpp_rows) if item is not None]
    if selfgen_path is not None:
        items.extend(load_selfgen_jsonl(selfgen_path))
    return deterministic_subsample(items, n, seed)


__all__ = [
    "CODE_PROMPT_TEMPLATE",
    "MBPP_DATASET",
    "build_code_prompts",
    "format_code_prompt",
    "load_selfgen_jsonl",
    "mbpp_item",
]
