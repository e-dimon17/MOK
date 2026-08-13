"""RLVR math prompt sets: GSM8K + MATH train splits (step G).

`build_math_prompts()` maps both corpora to answer-checkable items

    {"prompt": <question + boxed-answer instruction>,
     "reference_answer": <gold final answer>,
     "tag": "math"}

`datasets` (HuggingFace) is imported lazily and only when rows are not
injected, so tests and offline runs never touch the Hub. Subsampling is
deterministic in (seed, input order) — the same prompt set on every host.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from G.rewards.math_reward import extract_boxed

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
MATH_DATASET = "EleutherAI/hendrycks_math"
#: The seven MATH subject configs (hendrycks_math splits the corpus by topic).
MATH_CONFIGS: tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

#: Appended to every math prompt so `verify_math`'s boxed extraction fires.
MATH_PROMPT_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

def deterministic_subsample[T](items: Sequence[T], n: int | None, seed: int) -> list[T]:
    """First-class consensus-ish helper: sample n WITHOUT replacement, keep the
    original relative order, fully determined by (seed, len(items))."""
    if n is not None and n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    keep = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in keep]


def gsm8k_reference(answer: str) -> str | None:
    """Gold answer from a GSM8K `answer` field: text after `####`, commas
    stripped. None if the marker is missing/empty."""
    if "####" not in answer:
        return None
    tail = answer.rsplit("####", 1)[1].strip().split("\n")[0].strip()
    tail = tail.replace(",", "").replace("$", "").rstrip(".")
    return tail or None


def math_reference(solution: str) -> str | None:
    """Gold answer from a MATH `solution` field: the last `\\boxed{...}`."""
    return extract_boxed(solution)


def _gsm8k_items(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        question = row.get("question")
        reference = gsm8k_reference(str(row.get("answer") or ""))
        if isinstance(question, str) and question.strip() and reference:
            items.append(
                {
                    "prompt": question.strip() + MATH_PROMPT_SUFFIX,
                    "reference_answer": reference,
                    "tag": "math",
                }
            )
    return items


def _math_items(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        problem = row.get("problem")
        reference = math_reference(str(row.get("solution") or ""))
        if isinstance(problem, str) and problem.strip() and reference:
            items.append(
                {
                    "prompt": problem.strip() + MATH_PROMPT_SUFFIX,
                    "reference_answer": reference,
                    "tag": "math",
                }
            )
    return items


def _load_default_gsm8k() -> Iterable[Mapping[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415 — lazy, [post] extra

    return load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split="train")


def _load_default_math() -> list[Mapping[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415 — lazy, [post] extra

    rows: list[Mapping[str, Any]] = []
    for config in MATH_CONFIGS:
        rows.extend(load_dataset(MATH_DATASET, config, split="train"))
    return rows


def build_math_prompts(
    *,
    n: int | None = None,
    seed: int = 0,
    gsm8k_rows: Iterable[Mapping[str, Any]] | None = None,
    math_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """GSM8K + MATH train items, deterministically subsampled to `n`.

    `gsm8k_rows` / `math_rows` inject fixtures (tests, offline); by default the
    train splits stream from the Hub via `datasets` (lazy import).
    """
    if gsm8k_rows is None:
        gsm8k_rows = _load_default_gsm8k()
    if math_rows is None:
        math_rows = _load_default_math()
    items = _gsm8k_items(gsm8k_rows) + _math_items(math_rows)
    return deterministic_subsample(items, n, seed)


__all__ = [
    "GSM8K_CONFIG",
    "GSM8K_DATASET",
    "MATH_CONFIGS",
    "MATH_DATASET",
    "MATH_PROMPT_SUFFIX",
    "build_math_prompts",
    "deterministic_subsample",
    "gsm8k_reference",
    "math_reference",
]
