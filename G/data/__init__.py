"""RLVR prompt-set builders (step G): GSM8K + MATH (tag='math') and MBPP +
self-generated JSONL (tag='code'). `datasets` loads lazily; rows injectable."""

from G.data.rlvr_code import build_code_prompts, load_selfgen_jsonl, mbpp_item
from G.data.rlvr_math import build_math_prompts, deterministic_subsample

__all__ = [
    "build_code_prompts",
    "build_math_prompts",
    "deterministic_subsample",
    "load_selfgen_jsonl",
    "mbpp_item",
]
