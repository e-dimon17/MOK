"""Step G: grpo.yaml parsing, reward_router dispatch, TRL reward adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import G.grpo_train as grpo_train
from G.grpo_train import (
    GRPOSettings,
    VllmSettings,
    build_rlvr_dataset,
    endpoint_for_rank,
    load_settings,
    make_trl_reward_fn,
    reward_router,
)

REPO = Path(__file__).resolve().parents[2]

MATH_SAMPLE = {"tag": "math", "reference_answer": "4", "tests": []}
CODE_SAMPLE = {"tag": "code", "reference_answer": "", "tests": ["assert add(1, 1) == 2"]}
ADD_COMPLETION = "```python\ndef add(a, b):\n    return a + b\n```"


def test_load_settings_parses_shipped_yaml() -> None:
    settings = load_settings(REPO / "G" / "configs" / "grpo.yaml")
    assert settings.group_size == 8
    assert settings.lr == pytest.approx(1e-6)
    assert settings.kl_coef == pytest.approx(0.04)
    assert settings.bf16 is True
    assert settings.vllm.server_mode is True
    assert settings.vllm.endpoints == ("http://127.0.0.1:8000",)
    assert settings.code_timeout_s == 6
    assert settings.code_sandbox == "auto"
    assert settings.datasets["math"]["n"] == 12000
    assert settings.datasets["code"]["n"] == 6000
    # micro batch counts completions; groups must tile evenly onto it
    assert settings.group_size % settings.micro_batch_size == 0 or (
        settings.micro_batch_size % settings.group_size == 0
    )


def test_vllm_settings_validation() -> None:
    with pytest.raises(ValueError):
        VllmSettings(server_mode=True, endpoints=())
    with pytest.raises(ValueError):
        VllmSettings(endpoints=("ftp://nope:1",))
    ok = VllmSettings(endpoints=("http://10.0.0.1:8000", "https://roll.example.com"))
    assert len(ok.endpoints) == 2


def test_endpoint_for_rank_round_robin() -> None:
    settings = GRPOSettings(
        model_dir="m",
        output_dir="o",
        vllm=VllmSettings(endpoints=("http://a:1", "http://b:2")),
    )
    assert endpoint_for_rank(settings, 0) == "http://a:1"
    assert endpoint_for_rank(settings, 1) == "http://b:2"
    assert endpoint_for_rank(settings, 2) == "http://a:1"


# --------------------------------------------------------------------------- #
# reward_router dispatch
# --------------------------------------------------------------------------- #


def test_reward_router_math_dispatch() -> None:
    assert reward_router(MATH_SAMPLE, "So the answer is \\boxed{4}.") == 1.0
    assert reward_router(MATH_SAMPLE, "So the answer is \\boxed{5}.") == 0.0


def test_reward_router_code_dispatch_forwards_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_verify_code(completion, tests, *, timeout_s, sandbox):
        seen.update(completion=completion, tests=tests, timeout_s=timeout_s, sandbox=sandbox)
        return 0.25

    monkeypatch.setattr(grpo_train, "verify_code", fake_verify_code)
    score = reward_router(CODE_SAMPLE, ADD_COMPLETION, code_timeout_s=2.5, code_sandbox="rlimit")
    assert score == 0.25
    assert seen == {
        "completion": ADD_COMPLETION,
        "tests": CODE_SAMPLE["tests"],
        "timeout_s": 2.5,
        "sandbox": "rlimit",
    }


def test_reward_router_code_dispatch_real_sandbox() -> None:
    assert reward_router(CODE_SAMPLE, ADD_COMPLETION, code_sandbox="rlimit") == 1.0


def test_reward_router_unknown_tag_raises() -> None:
    with pytest.raises(ValueError, match="unknown task tag"):
        reward_router({"tag": "trivia"}, "whatever")
    with pytest.raises(ValueError, match="unknown task tag"):
        reward_router({}, "whatever")


# --------------------------------------------------------------------------- #
# TRL adapter: batched columns -> per-row samples
# --------------------------------------------------------------------------- #


def test_make_trl_reward_fn_columns() -> None:
    fn = make_trl_reward_fn(code_sandbox="rlimit")
    rewards = fn(
        prompts=["p1", "p2"],
        completions=["\\boxed{4}", ADD_COMPLETION],
        tag=["math", "code"],
        reference_answer=["4", ""],
        tests=[[], ["assert add(2, 3) == 5"]],
        trainer_state="not-a-column",  # non-list kwargs must be ignored
    )
    assert rewards == [1.0, 1.0]


def test_make_trl_reward_fn_conversational_completions() -> None:
    fn = make_trl_reward_fn()
    rewards = fn(
        prompts=["p"],
        completions=[[{"role": "assistant", "content": "the result is \\boxed{4}"}]],
        tag=["math"],
        reference_answer=["4"],
        tests=[[]],
    )
    assert rewards == [1.0]


def test_make_trl_reward_fn_requires_completions() -> None:
    with pytest.raises(ValueError):
        make_trl_reward_fn()(prompts=["p"])


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #

GSM8K_ROWS = [{"question": f"Q{i}?", "answer": f"steps\n#### {i}"} for i in range(4)]
MATH_ROWS = [{"problem": f"P{i}", "solution": f"\\boxed{{{i}}}"} for i in range(3)]
MBPP_ROWS = [
    {"text": f"Write function f{i}.", "test_list": [f"assert f{i}(0) == {i}"], "test_setup_code": ""}
    for i in range(3)
]


def _settings(**kwargs) -> GRPOSettings:
    base = {
        "model_dir": "m",
        "output_dir": "o",
        "datasets": {"math": {"n": None}, "code": {"n": None, "selfgen_jsonl": None}},
    }
    base.update(kwargs)
    return GRPOSettings(**base)


def test_build_rlvr_dataset_uniform_schema_and_deterministic() -> None:
    settings = _settings()
    items = build_rlvr_dataset(
        settings, gsm8k_rows=GSM8K_ROWS, math_rows=MATH_ROWS, mbpp_rows=MBPP_ROWS
    )
    assert len(items) == 4 + 3 + 3
    assert all(set(item) == {"prompt", "tag", "reference_answer", "tests"} for item in items)
    assert {item["tag"] for item in items} == {"math", "code"}
    again = build_rlvr_dataset(
        settings, gsm8k_rows=GSM8K_ROWS, math_rows=MATH_ROWS, mbpp_rows=MBPP_ROWS
    )
    assert items == again  # seed-deterministic shuffle
    other_seed = build_rlvr_dataset(
        _settings(seed=7), gsm8k_rows=GSM8K_ROWS, math_rows=MATH_ROWS, mbpp_rows=MBPP_ROWS
    )
    assert sorted(map(str, other_seed)) == sorted(map(str, items))  # same set, other order


def test_build_rlvr_dataset_caps_and_flags() -> None:
    capped = build_rlvr_dataset(
        _settings(max_prompts=5), gsm8k_rows=GSM8K_ROWS, math_rows=MATH_ROWS, mbpp_rows=MBPP_ROWS
    )
    assert len(capped) == 5
    math_only = build_rlvr_dataset(
        _settings(datasets={"math": {"n": 2}}), gsm8k_rows=GSM8K_ROWS, math_rows=MATH_ROWS
    )
    assert len(math_only) == 2
    assert all(item["tag"] == "math" for item in math_only)
    assert build_rlvr_dataset(_settings(datasets={})) == []
    # Everything above ran without the heavy post-training stack.
    assert "trl" not in sys.modules
    assert "vllm" not in sys.modules
