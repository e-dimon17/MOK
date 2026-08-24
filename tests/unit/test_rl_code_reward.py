"""RL: sandboxed code verifier — rlimit fallback path (bwrap absent in CI)."""

from __future__ import annotations

import time

import pytest

from rl.rewards.code_reward import (
    ExecResult,
    extract_code,
    resolve_sandbox,
    run_code_batch,
    run_snippet,
    verify_code,
)

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"
LOOP = "```python\nwhile True:\n    pass\n```"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def test_extract_code_python_fence() -> None:
    assert extract_code(GOOD) == "def add(a, b):\n    return a + b"


def test_extract_code_prefers_last_python_block() -> None:
    completion = "First try:\n```python\nx = 1\n```\nActually:\n```python\nx = 2\n```"
    assert extract_code(completion) == "x = 2"


def test_extract_code_generic_fence_and_raw_fallback() -> None:
    assert extract_code("```\ny = 3\n```") == "y = 3"
    assert extract_code("z = 4") == "z = 4"          # raw completion fallback
    assert extract_code("   \n  ") is None
    # a python fence beats a later generic fence
    assert extract_code("```python\na = 1\n```\n```\nb = 2\n```") == "a = 1"


def test_resolve_sandbox() -> None:
    assert resolve_sandbox("rlimit") == "rlimit"
    assert resolve_sandbox("bwrap") == "bwrap"
    assert resolve_sandbox("auto") in ("bwrap", "rlimit")
    with pytest.raises(ValueError):
        resolve_sandbox("chroot")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The three verdicts (rlimit fallback path)
# --------------------------------------------------------------------------- #


def test_verify_code_pass() -> None:
    score = verify_code(GOOD, ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"], sandbox="rlimit")
    assert score == 1.0


def test_verify_code_fail() -> None:
    assert verify_code(BAD, ["assert add(1, 2) == 3"], sandbox="rlimit") == 0.0


def test_verify_code_timeout_killed() -> None:
    start = time.monotonic()
    score = verify_code(LOOP, ["assert True"], timeout_s=1.0, sandbox="rlimit")
    elapsed = time.monotonic() - start
    assert score == 0.0
    assert elapsed < 5.0  # killed promptly, not left to run


def test_verify_code_fraction_passed() -> None:
    tests = ["assert add(1, 2) == 3", "assert add(2, 2) == 0"]
    assert verify_code(BAD, tests, sandbox="rlimit") == 0.5     # a - b: only 2-2==0 passes


def test_verify_code_isolation_per_test() -> None:
    # A crashing test must not poison the others (each runs in its own process).
    code = "```python\ndef f(x):\n    if x < 0:\n        import sys; sys.exit(3)\n    return x\n```"
    tests = ["assert f(-1) == -1", "assert f(5) == 5"]
    assert verify_code(code, tests, sandbox="rlimit") == 0.5


def test_verify_code_no_code_and_no_tests() -> None:
    assert verify_code("   ", ["assert True"], sandbox="rlimit") == 0.0
    with pytest.raises(ValueError):
        verify_code(GOOD, [])


def test_run_snippet_reports_stderr_and_timeout_flag() -> None:
    result = run_snippet("assert False, 'boom'", sandbox="rlimit")
    assert isinstance(result, ExecResult)
    assert not result.passed and not result.timed_out
    assert "boom" in result.stderr
    hung = run_snippet("while True:\n    pass", timeout_s=1.0, sandbox="rlimit")
    assert hung.timed_out and not hung.passed


# --------------------------------------------------------------------------- #
# Async batch runner
# --------------------------------------------------------------------------- #


async def test_run_code_batch_orders_and_scores() -> None:
    items = [
        {"completion": GOOD, "tests": ["assert add(1, 1) == 2"]},
        {"completion": BAD, "tests": ["assert add(1, 1) == 2"]},
        {"completion": GOOD, "tests": ["assert add(1, 1) == 2", "assert add(1, 1) == 3"]},
    ]
    scores = await run_code_batch(items, concurrency=2, sandbox="rlimit")
    assert scores == [1.0, 0.0, 0.5]


async def test_run_code_batch_validates_concurrency() -> None:
    with pytest.raises(ValueError):
        await run_code_batch([], concurrency=0)
