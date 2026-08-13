"""H/run_evals.py — pure helpers (markdown golden, extraction, humaneval cmd)
plus the CLI with lm-eval / the bigcode harness monkeypatched away."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from H import run_evals
from H.run_evals import (
    DEFAULT_TASKS,
    extract_results,
    humaneval_cmd,
    main,
    results_to_markdown,
    run_humaneval,
)

# --------------------------------------------------------------------------- #
# extract_results
# --------------------------------------------------------------------------- #


def test_extract_results_from_simple_evaluate_output():
    raw = {
        "results": {
            "mmlu": {"acc,none": 0.6512, "acc_stderr,none": 0.0043, "alias": "mmlu"},
            "gsm8k_cot": {
                "exact_match,strict-match": 0.5,
                "exact_match,flexible-extract": 0.55,
                "alias": " gsm8k_cot",
            },
            "winogrande": {"acc,none": True, "alias": "winogrande"},  # bool is not a score
            "empty_task": {"alias": "empty"},
        },
        "configs": {"mmlu": {"num_fewshot": 5}},
    }
    assert extract_results(raw) == {
        "mmlu": {"acc": 0.6512, "acc_stderr": 0.0043},
        "gsm8k_cot": {"exact_match/strict-match": 0.5, "exact_match/flexible-extract": 0.55},
    }


def test_extract_results_accepts_bare_results_mapping():
    assert extract_results({"humaneval": {"pass@1": 0.42}}) == {"humaneval": {"pass@1": 0.42}}


# --------------------------------------------------------------------------- #
# results_to_markdown golden
# --------------------------------------------------------------------------- #


def test_results_to_markdown_golden():
    results = {
        "mmlu": {"acc": 0.6512},
        "gsm8k_cot": {"exact_match": 0.55},
        "humaneval": {"pass@1": 0.421875},
    }
    expected = (
        "| Task | Metric | Value |\n"
        "|---|---|---:|\n"
        "| gsm8k_cot | exact_match | 0.5500 |\n"
        "| humaneval | pass@1 | 0.4219 |\n"
        "| mmlu | acc | 0.6512 |\n"
    )
    assert results_to_markdown(results) == expected


def test_results_to_markdown_empty_and_integers():
    assert results_to_markdown({}) == "| Task | Metric | Value |\n|---|---|---:|\n"
    table = results_to_markdown({"t": {"n_samples": 164.0}})
    assert "| t | n_samples | 164 |" in table


# --------------------------------------------------------------------------- #
# HumanEval helper
# --------------------------------------------------------------------------- #


def test_humaneval_cmd_contents(tmp_path):
    cmd = humaneval_cmd(tmp_path / "model", n_samples=50, temperature=0.8, batch_size=25)
    assert cmd[:3] == ["accelerate", "launch", "main.py"]
    assert cmd[cmd.index("--model") + 1] == str(tmp_path / "model")
    assert cmd[cmd.index("--tasks") + 1] == "humaneval"
    assert cmd[cmd.index("--n_samples") + 1] == "50"
    assert cmd[cmd.index("--temperature") + 1] == "0.8"
    assert cmd[cmd.index("--batch_size") + 1] == "25"
    assert "--allow_code_execution" in cmd
    assert "--trust_remote_code" in cmd
    assert cmd[cmd.index("--metric_output_path") + 1] == "humaneval_metrics.json"


def test_run_humaneval_guard_missing_harness(tmp_path):
    with pytest.raises(FileNotFoundError, match="bigcode-evaluation-harness not found"):
        run_humaneval(tmp_path / "model", tmp_path / "no-harness")


def test_run_humaneval_with_fake_harness(tmp_path, monkeypatch):
    harness = tmp_path / "bigcode-evaluation-harness"
    harness.mkdir()
    (harness / "main.py").write_text("# harness entry\n")

    def fake_run(cmd, *, cwd, check, timeout):
        assert check is True
        assert cmd[:3] == ["accelerate", "launch", "main.py"]
        (Path(cwd) / "humaneval_metrics.json").write_text(
            json.dumps({"humaneval": {"pass@1": 0.42, "pass@10": 0.61}, "config": {"model": "x"}})
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run_evals.subprocess, "run", fake_run)
    scores = run_humaneval(tmp_path / "model", harness)
    assert scores == {"humaneval": {"pass@1": 0.42, "pass@10": 0.61}}


def test_run_humaneval_propagates_subprocess_failure(tmp_path, monkeypatch):
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "main.py").write_text("")

    def fail_run(cmd, *, cwd, check, timeout):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(run_evals.subprocess, "run", fail_run)
    with pytest.raises(subprocess.CalledProcessError):
        run_humaneval(tmp_path / "model", harness)


# --------------------------------------------------------------------------- #
# CLI (lm-eval monkeypatched)
# --------------------------------------------------------------------------- #


def test_main_writes_evals_json_and_markdown(tmp_path, monkeypatch):
    seen = {}

    def fake_lm_eval(model_path, *, tasks, backend, model_args, batch_size, limit):
        seen.update(
            model_path=model_path, tasks=tasks, backend=backend,
            model_args=model_args, batch_size=batch_size, limit=limit,
        )
        return {"mmlu": {"acc": 0.65}, "hellaswag": {"acc_norm": 0.71}}

    monkeypatch.setattr(run_evals, "run_lm_eval", fake_lm_eval)
    out = tmp_path / "evals.json"
    md = tmp_path / "evals.md"
    rc = main(
        [
            "--model-path", "releases/mok-54b-chat",
            "--backend", "hf",
            "--limit", "8",
            "--out", str(out),
            "--markdown-out", str(md),
        ]
    )
    assert rc == 0
    assert seen["model_path"] == "releases/mok-54b-chat"
    assert seen["tasks"] == DEFAULT_TASKS
    assert seen["backend"] == "hf"
    assert seen["limit"] == 8
    assert json.loads(out.read_text()) == {"mmlu": {"acc": 0.65}, "hellaswag": {"acc_norm": 0.71}}
    assert md.read_text() == results_to_markdown({"mmlu": {"acc": 0.65}, "hellaswag": {"acc_norm": 0.71}})


def test_main_merges_humaneval(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evals, "run_lm_eval", lambda *a, **kw: {"mmlu": {"acc": 0.65}})
    monkeypatch.setattr(
        run_evals, "run_humaneval", lambda model, harness, **kw: {"humaneval": {"pass@1": 0.42}}
    )
    out = tmp_path / "evals.json"
    rc = main(
        [
            "--model-path", "m",
            "--tasks", "mmlu",
            "--out", str(out),
            "--humaneval-harness", str(tmp_path),
        ]
    )
    assert rc == 0
    assert json.loads(out.read_text()) == {"mmlu": {"acc": 0.65}, "humaneval": {"pass@1": 0.42}}


def test_main_rejects_empty_tasks(tmp_path, capsys):
    rc = main(["--model-path", "m", "--tasks", " , ", "--out", str(tmp_path / "e.json")])
    assert rc == 2
    assert "no tasks" in capsys.readouterr().err
