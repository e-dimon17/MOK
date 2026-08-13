"""Benchmark runner for the release: lm-eval-harness suites + HumanEval.

Console script `mok-eval` (pyproject). Runs the release task list

    mmlu, gsm8k_cot, ifeval, arc_challenge, hellaswag, winogrande

through lm-eval-harness on a vllm (default) or hf backend, plus HumanEval via
a guarded bigcode-evaluation-harness subprocess, and writes:

  - evals.json      {task: {metric: value}} — the exact `eval_results` input
                    H/provenance.build_bundle expects
  - evals.md        (optional) the markdown benchmarks table for the model card

lm-eval and the bigcode harness are heavy optional deps; both are reached only
inside the run functions, so this module imports instantly and its pure pieces
(results_to_markdown, extract_results, humaneval_cmd) are CPU-unit-tested.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: The release eval suite (playbook step H). Order is presentation order.
DEFAULT_TASKS: tuple[str, ...] = (
    "mmlu",
    "gsm8k_cot",
    "ifeval",
    "arc_challenge",
    "hellaswag",
    "winogrande",
)

HUMANEVAL_TASK = "humaneval"
HUMANEVAL_METRICS_FILENAME = "humaneval_metrics.json"


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #


def extract_results(raw: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Flatten lm-eval-harness output into {task: {metric: value}}.

    Accepts either the full simple_evaluate() dict (with a 'results' key) or
    the results mapping itself. Keeps numeric metrics only, strips lm-eval's
    ',<filter>' suffixes (e.g. 'acc,none' -> 'acc'; a non-default filter like
    'exact_match,strict-match' keeps its filter as 'exact_match/strict-match'),
    and drops 'alias' entries.
    """
    results = raw.get("results", raw)
    out: dict[str, dict[str, float]] = {}
    for task in sorted(results):
        metrics = results[task]
        if not isinstance(metrics, Mapping):
            continue
        clean: dict[str, float] = {}
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            name, _, metric_filter = key.partition(",")
            if name == "alias":
                continue
            if metric_filter and metric_filter != "none":
                name = f"{name}/{metric_filter}"
            clean[name] = float(value)
        if clean:
            out[task] = clean
    return out


def _format_value(value: float) -> str:
    if float(value).is_integer() and abs(value) >= 1:
        return str(int(value))
    return f"{value:.4f}"


def results_to_markdown(results: Mapping[str, Mapping[str, float]]) -> str:
    """Render {task: {metric: value}} as a GitHub-flavored markdown table."""
    lines = ["| Task | Metric | Value |", "|---|---|---:|"]
    for task in sorted(results):
        for metric in sorted(results[task]):
            lines.append(f"| {task} | {metric} | {_format_value(results[task][metric])} |")
    return "\n".join(lines) + "\n"


def humaneval_cmd(
    model_dir: str | Path,
    *,
    metrics_out: str | Path = HUMANEVAL_METRICS_FILENAME,
    n_samples: int = 20,
    temperature: float = 0.2,
    batch_size: int = 10,
) -> list[str]:
    """The bigcode-evaluation-harness invocation for HumanEval (pass@1).

    Returned as an argv list to run with cwd=<bigcode-evaluation-harness
    checkout>. --allow_code_execution is required by the harness; generated
    code still runs in its sandboxed executor. --trust_remote_code loads the
    custom MoK-MoE modeling class shipped by step F.
    """
    return [
        "accelerate",
        "launch",
        "main.py",
        "--model",
        str(model_dir),
        "--tasks",
        HUMANEVAL_TASK,
        "--n_samples",
        str(n_samples),
        "--temperature",
        str(temperature),
        "--batch_size",
        str(batch_size),
        "--allow_code_execution",
        "--trust_remote_code",
        "--metric_output_path",
        str(metrics_out),
    ]


# --------------------------------------------------------------------------- #
# Heavy runners (lazy deps)
# --------------------------------------------------------------------------- #


def run_lm_eval(
    model_path: str,
    *,
    tasks: Sequence[str],
    backend: str = "vllm",
    model_args: str = "",
    batch_size: str = "auto",
    limit: int | None = None,
) -> dict[str, dict[str, float]]:
    """Run lm-eval-harness (lazy import) and return {task: {metric: value}}."""
    import lm_eval  # noqa: PLC0415

    args = f"pretrained={model_path},trust_remote_code=True"
    if model_args:
        args = f"{args},{model_args}"
    output = lm_eval.simple_evaluate(
        model=backend,
        model_args=args,
        tasks=list(tasks),
        batch_size=batch_size,
        limit=limit,
    )
    if output is None:  # non-rank-0 under distributed launch
        return {}
    return extract_results(output)


def run_humaneval(
    model_dir: str | Path,
    harness_dir: str | Path,
    *,
    n_samples: int = 20,
    temperature: float = 0.2,
    batch_size: int = 10,
    timeout_s: float | None = None,
) -> dict[str, dict[str, float]]:
    """Guarded bigcode-evaluation-harness run; returns {'humaneval': {metric: value}}.

    Guards: the harness checkout must exist (main.py present) and the
    subprocess must succeed — otherwise this raises instead of silently
    shipping a release without code scores.
    """
    harness = Path(harness_dir)
    if not (harness / "main.py").is_file():
        raise FileNotFoundError(
            f"bigcode-evaluation-harness not found at {harness} (expected main.py); "
            "clone https://github.com/bigcode-project/bigcode-evaluation-harness"
        )
    metrics_path = harness / HUMANEVAL_METRICS_FILENAME
    cmd = humaneval_cmd(
        model_dir,
        metrics_out=HUMANEVAL_METRICS_FILENAME,
        n_samples=n_samples,
        temperature=temperature,
        batch_size=batch_size,
    )
    subprocess.run(cmd, cwd=harness, check=True, timeout=timeout_s)
    raw = json.loads(metrics_path.read_bytes())
    scores = raw.get(HUMANEVAL_TASK, raw)
    return extract_results({HUMANEVAL_TASK: scores})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-eval",
        description="Run the release benchmark suite and write evals.json (+ markdown table).",
    )
    parser.add_argument("--model-path", required=True, help="HF model dir or hub id to evaluate")
    parser.add_argument("--backend", choices=("vllm", "hf"), default="vllm", help="lm-eval model backend")
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help=f"comma-separated lm-eval task list (default: {','.join(DEFAULT_TASKS)})",
    )
    parser.add_argument("--model-args", default="", help="extra lm-eval model_args, comma-separated")
    parser.add_argument("--batch-size", default="auto", help="lm-eval batch size (default auto)")
    parser.add_argument("--limit", type=int, default=None, help="cap examples per task (smoke runs)")
    parser.add_argument("--out", type=Path, default=Path("evals.json"), help="results JSON path")
    parser.add_argument("--markdown-out", type=Path, default=None, help="also write the markdown table")
    parser.add_argument(
        "--humaneval-harness",
        type=Path,
        default=None,
        help="bigcode-evaluation-harness checkout; when set, HumanEval is run and merged",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    if not tasks:
        print("error: no tasks given", file=sys.stderr)
        return 2

    results = run_lm_eval(
        args.model_path,
        tasks=tasks,
        backend=args.backend,
        model_args=args.model_args,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    if args.humaneval_harness is not None:
        results.update(run_humaneval(args.model_path, args.humaneval_harness))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(results_to_markdown(results), encoding="utf-8")

    print(results_to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
