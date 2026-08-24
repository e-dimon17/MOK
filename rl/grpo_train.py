"""TRL GRPOTrainer harness for RLVR, RL phase 2 (`mok-rl`).

Method (playbook): GRPO-style RL on verifiable tasks only — math answers
checked by sympy equivalence, code checked by sandboxed execution. Group size
8, LR 1e-6, KL penalty against the DPO checkpoint, rollouts served by
external vLLM servers (TRL server mode; endpoint list in rl/configs/grpo.yaml,
one `trl vllm-serve`-compatible server per rollout node, each having
registered the MoK architecture via `rl.vllm_plugin.register_mok_moe`).

`reward_router` is the single dispatch point: an RLVR sample carries a task
tag ("math" -> `verify_math` against `reference_answer`; "code" ->
`verify_code` against `tests`). `make_trl_reward_fn` adapts it to TRL's
reward-function calling convention (batched columns as kwargs).

Heavy deps (transformers/trl/datasets) import lazily inside `run`; settings
parsing, routing, and dataset assembly are pure and CPU-unit-tested.
"""

from __future__ import annotations

import argparse
import os
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mok_core.config.loader import load_yaml
from rl.data.rlvr_code import build_code_prompts
from rl.data.rlvr_math import build_math_prompts
from rl.rewards.code_reward import SandboxMode, verify_code
from rl.rewards.math_reward import verify_math

# --------------------------------------------------------------------------- #
# Settings (pure YAML -> dataclass; rl/configs/grpo.yaml)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VllmSettings:
    server_mode: bool = True
    endpoints: tuple[str, ...] = ("http://127.0.0.1:8000",)

    def __post_init__(self) -> None:
        if self.server_mode and not self.endpoints:
            raise ValueError("vllm server mode needs at least one endpoint")
        for endpoint in self.endpoints:
            parsed = urlparse(endpoint)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError(f"bad vllm endpoint {endpoint!r}; expected http(s)://host[:port]")


@dataclass(frozen=True)
class GRPOSettings:
    model_dir: str
    output_dir: str
    seed: int = 42
    group_size: int = 8         # completions per prompt (num_generations)
    lr: float = 1e-6
    kl_coef: float = 0.04       # GRPO beta — KL penalty vs the frozen reference
    bf16: bool = True
    micro_batch_size: int = 8   # completions per device per forward
    grad_accum: int = 8
    max_prompt_length: int = 1024
    max_completion_length: int = 2048
    temperature: float = 1.0
    epochs: float = 1.0
    logging_steps: int = 10
    save_steps: int = 100
    max_prompts: int | None = None  # cap for smoke runs
    code_timeout_s: float = 6.0
    code_sandbox: SandboxMode = "auto"
    vllm: VllmSettings = field(default_factory=VllmSettings)
    datasets: dict[str, Any] = field(default_factory=dict)


def load_settings(path: str | Path) -> GRPOSettings:
    data = load_yaml(path)
    vllm_raw = data.pop("vllm", {}) or {}
    endpoints = vllm_raw.get("endpoints")
    vllm = VllmSettings(
        server_mode=bool(vllm_raw.get("server_mode", True)),
        endpoints=tuple(endpoints) if endpoints else VllmSettings.endpoints,
    )
    return GRPOSettings(vllm=vllm, **data)


def endpoint_for_rank(settings: GRPOSettings, rank: int | None = None) -> str:
    """Training ranks round-robin over the rollout servers."""
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    endpoints = settings.vllm.endpoints
    return endpoints[rank % len(endpoints)]


# --------------------------------------------------------------------------- #
# Reward routing
# --------------------------------------------------------------------------- #


def reward_router(
    sample: Mapping[str, Any],
    completion: str,
    *,
    code_timeout_s: float = 6.0,
    code_sandbox: SandboxMode = "auto",
) -> float:
    """Dispatch one (sample, completion) pair to its task verifier by tag."""
    tag = sample.get("tag")
    if tag == "math":
        return verify_math(completion, str(sample["reference_answer"]))
    if tag == "code":
        return verify_code(
            completion, list(sample["tests"]), timeout_s=code_timeout_s, sandbox=code_sandbox
        )
    raise ValueError(f"unknown task tag {tag!r}; expected 'math' or 'code'")


def _completion_text(completion: Any) -> str:
    """TRL passes either a plain string (standard format) or a message list
    (conversational format) per completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)):
        return "".join(
            str(m.get("content", "")) for m in completion if isinstance(m, Mapping)
        )
    raise TypeError(f"unsupported completion type {type(completion).__name__}")


def make_trl_reward_fn(
    *, code_timeout_s: float = 6.0, code_sandbox: SandboxMode = "auto"
) -> Callable[..., list[float]]:
    """Adapter to TRL's reward interface: `fn(prompts, completions, **cols)`
    with every extra dataset column passed as an equally-long list. Rebuilds
    the per-row sample dicts and routes each through `reward_router`."""

    def rlvr_reward(
        prompts: list[Any] | None = None, completions: list[Any] | None = None, **cols: Any
    ) -> list[float]:
        if completions is None:
            raise ValueError("TRL reward function called without completions")
        n = len(completions)
        columns = {
            key: values
            for key, values in cols.items()
            if isinstance(values, (list, tuple)) and len(values) == n
        }
        rewards: list[float] = []
        for i, completion in enumerate(completions):
            sample = {key: values[i] for key, values in columns.items()}
            rewards.append(
                reward_router(
                    sample,
                    _completion_text(completion),
                    code_timeout_s=code_timeout_s,
                    code_sandbox=code_sandbox,
                )
            )
        return rewards

    rlvr_reward.__name__ = "rlvr_reward"
    return rlvr_reward


# --------------------------------------------------------------------------- #
# Dataset assembly (uniform schema across tags; deterministic order)
# --------------------------------------------------------------------------- #


def build_rlvr_dataset(
    settings: GRPOSettings,
    *,
    gsm8k_rows: Iterable[Mapping[str, Any]] | None = None,
    math_rows: Iterable[Mapping[str, Any]] | None = None,
    mbpp_rows: Iterable[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Math + code prompt mixture per the settings' dataset map, shuffled
    deterministically by seed and padded to a uniform column schema
    {prompt, tag, reference_answer, tests} (HF Dataset needs one schema)."""
    spec = settings.datasets
    items: list[dict[str, Any]] = []
    math_spec = spec.get("math")
    if math_spec:
        items.extend(
            build_math_prompts(
                n=math_spec.get("n"), seed=settings.seed, gsm8k_rows=gsm8k_rows, math_rows=math_rows
            )
        )
    code_spec = spec.get("code")
    if code_spec:
        items.extend(
            build_code_prompts(
                n=code_spec.get("n"),
                seed=settings.seed,
                mbpp_rows=mbpp_rows,
                selfgen_path=code_spec.get("selfgen_jsonl"),
            )
        )
    uniform = [
        {
            "prompt": item["prompt"],
            "tag": item["tag"],
            "reference_answer": item.get("reference_answer", ""),
            "tests": item.get("tests", []),
        }
        for item in items
    ]
    random.Random(settings.seed).shuffle(uniform)
    if settings.max_prompts is not None:
        uniform = uniform[: settings.max_prompts]
    return uniform


# --------------------------------------------------------------------------- #
# Trainer entry point (all heavy imports local)
# --------------------------------------------------------------------------- #


def run(settings: GRPOSettings) -> None:
    import dataclasses  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import PreTrainedTokenizerFast, set_seed  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    from sft.hf_model.modeling_mok_moe import MokMoeForCausalLM  # noqa: PLC0415

    set_seed(settings.seed)

    model = MokMoeForCausalLM.from_pretrained(
        settings.model_dir, dtype=torch.bfloat16 if settings.bf16 else torch.float32
    )
    model.config.use_cache = False
    tokenizer = PreTrainedTokenizerFast.from_pretrained(settings.model_dir)

    items = build_rlvr_dataset(settings)
    if not items:
        raise RuntimeError("RLVR prompt set is empty — check the datasets section of grpo.yaml")
    train_ds = Dataset.from_list(items)

    endpoint = endpoint_for_rank(settings)
    parsed = urlparse(endpoint)
    config_kwargs: dict[str, Any] = {
        "output_dir": settings.output_dir,
        "num_train_epochs": settings.epochs,
        "per_device_train_batch_size": settings.micro_batch_size,
        "gradient_accumulation_steps": settings.grad_accum,
        "learning_rate": settings.lr,
        "beta": settings.kl_coef,
        "num_generations": settings.group_size,
        "max_prompt_length": settings.max_prompt_length,
        "max_completion_length": settings.max_completion_length,
        "temperature": settings.temperature,
        "bf16": settings.bf16,
        "gradient_checkpointing": True,
        "logging_steps": settings.logging_steps,
        "save_steps": settings.save_steps,
        "save_total_limit": 5,
        "seed": settings.seed,
        "report_to": [],
        "use_vllm": settings.vllm.server_mode,
        # TRL vllm server mode; both spellings supplied, filtered to whatever
        # the installed TRL version's GRPOConfig actually declares.
        "vllm_mode": "server",
        "vllm_server_base_url": endpoint,
        "vllm_server_host": parsed.hostname,
        "vllm_server_port": parsed.port or (443 if parsed.scheme == "https" else 80),
    }
    allowed = {f.name for f in dataclasses.fields(GRPOConfig)}
    args = GRPOConfig(**{k: v for k, v in config_kwargs.items() if k in allowed})

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            make_trl_reward_fn(
                code_timeout_s=settings.code_timeout_s, code_sandbox=settings.code_sandbox
            )
        ],
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(Path(settings.output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(settings.output_dir) / "final"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-rl", description="RLVR (GRPO) tune MoK-54B-chat with TRL + vLLM rollouts."
    )
    parser.add_argument("--config", type=Path, default=Path("rl/configs/grpo.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(load_settings(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
