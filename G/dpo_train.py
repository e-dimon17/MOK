"""TRL DPOTrainer harness for MoK-54B-chat, phase 1 of step G.

Method (playbook): DPO on the Tulu-3-preference + UltraFeedback mixture,
1 epoch, LR 5e-7, beta 0.1, bf16; the reference model is the SFT checkpoint
(which is also the policy init). Config: G/configs/dpo.yaml.

Preference rows are normalized to explicit-string `{prompt, chosen, rejected}`
via step F's ChatML chat template (`F.data_prep.CHAT_TEMPLATE`): the prompt
ends with the generation header `<|im_start|>assistant\\n` and each completion
is `content<|im_end|>\\n` — byte-identical to how the SFT model was trained
and to what `add_generation_prompt=True` produces at inference.

All heavy deps (transformers/trl/datasets) import lazily inside `run`; the
settings parsing and preference normalization are pure and CPU-unit-tested.

Launch:
    accelerate launch --config_file F/configs/fsdp2.yaml -m G.dpo_train \
        --config G/configs/dpo.yaml
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from F.data_prep import IM_END, IM_START, VALID_ROLES
from mok_core.config.loader import load_yaml

TULU3_PREF_DATASET = "allenai/llama-3.1-tulu-3-8b-preference-mixture"
ULTRAFEEDBACK_DATASET = "HuggingFaceH4/ultrafeedback_binarized"
ULTRAFEEDBACK_SPLIT = "train_prefs"


# --------------------------------------------------------------------------- #
# Settings (pure YAML -> dataclass; G/configs/dpo.yaml)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DPOSettings:
    model_dir: str
    output_dir: str
    ref_model_dir: str | None = None  # None -> the SFT checkpoint itself (model_dir)
    seed: int = 42
    epochs: float = 1.0
    lr: float = 5e-7
    beta: float = 0.1
    bf16: bool = True
    micro_batch_size: int = 1
    grad_accum: int = 16
    max_prompt_length: int = 2048
    max_length: int = 4096
    logging_steps: int = 10
    save_steps: int = 500
    eval_holdout_examples: int = 512
    max_examples: int | None = None  # cap for smoke runs
    datasets: dict[str, Any] = field(default_factory=dict)


def load_settings(path: str | Path) -> DPOSettings:
    return DPOSettings(**load_yaml(path))


# --------------------------------------------------------------------------- #
# Chat-template rendering (mirrors F.data_prep.CHAT_TEMPLATE segment-for-segment)
# --------------------------------------------------------------------------- #


def render_messages(messages: Sequence[Mapping[str, str]]) -> str:
    """The template body: one `<|im_start|>role\\ncontent<|im_end|>\\n` per turn."""
    return "".join(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n" for m in messages)


def render_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    """Prompt half: rendered turns + the generation header (the template's
    `add_generation_prompt=True` tail)."""
    return render_messages(messages) + f"{IM_START}assistant\n"


def render_completion(content: str) -> str:
    """Completion half: closes the assistant turn opened by `render_prompt`."""
    return f"{content}{IM_END}\n"


# --------------------------------------------------------------------------- #
# Preference normalization -> {"prompt", "chosen", "rejected"} strings
# --------------------------------------------------------------------------- #


def _clean_conversation(raw: Any) -> list[dict[str, str]] | None:
    """A preference-side conversation: valid roles, str contents, and the last
    turn is the assistant response being judged. None if unusable."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    out: list[dict[str, str]] = []
    for message in raw:
        if not isinstance(message, Mapping):
            return None
        role, content = message.get("role"), message.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            return None
        out.append({"role": role, "content": content})
    if out[-1]["role"] != "assistant" or len(out) < 2:
        return None
    return out


def normalize_preference_row(row: Mapping[str, Any]) -> dict[str, str] | None:
    """One raw preference row -> template-rendered strings (None if unusable).

    Accepted shapes (covers Tulu-3-preference and UltraFeedback-binarized):
      - {"chosen": [msgs..., assistant], "rejected": [msgs..., assistant]}
        (each side carries the shared prompt turns; prompt := chosen[:-1])
      - {"prompt": str, "chosen": str, "rejected": str} (bare single-turn)
    """
    chosen_raw, rejected_raw = row.get("chosen"), row.get("rejected")
    if isinstance(chosen_raw, str) and isinstance(rejected_raw, str):
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return None
        if not chosen_raw.strip() or not rejected_raw.strip():
            return None
        prompt_messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        chosen_content, rejected_content = chosen_raw, rejected_raw
    else:
        chosen = _clean_conversation(chosen_raw)
        rejected = _clean_conversation(rejected_raw)
        if chosen is None or rejected is None:
            return None
        prompt_messages = chosen[:-1]
        chosen_content, rejected_content = chosen[-1]["content"], rejected[-1]["content"]
    return {
        "prompt": render_prompt(prompt_messages),
        "chosen": render_completion(chosen_content),
        "rejected": render_completion(rejected_content),
    }


def normalize_preferences(ds: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, str]]:
    """Normalize a raw preference dataset, silently skipping unusable rows."""
    for row in ds:
        example = normalize_preference_row(row)
        if example is not None:
            yield example


def build_preference_mixture(
    settings: DPOSettings,
    *,
    tulu_rows: Iterable[Mapping[str, Any]] | None = None,
    ultrafeedback_rows: Iterable[Mapping[str, Any]] | None = None,
) -> Iterator[dict[str, str]]:
    """Normalized examples per the settings' dataset map. Keys:
    `tulu3_preference` / `ultrafeedback` (truthy enables; row args inject)."""
    spec = settings.datasets
    if spec.get("tulu3_preference"):
        if tulu_rows is None:
            from datasets import load_dataset  # noqa: PLC0415 — lazy, [post] extra

            tulu_rows = load_dataset(TULU3_PREF_DATASET, split="train")
        yield from normalize_preferences(tulu_rows)
    if spec.get("ultrafeedback"):
        if ultrafeedback_rows is None:
            from datasets import load_dataset  # noqa: PLC0415

            ultrafeedback_rows = load_dataset(ULTRAFEEDBACK_DATASET, split=ULTRAFEEDBACK_SPLIT)
        yield from normalize_preferences(ultrafeedback_rows)


# --------------------------------------------------------------------------- #
# Trainer entry point (all heavy imports local)
# --------------------------------------------------------------------------- #


def run(settings: DPOSettings) -> None:
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import PreTrainedTokenizerFast, set_seed  # noqa: PLC0415
    from trl import DPOConfig, DPOTrainer  # noqa: PLC0415

    from F.hf_model.modeling_mok_moe import MokMoeForCausalLM  # noqa: PLC0415

    set_seed(settings.seed)
    dtype = torch.bfloat16 if settings.bf16 else torch.float32

    model = MokMoeForCausalLM.from_pretrained(settings.model_dir, dtype=dtype)
    model.config.use_cache = False
    # Reference model = the SFT checkpoint (playbook); explicit so a later
    # DPO round can keep referencing SFT rather than its own init.
    ref_model = MokMoeForCausalLM.from_pretrained(
        settings.ref_model_dir or settings.model_dir, dtype=dtype
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(settings.model_dir)

    examples: list[dict[str, str]] = []
    for i, example in enumerate(build_preference_mixture(settings)):
        if settings.max_examples is not None and i >= settings.max_examples:
            break
        examples.append(example)
    if not examples:
        raise RuntimeError("preference mixture is empty — check the datasets section of dpo.yaml")

    holdout = min(settings.eval_holdout_examples, max(1, len(examples) // 20))
    train_ds = Dataset.from_list(examples[holdout:])
    eval_ds = Dataset.from_list(examples[:holdout])

    args = DPOConfig(
        output_dir=settings.output_dir,
        beta=settings.beta,
        learning_rate=settings.lr,
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.micro_batch_size,
        per_device_eval_batch_size=settings.micro_batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        bf16=settings.bf16,
        gradient_checkpointing=True,
        max_prompt_length=settings.max_prompt_length,
        max_length=settings.max_length,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        eval_strategy="steps",
        eval_steps=settings.save_steps,
        save_total_limit=5,
        seed=settings.seed,
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(Path(settings.output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(settings.output_dir) / "final"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-dpo", description="DPO-align MoK-54B-instruct with TRL (step G, phase 1)."
    )
    parser.add_argument("--config", type=Path, default=Path("G/configs/dpo.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(load_settings(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
