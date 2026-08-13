"""TRL SFTTrainer harness for MoK-54B-instruct (`mok-sft` console script).

Method (playbook step F): 2 epochs over Tulu-3 + OpenHermes-2.5 + reasoning
traces, LR 5e-6 -> 1e-5 warmup then cosine to 0, seq 16384 with packing,
bf16, FSDP2 via `accelerate launch --config_file F/configs/fsdp2.yaml`.
Everything heavy (transformers/trl/datasets) imports lazily inside `main`, so
this module is importable on any host (config parsing and the LR schedule are
pure and unit-tested).

Launch:
    accelerate launch --config_file F/configs/fsdp2.yaml -m F.sft_train \
        --config F/configs/sft.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from F import data_prep
from mok_core.config.loader import load_yaml

# --------------------------------------------------------------------------- #
# Settings (pure YAML -> dataclass; F/configs/sft.yaml)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LRSettings:
    start: float = 5e-6      # warmup starts here (never a dead 0-LR step)
    peak: float = 1e-5
    warmup_ratio: float = 0.03


@dataclass(frozen=True)
class SFTSettings:
    model_dir: str
    output_dir: str
    seed: int = 42
    epochs: float = 2.0
    seq_len: int = 16384
    micro_batch_size: int = 1
    grad_accum: int = 8
    lr: LRSettings = field(default_factory=LRSettings)
    bf16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 200
    eval_steps: int = 200
    logging_steps: int = 10
    eval_holdout_examples: int = 512
    max_examples: int | None = None       # cap for smoke runs
    datasets: dict[str, Any] = field(default_factory=dict)
    eval_ngrams_path: str | None = None
    ngram_n: int = 13
    fsdp_config: str = "F/configs/fsdp2.yaml"


def load_settings(path: str | Path) -> SFTSettings:
    data = load_yaml(path)
    lr_raw = data.pop("lr", {})
    decon = data.pop("decontamination", {}) or {}
    return SFTSettings(
        lr=LRSettings(**lr_raw),
        eval_ngrams_path=decon.get("eval_ngrams_path"),
        ngram_n=int(decon.get("ngram_n", 13)),
        **data,
    )


# --------------------------------------------------------------------------- #
# LR schedule: linear 5e-6 -> 1e-5 warmup, cosine 1e-5 -> 0 (pure, tested)
# --------------------------------------------------------------------------- #


def cosine_warmup_lambda(
    *, start_lr: float, peak_lr: float, warmup_steps: int, total_steps: int
) -> Callable[[int], float]:
    """Multiplier for LambdaLR on an optimizer whose base lr == peak_lr.

    step in [0, warmup): linear start_lr -> peak_lr (start_lr at step 0);
    step in [warmup, total]: cosine peak_lr -> 0; clamped at 0 beyond.
    """
    if not 0 < start_lr <= peak_lr:
        raise ValueError(f"need 0 < start_lr <= peak_lr, got {start_lr} / {peak_lr}")
    if total_steps <= warmup_steps or warmup_steps < 0:
        raise ValueError(f"need 0 <= warmup_steps < total_steps, got {warmup_steps} / {total_steps}")

    def factor(step: int) -> float:
        if step < warmup_steps:
            frac = step / warmup_steps
            return (start_lr + (peak_lr - start_lr) * frac) / peak_lr
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def build_mixture(settings: SFTSettings) -> Iterable[dict]:
    """Yield normalized {"messages": ...} examples per the settings' dataset map.

    Keys: `tulu3`/`openhermes` (truthy enables; loaders stream from the Hub)
    and `reasoning_traces` (path to a local JSONL).
    """
    spec = settings.datasets
    if spec.get("tulu3"):
        yield from data_prep.tulu3()
    if spec.get("openhermes"):
        yield from data_prep.openhermes()
    traces = spec.get("reasoning_traces")
    if traces:
        yield from data_prep.reasoning_traces(traces)


def load_eval_ngrams(path: str | Path, n: int) -> set[str]:
    """Eval-suite n-gram bank from a JSON list of strings or a plain-text file
    (one document per line)."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        docs = json.loads(text)
    except json.JSONDecodeError:
        docs = text.splitlines()
    if not isinstance(docs, list):
        raise TypeError(f"{path}: expected a JSON list of strings or plain text lines")
    return data_prep.build_eval_ngrams((str(d) for d in docs), n)


# --------------------------------------------------------------------------- #
# Trainer entry point (all heavy imports local)
# --------------------------------------------------------------------------- #


def run(settings: SFTSettings) -> None:
    import torch  # noqa: PLC0415
    from datasets import Dataset  # noqa: PLC0415
    from transformers import PreTrainedTokenizerFast, set_seed  # noqa: PLC0415
    from trl import SFTConfig, SFTTrainer  # noqa: PLC0415

    from F.hf_model.modeling_mok_moe import MokMoeForCausalLM  # noqa: PLC0415

    set_seed(settings.seed)

    model = MokMoeForCausalLM.from_pretrained(
        settings.model_dir, dtype=torch.bfloat16 if settings.bf16 else torch.float32
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    tokenizer = PreTrainedTokenizerFast.from_pretrained(settings.model_dir)

    examples: list[dict] = []
    for i, example in enumerate(build_mixture(settings)):
        if settings.max_examples is not None and i >= settings.max_examples:
            break
        examples.append(example)
    if settings.eval_ngrams_path:
        bank = load_eval_ngrams(settings.eval_ngrams_path, settings.ngram_n)
        examples = data_prep.decontaminate(examples, bank, settings.ngram_n)
    if not examples:
        raise RuntimeError("SFT mixture is empty — check the datasets section of sft.yaml")

    rendered = []
    for example in examples:
        input_ids, labels = data_prep.render_chat(example["messages"], tokenizer)
        rendered.append({"input_ids": input_ids, "labels": labels})
    holdout = min(settings.eval_holdout_examples, max(1, len(rendered) // 20))
    train_ds = Dataset.from_list(rendered[holdout:])
    eval_ds = Dataset.from_list(rendered[:holdout])
    collator = data_prep.SFTPackCollator(seq_len=settings.seq_len, pad_id=model.config.pad_token_id)

    args = SFTConfig(
        output_dir=settings.output_dir,
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.micro_batch_size,
        per_device_eval_batch_size=settings.micro_batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        bf16=settings.bf16,
        gradient_checkpointing=settings.gradient_checkpointing,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        eval_strategy="steps",
        eval_steps=settings.eval_steps,
        save_total_limit=10,
        seed=settings.seed,
        max_length=None,          # our collator owns truncation/packing
        packing=False,            # ditto (SFTPackCollator packs to seq_len)
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        report_to=[],
    )

    # Custom schedule (5e-6 -> 1e-5 warmup, cosine to 0) via explicit optimizer.
    world = int(torch.distributed.get_world_size()) if torch.distributed.is_initialized() else 1
    steps_per_epoch = max(
        1, math.ceil(len(train_ds) / (settings.micro_batch_size * settings.grad_accum * world))
    )
    total_steps = max(2, int(steps_per_epoch * settings.epochs))
    warmup_steps = int(total_steps * settings.lr.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.lr.peak, betas=(0.9, 0.95), weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        cosine_warmup_lambda(
            start_lr=settings.lr.start,
            peak_lr=settings.lr.peak,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        ),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
        optimizers=(optimizer, scheduler),
    )
    trainer.train()
    trainer.save_model(str(Path(settings.output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(settings.output_dir) / "final"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mok-sft", description="SFT MoK-54B with TRL (step F).")
    parser.add_argument("--config", type=Path, default=Path("F/configs/sft.yaml"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(load_settings(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
