"""SFT: sft.yaml parsing, LR schedule math, fsdp2 config sanity (no TRL import)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from sft.sft_train import (
    SFTSettings,
    build_mixture,
    cosine_warmup_lambda,
    load_eval_ngrams,
    load_settings,
)

REPO = Path(__file__).resolve().parents[2]


def test_load_settings_parses_shipped_yaml() -> None:
    settings = load_settings(REPO / "sft" / "configs" / "sft.yaml")
    assert settings.epochs == 2
    assert settings.seq_len == 16384
    assert settings.lr.start == pytest.approx(5e-6)
    assert settings.lr.peak == pytest.approx(2e-5)  # ST-MoE sparse-FT direction (sft.yaml)
    assert settings.bf16 is True
    assert settings.gradient_checkpointing is True
    assert settings.save_steps == 200 and settings.eval_steps == 200
    assert settings.datasets["tulu3"] is True
    assert settings.ngram_n == 13 and settings.eval_ngrams_path is None
    assert settings.fsdp_config == "sft/configs/fsdp2.yaml"
    # Global batch ~0.5M tokens on the 8-rank node (ST-MoE: sparse models
    # prefer smaller batch + higher LR than the dense-tuned ~1M default).
    assert 8 * settings.micro_batch_size * settings.grad_accum * settings.seq_len >= 500_000


def test_fsdp2_yaml_references_our_layer_class() -> None:
    with open(REPO / "sft" / "configs" / "fsdp2.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["distributed_type"] == "FSDP"
    assert cfg["mixed_precision"] == "bf16"
    assert cfg["num_processes"] == 8
    assert cfg["fsdp_config"]["fsdp_version"] == 2
    assert cfg["fsdp_config"]["fsdp_transformer_layer_cls_to_wrap"] == "MokMoeDecoderLayer"


def test_cosine_warmup_lambda_shape() -> None:
    factor = cosine_warmup_lambda(start_lr=5e-6, peak_lr=1e-5, warmup_steps=10, total_steps=100)
    peak = 1e-5
    assert factor(0) * peak == pytest.approx(5e-6)            # starts at 5e-6, no dead step
    assert factor(5) * peak == pytest.approx(7.5e-6)          # linear midpoint
    assert factor(10) == pytest.approx(1.0)                   # peak at warmup end
    assert factor(55) == pytest.approx(0.5)                   # cosine midpoint
    assert factor(100) == pytest.approx(0.0)
    assert factor(1000) == pytest.approx(0.0)                 # clamped past the end
    values = [factor(s) for s in range(101)]
    rising, falling = values[:11], values[10:]
    assert all(a < b for a, b in zip(rising, rising[1:], strict=False))
    assert all(a >= b for a, b in zip(falling, falling[1:], strict=False))


def test_cosine_warmup_lambda_validation() -> None:
    with pytest.raises(ValueError):
        cosine_warmup_lambda(start_lr=2e-5, peak_lr=1e-5, warmup_steps=1, total_steps=10)
    with pytest.raises(ValueError):
        cosine_warmup_lambda(start_lr=5e-6, peak_lr=1e-5, warmup_steps=10, total_steps=10)


def test_build_mixture_empty_spec_yields_nothing() -> None:
    settings = SFTSettings(model_dir="x", output_dir="y", datasets={})
    assert list(build_mixture(settings)) == []
    # Nothing above may have imported the heavy post-training stack.
    assert "trl" not in sys.modules
    assert "datasets" not in sys.modules


def test_load_eval_ngrams_json_and_text(tmp_path) -> None:
    doc = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
    json_path = tmp_path / "eval.json"
    json_path.write_text(json.dumps([doc]))
    text_path = tmp_path / "eval.txt"
    text_path.write_text(doc + "\n")
    assert load_eval_ngrams(json_path, 13) == load_eval_ngrams(text_path, 13)
    assert len(load_eval_ngrams(json_path, 13)) == 2  # 14 tokens -> two 13-grams
