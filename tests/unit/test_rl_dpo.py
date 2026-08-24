"""RL: dpo.yaml parsing + preference normalization via the sft chat template."""

from __future__ import annotations

import sys
from pathlib import Path

import jinja2
import pytest

from rl.dpo_train import (
    DPOSettings,
    build_preference_mixture,
    load_settings,
    normalize_preference_row,
    normalize_preferences,
    render_completion,
    render_messages,
    render_prompt,
)
from sft.data_prep import CHAT_TEMPLATE

REPO = Path(__file__).resolve().parents[2]

CONVERSATION = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What is 2+2?"},
]


def test_load_settings_parses_shipped_yaml() -> None:
    settings = load_settings(REPO / "rl" / "configs" / "dpo.yaml")
    assert settings.epochs == 1
    assert settings.lr == pytest.approx(5e-7)
    assert settings.beta == pytest.approx(0.1)
    assert settings.bf16 is True
    assert settings.ref_model_dir == settings.model_dir  # ref = SFT checkpoint
    assert settings.datasets["tulu3_preference"] is True
    assert settings.datasets["ultrafeedback"] is True
    assert settings.max_examples is None
    assert settings.max_length >= settings.max_prompt_length


# --------------------------------------------------------------------------- #
# Rendering == the F chat template, byte for byte
# --------------------------------------------------------------------------- #


def test_render_matches_f_chat_template_jinja() -> None:
    template = jinja2.Template(CHAT_TEMPLATE)
    full = [*CONVERSATION, {"role": "assistant", "content": "4"}]
    assert render_messages(full) == template.render(messages=full, add_generation_prompt=False)
    assert render_prompt(CONVERSATION) == template.render(
        messages=CONVERSATION, add_generation_prompt=True
    )


def test_prompt_plus_completion_reconstructs_conversation() -> None:
    full = [*CONVERSATION, {"role": "assistant", "content": "4"}]
    assert render_prompt(CONVERSATION) + render_completion("4") == render_messages(full)


def test_render_prompt_literal() -> None:
    assert render_prompt([{"role": "user", "content": "hi"}]) == (
        "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert render_completion("yo") == "yo<|im_end|>\n"


# --------------------------------------------------------------------------- #
# Normalization shapes
# --------------------------------------------------------------------------- #


def test_normalize_message_list_row() -> None:
    row = {
        "chosen": [*CONVERSATION, {"role": "assistant", "content": "4"}],
        "rejected": [*CONVERSATION, {"role": "assistant", "content": "5"}],
    }
    example = normalize_preference_row(row)
    assert example is not None
    assert example["prompt"] == render_prompt(CONVERSATION)
    assert example["chosen"] == "4<|im_end|>\n"
    assert example["rejected"] == "5<|im_end|>\n"


def test_normalize_bare_string_row() -> None:
    row = {"prompt": "What is 2+2?", "chosen": "4", "rejected": "5"}
    example = normalize_preference_row(row)
    assert example is not None
    assert example["prompt"] == "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    assert example["chosen"] == "4<|im_end|>\n"
    assert example["rejected"] == "5<|im_end|>\n"


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"chosen": "4", "rejected": "5"},                                   # string form, no prompt
        {"prompt": "  ", "chosen": "4", "rejected": "5"},                   # blank prompt
        {"chosen": [{"role": "user", "content": "q"}], "rejected": []},     # no assistant turn
        {
            "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            "rejected": [{"role": "user", "content": "q"}],                 # rejected not assistant-final
        },
        {
            "chosen": [{"role": "alien", "content": "q"}, {"role": "assistant", "content": "a"}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}],
        },
        {
            "chosen": [{"role": "user", "content": ""}, {"role": "assistant", "content": "a"}],
            "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "b"}],
        },
    ],
)
def test_normalize_rejects_unusable_rows(row: dict) -> None:
    assert normalize_preference_row(row) is None


def test_normalize_preferences_skips_bad_rows() -> None:
    rows = [
        {"prompt": "q", "chosen": "a", "rejected": "b"},
        {"bogus": True},
        {"prompt": "q2", "chosen": "a2", "rejected": "b2"},
    ]
    out = list(normalize_preferences(rows))
    assert len(out) == 2
    assert all(set(example) == {"prompt", "chosen", "rejected"} for example in out)


def test_build_preference_mixture_respects_spec_and_stays_light() -> None:
    tulu = [{"prompt": "t", "chosen": "a", "rejected": "b"}]
    uf = [{"prompt": "u", "chosen": "c", "rejected": "d"}]
    both = DPOSettings(
        model_dir="x", output_dir="y", datasets={"tulu3_preference": True, "ultrafeedback": True}
    )
    out = list(build_preference_mixture(both, tulu_rows=tulu, ultrafeedback_rows=uf))
    assert [example["prompt"] for example in out] == [render_prompt([{"role": "user", "content": "t"}]),
                                                      render_prompt([{"role": "user", "content": "u"}])]
    only_uf = DPOSettings(model_dir="x", output_dir="y", datasets={"ultrafeedback": True})
    out = list(build_preference_mixture(only_uf, tulu_rows=tulu, ultrafeedback_rows=uf))
    assert len(out) == 1
    none = DPOSettings(model_dir="x", output_dir="y", datasets={})
    assert list(build_preference_mixture(none)) == []
    # Nothing above may have pulled in the heavy post-training stack.
    assert "trl" not in sys.modules
    assert "datasets" not in sys.modules
