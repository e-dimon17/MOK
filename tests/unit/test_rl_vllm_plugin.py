"""RL: vLLM plugin — pure name-mapping tables + import-guard behavior.

The model class itself needs vllm (absent in CI); these tests pin everything
that does NOT: the HF->vLLM weight-name mapping (proven against the real
parameter names of a tiny sft/hf_model checkpoint) and the lazy guard that
raises a clear 'vllm>=0.8 required' error instead of a bare ModuleNotFound.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

import rl.vllm_plugin.mok_moe_vllm as plugin
from rl.vllm_plugin.mok_moe_vllm import (
    HF_ARCHITECTURE,
    STACKED_PARAMS_MAPPING,
    VLLM_CLASS_PATH,
    expert_params_mapping,
    is_routed_expert_weight,
    map_dense_name,
    register_mok_moe,
    vllm_param_of,
)
from sft.hf_model.configuration_mok_moe import MokMoeConfig
from sft.hf_model.modeling_mok_moe import MokMoeForCausalLM

VLLM_INSTALLED = importlib.util.find_spec("vllm") is not None

NUM_EXPERTS = 8


def tiny_hf_model() -> MokMoeForCausalLM:
    torch.manual_seed(0)
    return MokMoeForCausalLM(
        MokMoeConfig(
            vocab_size=64,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            intermediate_size=8,
            num_dense_layers=1,
            dense_intermediate_size=24,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=2,
            max_position_embeddings=32,
        )
    )


# --------------------------------------------------------------------------- #
# Name-mapping tables (pure)
# --------------------------------------------------------------------------- #


def test_stacked_mapping_goldens() -> None:
    assert map_dense_name("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.qkv_proj.weight",
        "q",
    )
    assert map_dense_name("model.layers.3.self_attn.v_proj.weight") == (
        "model.layers.3.self_attn.qkv_proj.weight",
        "v",
    )
    assert map_dense_name("model.layers.1.mlp.shared_experts.gate_proj.weight") == (
        "model.layers.1.mlp.shared_experts.gate_up_proj.weight",
        0,
    )
    assert map_dense_name("model.layers.1.mlp.shared_experts.up_proj.weight") == (
        "model.layers.1.mlp.shared_experts.gate_up_proj.weight",
        1,
    )
    # 1:1 params pass through with no shard id
    for name in (
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.e_score_correction_bias",
    ):
        assert map_dense_name(name) == (name, None)


def test_expert_mapping_table_shape() -> None:
    mapping = expert_params_mapping(NUM_EXPERTS)
    assert len(mapping) == 3 * NUM_EXPERTS
    assert {shard for _, _, _, shard in mapping} == {"w1", "w2", "w3"}
    assert {expert for _, _, expert, _ in mapping} == set(range(NUM_EXPERTS))
    # gate/up stack into w13, down goes to w2 (vLLM FusedMoE convention)
    by_shard = {shard: target for target, _, _, shard in mapping}
    assert by_shard == {"w1": "experts.w13_", "w3": "experts.w13_", "w2": "experts.w2_"}
    with pytest.raises(ValueError):
        expert_params_mapping(0)


def test_expert_ids_do_not_prefix_collide() -> None:
    # experts.1.* must never match experts.13.* (trailing dot in the fragment)
    name = "model.layers.0.mlp.experts.13.gate_proj.weight"
    assert vllm_param_of(name, 16) == "model.layers.0.mlp.experts.w13_weight"
    mapping = expert_params_mapping(16)
    hits = [row for row in mapping if row[1] in name]
    assert len(hits) == 1 and hits[0][2] == 13


def test_routed_vs_shared_expert_classification() -> None:
    assert is_routed_expert_weight("model.layers.0.mlp.experts.0.down_proj.weight")
    assert not is_routed_expert_weight("model.layers.0.mlp.shared_experts.down_proj.weight")
    assert not is_routed_expert_weight("model.layers.0.mlp.gate.weight")


def test_mapping_covers_every_real_checkpoint_tensor() -> None:
    """Every tensor a converted checkpoint actually ships (tiny sft/hf_model
    state dict) must map to exactly the expected vLLM parameter set."""
    state = tiny_hf_model().state_dict()
    mapped = {vllm_param_of(name, NUM_EXPERTS) for name in state}
    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for i in range(2):
        prefix = f"model.layers.{i}"
        expected |= {
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.self_attn.qkv_proj.weight",
            f"{prefix}.self_attn.o_proj.weight",
        }
        if i == 0:  # dense block (num_dense_layers=1): plain SwiGLU, no router/experts
            expected |= {
                f"{prefix}.mlp.gate_up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
            }
        else:
            expected |= {
                f"{prefix}.mlp.gate.weight",
                f"{prefix}.mlp.gate.e_score_correction_bias",
                f"{prefix}.mlp.experts.w13_weight",
                f"{prefix}.mlp.experts.w2_weight",
                f"{prefix}.mlp.shared_experts.gate_up_proj.weight",
                f"{prefix}.mlp.shared_experts.down_proj.weight",
            }
    assert mapped == expected


def test_vllm_param_of_rejects_unknown_expert_weight() -> None:
    with pytest.raises(KeyError):
        vllm_param_of("model.layers.0.mlp.experts.0.mystery_proj.weight", NUM_EXPERTS)


def test_registry_constants() -> None:
    assert HF_ARCHITECTURE == "MokMoeForCausalLM"  # == sft/hf_model class name
    assert MokMoeForCausalLM.__name__ == HF_ARCHITECTURE
    module_path, _, class_name = VLLM_CLASS_PATH.partition(":")
    assert module_path == plugin.__name__
    assert class_name == "MokMoeForCausalLM_vLLM"
    assert len(STACKED_PARAMS_MAPPING) == 5


# --------------------------------------------------------------------------- #
# Import-guard behavior (vllm absent in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(VLLM_INSTALLED, reason="guard behavior is only observable without vllm")
def test_register_raises_clear_error_without_vllm() -> None:
    with pytest.raises(ImportError, match="vllm>=0.8"):
        register_mok_moe()


@pytest.mark.skipif(VLLM_INSTALLED, reason="guard behavior is only observable without vllm")
def test_model_class_access_raises_clear_error_without_vllm() -> None:
    with pytest.raises(ImportError, match="vllm>=0.8"):
        _ = plugin.MokMoeForCausalLM_vLLM
    with pytest.raises(ImportError, match="vllm>=0.8"):
        from rl.vllm_plugin.mok_moe_vllm import MokMoeForCausalLM_vLLM  # noqa: F401


def test_module_getattr_still_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = plugin.no_such_attribute


def test_importing_plugin_never_imports_vllm() -> None:
    # The plugin package (and everything above) is already imported; vllm must
    # not have been dragged in by module import or the pure helpers.
    assert "rl.vllm_plugin.mok_moe_vllm" in sys.modules
    if not VLLM_INSTALLED:
        assert "vllm" not in sys.modules
