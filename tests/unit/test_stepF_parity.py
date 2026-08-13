"""Step F PARITY: mok_core reference model -> remap_state_dict -> HF model.

fp32 end-to-end (bf16 masters upcast exactly), logits must agree to 1e-4 —
the CPU twin of the release parity gate in F/verify_conversion.py.
"""

from __future__ import annotations

import pytest
import torch

from F.convert_dcp_to_hf import ConversionError, remap_state_dict
from F.hf_model.configuration_mok_moe import MokMoeConfig
from F.hf_model.modeling_mok_moe import MokMoeForCausalLM
from mok_core.config import ModelConfig
from mok_core.model import build_reference_model
from mok_core.model.attention import sdpa_backend


def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        num_layers=2,
        hidden_size=256,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=128,
        vocab_size=512,
        seq_len=256,
        num_experts=8,
        top_k=2,
        intermediate_size=256,
        num_dense_layers=1,
        dense_intermediate_size=512,
        ep_size=4,
    )


@pytest.fixture(scope="module")
def parity_pair() -> tuple[torch.nn.Module, MokMoeForCausalLM, ModelConfig]:
    cfg = tiny_cfg()
    reference = build_reference_model(cfg, seed=3)
    # Non-zero selection bias so the DeepSeek-style biased top-k is exercised:
    # selection must follow the bias while gate weights stay unbiased.
    for i, moe in enumerate(reference.moe_layers()):
        with torch.no_grad():
            moe.router.balance_bias.copy_(
                torch.linspace(-0.2, 0.2, cfg.num_experts).roll(i)
            )
    reference = reference.float()  # bf16 -> fp32 upcast is exact
    hf_cfg = MokMoeConfig.from_model_config(cfg)
    hf_sd = remap_state_dict(dict(reference.iter_master_params()), hf_cfg)
    torch.manual_seed(0)
    hf_model = MokMoeForCausalLM(hf_cfg)  # fp32 default dtype
    hf_model.load_state_dict(hf_sd, strict=True)
    reference.eval()
    hf_model.eval()
    return reference, hf_model, cfg


def test_remap_covers_hf_model_exactly(parity_pair) -> None:
    reference, hf_model, cfg = parity_pair
    hf_sd = remap_state_dict(dict(reference.iter_master_params()), hf_model.config)
    model_keys = set(hf_model.state_dict().keys())
    assert set(hf_sd.keys()) == model_keys  # strict both ways


def test_qkv_split_and_bias_mapping(parity_pair) -> None:
    reference, hf_model, cfg = parity_pair
    nq, nkv, hd = cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim
    qkv = dict(reference.iter_master_params())["blocks.0.attn.qkv.weight"]
    attn = hf_model.model.layers[0].self_attn
    torch.testing.assert_close(attn.q_proj.weight, qkv[: nq * hd])
    torch.testing.assert_close(attn.k_proj.weight, qkv[nq * hd : (nq + nkv) * hd])
    torch.testing.assert_close(attn.v_proj.weight, qkv[(nq + nkv) * hd :])
    torch.testing.assert_close(
        hf_model.model.layers[1].mlp.gate.e_score_correction_bias,
        reference.blocks[1].moe.router.balance_bias,
    )
    torch.testing.assert_close(
        hf_model.model.layers[1].mlp.experts[5].down_proj.weight,
        reference.blocks[1].moe.routed_down[5],
    )


def test_logit_parity_fp32(parity_pair) -> None:
    reference, hf_model, cfg = parity_pair
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randint(0, cfg.vocab_size, (2, 24), generator=generator)
    with torch.no_grad(), sdpa_backend():
        ref_logits = reference(tokens).logits
        hf_logits = hf_model(input_ids=tokens).logits
    assert ref_logits.dtype == hf_logits.dtype == torch.float32
    torch.testing.assert_close(hf_logits, ref_logits, atol=1e-4, rtol=1e-4)


def test_router_selection_parity(parity_pair) -> None:
    """Biased top-k selection identical layer by layer (not just final logits)."""
    reference, hf_model, cfg = parity_pair
    generator = torch.Generator().manual_seed(12)
    tokens = torch.randint(0, cfg.vocab_size, (1, 16), generator=generator)
    with torch.no_grad(), sdpa_backend():
        ref_out = reference(tokens)
    hidden = ref_out.loss_inputs  # per-layer RouterOutput from the reference forward
    # Replay the HF gate on the same normed activations by re-running the HF
    # model and comparing dispatch counts via the final logits path is implicit;
    # here we check the gate math directly on a shared random activation.
    x = torch.randn(32, cfg.hidden_size, generator=generator)
    ref_route = reference.blocks[1].moe.router(x)
    hf_weights, hf_experts = hf_model.model.layers[1].mlp.gate(x)
    assert torch.equal(hf_experts, ref_route.experts)
    torch.testing.assert_close(hf_weights, ref_route.weights)
    assert len(hidden) == cfg.num_layers - cfg.num_dense_layers  # dense blocks route nothing


def test_remap_rejects_contract_drift(parity_pair) -> None:
    reference, hf_model, _ = parity_pair
    sd = dict(reference.iter_master_params())
    missing = dict(sd)
    del missing["blocks.1.moe.shared_up"]
    with pytest.raises(ConversionError, match="missing"):
        remap_state_dict(missing, hf_model.config)
    extra = dict(sd)
    extra["blocks.1.moe.legacy_tensor"] = torch.zeros(1)
    with pytest.raises(ConversionError, match="unmapped"):
        remap_state_dict(extra, hf_model.config)
    bad_shape = dict(sd)
    bad_shape["embed.weight"] = torch.zeros(4, 4)
    with pytest.raises(ConversionError, match="shape"):
        remap_state_dict(bad_shape, hf_model.config)
