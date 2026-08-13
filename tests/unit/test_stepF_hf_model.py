"""Step F: MokMoeForCausalLM structure, forward/loss, checkpointing, generation."""

from __future__ import annotations

import pytest
import torch

from F.hf_model.configuration_mok_moe import MokMoeConfig
from F.hf_model.modeling_mok_moe import MokMoeForCausalLM


def tiny_hf_cfg(**overrides) -> MokMoeConfig:
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 16,
        "num_dense_layers": 1,
        "dense_intermediate_size": 48,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 64,
    }
    kwargs.update(overrides)
    return MokMoeConfig(**kwargs)


def build_tiny(seed: int = 0) -> MokMoeForCausalLM:
    torch.manual_seed(seed)
    return MokMoeForCausalLM(tiny_hf_cfg())


def test_config_mirrors_and_validates() -> None:
    cfg = tiny_hf_cfg()
    assert cfg.model_type == "mok_moe"
    assert cfg.tie_word_embeddings is False
    assert (cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id) == (0, 1, 2)
    with pytest.raises(ValueError):
        tiny_hf_cfg(num_attention_heads=4, num_key_value_heads=3)
    with pytest.raises(ValueError):
        tiny_hf_cfg(num_experts_per_tok=9)  # > num_experts


def test_forward_shapes_and_loss() -> None:
    model = build_tiny()
    ids = torch.randint(0, 128, (2, 16))
    out = model(input_ids=ids, labels=ids.clone())
    assert out.logits.shape == (2, 16, 128)
    assert out.loss is not None and torch.isfinite(out.loss)
    out.loss.backward()
    gate = model.model.layers[1].mlp.gate
    assert gate.weight.grad is not None                 # router is trainable
    assert not gate.e_score_correction_bias.requires_grad  # selection bias frozen (buffer)
    assert model.lm_head.weight.grad is not None
    assert model.model.embed_tokens.weight.grad is not None  # untied


def test_labels_ignore_index_masks_loss() -> None:
    model = build_tiny()
    ids = torch.randint(0, 128, (1, 12))
    fully_masked = torch.full_like(ids, -100)
    out = model(input_ids=ids, labels=fully_masked)
    assert torch.isnan(out.loss)  # zero live targets -> mean over nothing
    half = ids.clone()
    half[:, :6] = -100
    assert torch.isfinite(model(input_ids=ids, labels=half).loss)


def test_causality_prefix_invariance() -> None:
    model = build_tiny()
    model.eval()
    ids = torch.randint(0, 128, (1, 10))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 128
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model(input_ids=changed).logits
    torch.testing.assert_close(a[:, :-1], b[:, :-1])   # earlier positions untouched
    assert not torch.allclose(a[:, -1], b[:, -1])      # last position reacts


def test_gradient_checkpointing_matches() -> None:
    model = build_tiny()
    ids = torch.randint(0, 128, (2, 16))
    loss_plain = model(input_ids=ids, labels=ids.clone()).loss
    model.gradient_checkpointing_enable()
    loss_gc = model(input_ids=ids, labels=ids.clone()).loss
    torch.testing.assert_close(loss_plain, loss_gc)
    loss_gc.backward()
    assert model.model.layers[1].self_attn.q_proj.weight.grad is not None


def test_generate_greedy_with_cache_matches_no_cache() -> None:
    model = build_tiny()
    model.eval()
    prompt = torch.randint(0, 128, (2, 5))
    with_cache = model.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=True)
    no_cache = model.generate(prompt, max_new_tokens=6, do_sample=False, use_cache=False)
    assert with_cache.shape == (2, 11)
    assert torch.equal(with_cache, no_cache)  # KV-cache path is exact


def test_attention_mask_blocks_padded_positions() -> None:
    """A fully-visible batch row must be unaffected by other rows' padding."""
    model = build_tiny()
    model.eval()
    ids = torch.randint(0, 128, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, :4] = 0  # row 1 left-padded
    with torch.no_grad():
        full = model(input_ids=ids[:1]).logits
        masked = model(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(full[0], masked[0], atol=1e-5, rtol=1e-5)


def test_output_hidden_states() -> None:
    model = build_tiny()
    out = model(input_ids=torch.randint(0, 128, (1, 4)), output_hidden_states=True)
    assert out.hidden_states is not None
    assert len(out.hidden_states) == 3  # embeddings, after layer 0, final norm
    assert out.hidden_states[-1].shape == (1, 4, 32)
