"""MokMoeForCausalLM — standard-HF-kernel replica of the MoK-54B architecture.

MATH CONTRACT: every op mirrors `mok_core.model.{attention,router,moe,transformer}`
exactly (the pure-PyTorch reference backend), so a converted checkpoint is
numerically interchangeable with the training model:

  - Llama-style GQA causal attention, bias-free fused-equivalent projections,
    RoPE tables built in fp32 and cast (`attention.rope_cos_sin`), torch SDPA
    only — NO flash-attn dependency.
  - Pre-norm RMSNorm blocks: x + attn(norm(x)); x + moe(norm(x)) with the same
    [B,S,H] -> [T,H] flatten (`transformer.MoKBlock`).
  - 128-expert top-8 router with a DeepSeek-V3-style SELECTION-ONLY bias buffer
    `e_score_correction_bias` (== `Router.balance_bias`): fp32 logits, bias
    added for top-k selection only, gate weights are a softmax over the
    SELECTED experts' UNBIASED logits (`router.Router.forward`).
  - SwiGLU experts: silu(x @ Wg.T) * (x @ Wu.T) @ Wd.T, expert-major
    accumulation into a flat [T*topk, H] buffer, fp32 mixture reduction,
    plus an UNGATED always-on shared expert added in fp32
    (`moe.MoKMoELayer._reference_forward`).

Nothing is tied. Gradient checkpointing supported. KV cache via
`transformers.DynamicCache` for generation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_mok_moe import MokMoeConfig

# --------------------------------------------------------------------------- #
# RoPE — identical math to mok_core.model.attention.rope_cos_sin/apply_rope
# --------------------------------------------------------------------------- #


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate `x` [B, S, n_heads, hd] by cos/sin [B, S, hd] (broadcast over heads)."""
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    return x * cos + rotate_half(x) * sin


class MokMoeRotaryEmbedding(nn.Module):
    """cos/sin tables from integer positions; fp32 build then cast, like the
    training stack. Stateless (no buffers) so meta-device materialization in
    `from_pretrained` has nothing to lose."""

    def __init__(self, config: MokMoeConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = float(config.rope_theta)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: int64 [B, S] -> (cos, sin) each [B, S, head_dim] in `dtype`."""
        device = position_ids.device
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        freqs = position_ids[..., None].to(torch.float32) * inv_freq  # [B, S, hd/2]
        emb = torch.cat((freqs, freqs), dim=-1)                       # [B, S, hd]
        return emb.cos().to(dtype), emb.sin().to(dtype)


# --------------------------------------------------------------------------- #
# Attention — mirrors mok_core.model.attention.CausalSelfAttention
# --------------------------------------------------------------------------- #


class MokMoeAttention(nn.Module):
    """GQA causal SDPA attention. q/k/v are separate Linears; the converter
    splits the training stack's fused `qkv` weight row-wise, which is exact."""

    def __init__(self, config: MokMoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        past_key_values: Cache | None,
        cache_position: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        nq, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        q = self.q_proj(hidden_states).view(batch, seq, nq, hd)
        k = self.k_proj(hidden_states).view(batch, seq, nkv, hd)
        v = self.v_proj(hidden_states).view(batch, seq, nkv, hd)

        # RoPE before the head transpose, exactly like the training stack.
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        q = q.transpose(1, 2)  # [B, nq, S, hd]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx, {"cache_position": cache_position})

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            enable_gqa=(nq != nkv),
        )
        y = y.transpose(1, 2).reshape(batch, seq, nq * hd)
        return self.o_proj(y)


# --------------------------------------------------------------------------- #
# MoE — mirrors mok_core.model.router.Router + moe.MoKMoELayer._reference_forward
# --------------------------------------------------------------------------- #


class MokMoeMLP(nn.Module):
    """One SwiGLU FFN: down(silu(gate(x)) * up(x)) — silu(x@Wg.T)*(x@Wu.T)@Wd.T.

    Serves both as an expert (width `intermediate_size`) and, via the override,
    as the dense FFN of the first `num_dense_layers` blocks (mirrors
    `mok_core.model.transformer.DenseSwiGLU`)."""

    def __init__(self, config: MokMoeConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        hidden = config.hidden_size
        inter = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MokMoeGate(nn.Module):
    """FP32 router with the aux-free SELECTION-ONLY bias (== Router.balance_bias).

    The bias steers WHICH experts are picked; the mixture weights are a softmax
    over the selected experts' UNBIASED logits, so dispatch is balanced without
    perturbing the gate values. `e_score_correction_bias` is a non-trainable
    buffer — frozen at its pretrained value throughout SFT.
    """

    def __init__(self, config: MokMoeConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts, dtype=torch.float32),
            persistent=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[T, H] -> (weights fp32 [T, topk], experts int64 [T, topk])."""
        logits = F.linear(hidden_states.float(), self.weight.float())        # fp32 [T, E]
        biased = logits + self.e_score_correction_bias.float()               # selection only
        _, experts = torch.topk(biased, self.top_k, dim=-1)                  # int64 [T, topk]
        selected = logits.gather(1, experts)                                 # UNBIASED logits
        weights = torch.softmax(selected, dim=-1)                            # fp32 [T, topk]
        return weights, experts


class MokMoeSparseMoeBlock(nn.Module):
    """Top-k routed experts + ungated always-on shared expert, [T, H] -> [T, H].

    Op-for-op replica of `MoKMoELayer._reference_forward` (expert-major
    accumulation into a flat [T*topk, H] buffer via index_copy, fp32 mixture
    reduction, shared output added in fp32, cast back to the input dtype).
    """

    def __init__(self, config: MokMoeConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.gate = MokMoeGate(config)
        self.experts = nn.ModuleList(MokMoeMLP(config) for _ in range(config.num_experts))
        self.shared_experts = MokMoeMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden = x.shape
        routing_weights, experts = self.gate(x)
        topk = experts.shape[1]
        flat_experts = experts.reshape(-1)                                     # [T*topk]
        token_idx = torch.arange(num_tokens, device=x.device).repeat_interleave(topk)

        flat_output = torch.zeros(num_tokens * topk, hidden, dtype=x.dtype, device=x.device)
        for expert_idx in range(self.num_experts):
            rows = (flat_experts == expert_idx).nonzero().flatten()
            if rows.numel() == 0:
                continue
            expert_out = self.experts[expert_idx](x[token_idx[rows]])
            flat_output = flat_output.index_copy(0, rows, expert_out)
        routed_output = (
            flat_output.view(num_tokens, topk, hidden).float() * routing_weights.unsqueeze(2)
        ).sum(1)

        shared_output = self.shared_experts(x)
        return (routed_output + shared_output.float()).to(x.dtype)


# --------------------------------------------------------------------------- #
# Decoder layer / model shells
# --------------------------------------------------------------------------- #


class MokMoeDecoderLayer(nn.Module):
    """Pre-norm block: x + attn(norm(x)); x + moe(norm(x)) — mirrors MoKBlock."""

    def __init__(self, config: MokMoeConfig, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MokMoeAttention(config, layer_idx)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # First num_dense_layers blocks: dense SwiGLU, no router/experts
        # (mirrors MoKBlock.is_dense on the training side).
        if layer_idx < config.num_dense_layers:
            self.mlp: nn.Module = MokMoeMLP(config, config.dense_intermediate_size)
        else:
            self.mlp = MokMoeSparseMoeBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        past_key_values: Cache | None,
        cache_position: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask,
            is_causal,
            past_key_values,
            cache_position,
        )
        batch, seq, hidden = hidden_states.shape
        h = self.post_attention_layernorm(hidden_states).reshape(batch * seq, hidden)
        y = self.mlp(h.contiguous())
        return hidden_states + y.view(batch, seq, hidden)


class MokMoePreTrainedModel(PreTrainedModel):
    config_class = MokMoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MokMoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # The router must stay fp32 even when the model loads in bf16/fp16
    # (dtype policy of the conversion — see sft/convert_dcp_to_hf.py).
    _keep_in_fp32_modules_strict = ["mlp.gate.weight", "e_score_correction_bias"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, MokMoeGate):
            module.weight.data.normal_(mean=0.0, std=self.config.router_init_std)
            module.e_score_correction_bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.RMSNorm):
            module.weight.data.fill_(1.0)


class MokMoeModel(MokMoePreTrainedModel):
    def __init__(self, config: MokMoeConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            MokMoeDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MokMoeRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("provide exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if self.gradient_checkpointing and self.training:
            use_cache = False
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        batch, seq = inputs_embeds.shape[:2]
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        cos, sin = self.rotary_emb(position_ids, inputs_embeds.dtype)
        if cos.shape[0] == 1 and batch > 1:
            cos = cos.expand(batch, -1, -1)
            sin = sin.expand(batch, -1, -1)

        kv_len = past_len + seq
        sdpa_mask, is_causal = self._sdpa_mask(
            attention_mask, cache_position, batch, seq, kv_len, inputs_embeds.dtype
        )

        hidden_states = inputs_embeds
        all_hidden_states: tuple[torch.Tensor, ...] | None = () if output_hidden_states else None
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states = (*all_hidden_states, hidden_states)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    cos,
                    sin,
                    sdpa_mask,
                    is_causal,
                    past_key_values,
                    cache_position,
                )
            else:
                hidden_states = layer(
                    hidden_states, cos, sin, sdpa_mask, is_causal, past_key_values, cache_position
                )
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states = (*all_hidden_states, hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )

    @staticmethod
    def _sdpa_mask(
        attention_mask: torch.Tensor | None,
        cache_position: torch.Tensor,
        batch: int,
        q_len: int,
        kv_len: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, bool]:
        """Additive fp mask [B, 1, q_len, kv_len], or (None, is_causal) fast path.

        Fast path (no padding mask, no past): plain `is_causal=True` SDPA —
        the exact op the training stack runs, so parity checks hit it.
        """
        if attention_mask is None:
            if kv_len == q_len:
                return None, q_len > 1
            if q_len == 1:
                return None, False  # single-token decode attends to the whole cache
        device = cache_position.device
        kv_idx = torch.arange(kv_len, device=device)
        allowed = kv_idx[None, :] <= cache_position[:, None]          # [q_len, kv_len]
        allowed = allowed[None, None].expand(batch, 1, q_len, kv_len)
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :kv_len].to(torch.bool)
            allowed = allowed & pad
        min_val = torch.finfo(dtype).min
        mask = torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), min_val, dtype=dtype, device=device),
        )
        return mask, False


class MokMoeForCausalLM(MokMoePreTrainedModel, GenerationMixin):
    def __init__(self, config: MokMoeConfig) -> None:
        super().__init__(config)
        self.model = MokMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )
        logits = self.lm_head(outputs.last_hidden_state)

        loss: torch.Tensor | None = None
        if labels is not None:
            # Standard causal shift; CE in fp32; -100 = masked (non-assistant turns).
            shift_logits = logits[..., :-1, :].float().contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "MokMoeAttention",
    "MokMoeDecoderLayer",
    "MokMoeForCausalLM",
    "MokMoeGate",
    "MokMoeMLP",
    "MokMoeModel",
    "MokMoePreTrainedModel",
    "MokMoeSparseMoeBlock",
]
