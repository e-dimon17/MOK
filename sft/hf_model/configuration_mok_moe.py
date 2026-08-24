"""HF configuration for the MoK-MoE architecture (`model_type="mok_moe"`).

Mirrors `mok_core.config.ModelConfig` field-for-field under HF-conventional
names so the converted checkpoint runs on standard HF kernels while
staying numerically identical to the mok reference backend:

    ModelConfig field          MokMoeConfig field
    -----------------          ------------------
    num_layers                 num_hidden_layers
    hidden_size                hidden_size
    num_q_heads                num_attention_heads
    num_kv_heads               num_key_value_heads
    head_dim                   head_dim
    vocab_size                 vocab_size
    seq_len                    max_position_embeddings
    rope_theta                 rope_theta
    rms_norm_eps               rms_norm_eps
    num_experts                num_experts
    top_k                      num_experts_per_tok
    intermediate_size          intermediate_size   (routed AND shared expert width)
    num_dense_layers           num_dense_layers    (first N layers: dense SwiGLU, no MoE)
    dense_intermediate_size    dense_intermediate_size

The router bias buffer travels as `e_score_correction_bias` (DeepSeek-V3
naming) and is SELECTION-ONLY, exactly like `Router.balance_bias` in
`mok_core/model/router.py`. Nothing is tied (`tie_word_embeddings=False`).

This file is copied into converted model directories for
`trust_remote_code=True` loading, so it must not import anything outside
`transformers` / stdlib at module level (`mok_core` is imported lazily inside
the two bridge methods, which only run inside this repository).
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class MokMoeConfig(PretrainedConfig):
    model_type = "mok_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        intermediate_size: int = 1024,
        num_dense_layers: int = 3,
        dense_intermediate_size: int = 9216,
        num_experts: int = 128,
        num_experts_per_tok: int = 8,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        router_init_std: float = 0.006,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        mok_provenance: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if num_attention_heads % max(num_key_value_heads, 1) != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) must be in [1, num_experts={num_experts}]"
            )
        if not 0 <= num_dense_layers < num_hidden_layers:
            raise ValueError(
                f"num_dense_layers ({num_dense_layers}) must be in [0, num_hidden_layers="
                f"{num_hidden_layers})"
            )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_dense_layers = num_dense_layers
        self.dense_intermediate_size = dense_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.router_init_std = router_init_std
        self.use_cache = use_cache
        # Conversion provenance: the checkpoint contract's meta.json
        # ({window, global_step, tokens_consumed, state_root, manifest_hash,
        # spec_version}) rides along in config.json.
        self.mok_provenance = mok_provenance
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # -- bridges to/from the training-side schema (repo-internal only) --------

    @classmethod
    def from_model_config(cls, cfg: Any, **kwargs: Any) -> MokMoeConfig:
        """Build from a `mok_core.config.ModelConfig` (lazy import; repo-side only)."""
        return cls(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_layers,
            num_attention_heads=cfg.num_q_heads,
            num_key_value_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            intermediate_size=cfg.intermediate_size,
            num_dense_layers=cfg.num_dense_layers,
            dense_intermediate_size=cfg.dense_intermediate_size,
            num_experts=cfg.num_experts,
            num_experts_per_tok=cfg.top_k,
            max_position_embeddings=cfg.seq_len,
            rope_theta=cfg.rope_theta,
            rms_norm_eps=cfg.rms_norm_eps,
            **kwargs,
        )

    def to_model_config(self) -> Any:
        """The matching `mok_core.config.ModelConfig` (lazy import; repo-side only).

        `ep_size` is set to the largest legal MoK EP size dividing `num_experts`
        purely to satisfy schema validation; consumers building reference models
        force ep_size=1 via `mok_core.model.reference_config` anyway.
        """
        from mok_core.config import ModelConfig  # noqa: PLC0415 — repo-side bridge only
        from mok_core.config.schemas import EP_SIZES  # noqa: PLC0415

        ep_candidates = [e for e in EP_SIZES if self.num_experts % e == 0]
        if not ep_candidates:
            raise ValueError(
                f"num_experts={self.num_experts} admits no legal MoK ep_size from {EP_SIZES}"
            )
        return ModelConfig(
            num_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            seq_len=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            num_experts=self.num_experts,
            top_k=self.num_experts_per_tok,
            intermediate_size=self.intermediate_size,
            num_dense_layers=self.num_dense_layers,
            dense_intermediate_size=self.dense_intermediate_size,
            ep_size=max(ep_candidates),
        )


__all__ = ["MokMoeConfig"]
