"""vLLM out-of-tree registration of the MoK-MoE architecture.

`register_mok_moe()` registers `MokMoeForCausalLM` (the `architectures` entry
written by sft/convert_dcp_to_hf.py) with vLLM's ModelRegistry, pointing at
`MokMoeForCausalLM_vLLM` below — a vLLM-native implementation of
sft/hf_model/modeling_mok_moe.py built from vLLM primitives:

  - GQA attention: QKVParallelLinear + `get_rope` (neox-style half rotation,
    same math as `mok_core.model.attention.apply_rope`) + paged `Attention`;
  - MoE: `FusedMoE` following vLLM's DeepSeek-V3 pattern — an fp32
    ReplicatedLinear gate carrying the `e_score_correction_bias` buffer, with
    a custom routing function that applies the bias for SELECTION ONLY and
    softmaxes the selected experts' UNBIASED logits (exactly `MokMoeGate`);
    the ungated shared expert is added to the FusedMoE output before the
    tensor-parallel all-reduce (reduce_results=False on both, DeepSeek-V3
    style);
  - untied ParallelLMHead + LogitsProcessor.

vLLM is a lazy, guarded import: this module itself imports instantly (the
name-mapping tables below are pure and CPU-unit-tested), and any touch of the
model class or `register_mok_moe()` without vLLM installed raises a clear
"vllm>=0.8 required" ImportError. Registration passes the class as a string
path, so vLLM worker subprocesses resolve it themselves (module-level
`__getattr__` builds the class on first access).

This file cannot execute in CI (no GPU/vLLM); its pure parts are tested by
tests/unit/test_rl_vllm_plugin.py.
"""

from __future__ import annotations

from typing import Any

import torch

_VLLM_REQUIRED_MSG = (
    "vllm>=0.8 is required for the MoK-MoE vLLM plugin "
    "(pip install 'mok-subnet[post]' on a rollout/eval node)"
)

#: Registry architecture name — must match `architectures` in the converted
#: checkpoint's config.json (the sft/hf_model class name).
HF_ARCHITECTURE = "MokMoeForCausalLM"

#: String path handed to ModelRegistry.register_model — resolved lazily by
#: vLLM (also inside its worker subprocesses) via module __getattr__ below.
VLLM_CLASS_PATH = "rl.vllm_plugin.mok_moe_vllm:MokMoeForCausalLM_vLLM"


# --------------------------------------------------------------------------- #
# Pure weight-name mapping tables (HF checkpoint -> vLLM params); no vllm need
# --------------------------------------------------------------------------- #

#: (vllm fused param fragment, HF checkpoint fragment, shard id) — the
#: standard vLLM stacked-parameter convention: q/k/v rows stack into qkv_proj,
#: gate/up columns stack into gate_up_proj (shared expert only; routed experts
#: go through the FusedMoE mapping below).
STACKED_PARAMS_MAPPING: tuple[tuple[str, str, str | int], ...] = (
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
)

_ROUTED_EXPERT_MARKER = ".mlp.experts."


def is_routed_expert_weight(name: str) -> bool:
    """True for per-expert routed weights (loaded through FusedMoE)."""
    return _ROUTED_EXPERT_MARKER in name


def map_dense_name(name: str) -> tuple[str, str | int | None]:
    """HF checkpoint name -> (vLLM param name, shard_id) for NON-routed
    weights. shard_id is None for 1:1 params (norms, gate, embeddings, ...)."""
    if not is_routed_expert_weight(name):
        for target, source, shard_id in STACKED_PARAMS_MAPPING:
            if source in name:
                return name.replace(source, target), shard_id
    return name, None


def expert_params_mapping(num_experts: int) -> list[tuple[str, str, int, str]]:
    """(vllm param fragment, HF fragment, expert_id, shard_id) rows, mirroring
    `FusedMoE.make_expert_params_mapping(gate_proj, down_proj, up_proj)`:
    gate->w1 and up->w3 stack into `experts.w13_weight`; down->w2 is
    `experts.w2_weight`. Kept as our own pure function so the consensus
    name mapping is testable (and stable) without vllm installed."""
    if num_experts < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    mapping: list[tuple[str, str, int, str]] = []
    for expert_id in range(num_experts):
        for shard_id, proj in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            target = "experts.w13_" if shard_id in ("w1", "w3") else "experts.w2_"
            mapping.append((target, f"experts.{expert_id}.{proj}.", expert_id, shard_id))
    return mapping


def vllm_param_of(name: str, num_experts: int) -> str:
    """The vLLM parameter an HF checkpoint tensor loads into (pure helper —
    used by tests to prove the tables cover every real checkpoint name)."""
    if is_routed_expert_weight(name):
        for target, source, _expert_id, _shard in expert_params_mapping(num_experts):
            if source in name:
                return name.replace(source, target)
        raise KeyError(f"routed expert weight {name!r} matches no expert mapping row")
    return map_dense_name(name)[0]


# --------------------------------------------------------------------------- #
# Registration (vllm lazy + guarded)
# --------------------------------------------------------------------------- #


def _require_vllm() -> Any:
    """Import vllm's ModelRegistry or raise the canonical guard error."""
    try:
        from vllm import ModelRegistry  # noqa: PLC0415 — lazy, [post] extra
    except ImportError as exc:
        raise ImportError(_VLLM_REQUIRED_MSG) from exc
    return ModelRegistry


def register_mok_moe() -> None:
    """Register the MoK-MoE architecture with vLLM (idempotent).

    Call once per vLLM process before building the engine — e.g. from a
    `vllm.general_plugins` entry point, a sitecustomize, or the rollout
    server's startup script. Raises ImportError if vllm is not installed.
    """
    registry = _require_vllm()
    try:
        registered = HF_ARCHITECTURE in registry.get_supported_archs()
    except Exception:  # noqa: BLE001 — registry introspection is best-effort
        registered = False
    if not registered:
        registry.register_model(HF_ARCHITECTURE, VLLM_CLASS_PATH)


# --------------------------------------------------------------------------- #
# The vLLM model class (built on first access; needs vllm)
# --------------------------------------------------------------------------- #

_CLASS_CACHE: dict[str, type] = {}


def _build_vllm_model_class() -> type:  # noqa: C901 — one closed-over class hierarchy
    if "cls" in _CLASS_CACHE:
        return _CLASS_CACHE["cls"]
    try:
        from torch import nn  # noqa: PLC0415
        from vllm.attention import Attention  # noqa: PLC0415
        from vllm.config import VllmConfig  # noqa: PLC0415
        from vllm.distributed import (  # noqa: PLC0415
            get_tensor_model_parallel_world_size,
            tensor_model_parallel_all_reduce,
        )
        from vllm.model_executor.layers.activation import SiluAndMul  # noqa: PLC0415
        from vllm.model_executor.layers.fused_moe import FusedMoE  # noqa: PLC0415
        from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: PLC0415
        from vllm.model_executor.layers.linear import (  # noqa: PLC0415
            MergedColumnParallelLinear,
            QKVParallelLinear,
            ReplicatedLinear,
            RowParallelLinear,
        )
        from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: PLC0415
        from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: PLC0415
        from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: PLC0415
            ParallelLMHead,
            VocabParallelEmbedding,
        )
        from vllm.model_executor.model_loader.weight_utils import (  # noqa: PLC0415
            default_weight_loader,
        )
    except ImportError as exc:
        raise ImportError(_VLLM_REQUIRED_MSG) from exc

    try:
        from vllm.model_executor.models.utils import maybe_prefix  # noqa: PLC0415
    except ImportError:  # pragma: no cover — helper moved across vllm versions

        def maybe_prefix(prefix: str, name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

    class MokMoeMLP(nn.Module):
        """Shared (ungated, always-on) SwiGLU expert."""

        def __init__(
            self, hidden_size: int, intermediate_size: int, quant_config: Any, prefix: str
        ) -> None:
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                hidden_size,
                [intermediate_size] * 2,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                intermediate_size,
                hidden_size,
                bias=False,
                reduce_results=False,  # reduced with the FusedMoE output (DeepSeek-V3 pattern)
                quant_config=quant_config,
                prefix=f"{prefix}.down_proj",
            )
            self.act_fn = SiluAndMul()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            gate_up, _ = self.gate_up_proj(x)
            x, _ = self.down_proj(self.act_fn(gate_up))
            return x

    class MokMoeSparseMoeBlock(nn.Module):
        """FusedMoE routed experts + shared expert, replicating MokMoeGate's
        selection-only `e_score_correction_bias` routing."""

        def __init__(self, config: Any, quant_config: Any, prefix: str) -> None:
            super().__init__()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.gate = ReplicatedLinear(
                config.hidden_size,
                config.num_experts,
                bias=False,
                params_dtype=torch.float32,  # fp32 router, matching training/HF
                quant_config=None,
                prefix=f"{prefix}.gate",
            )
            # Loaded from `...mlp.gate.e_score_correction_bias` (an HF buffer);
            # a frozen Parameter here so named_parameters() picks it up.
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.zeros(config.num_experts, dtype=torch.float32), requires_grad=False
            )
            self.experts = FusedMoE(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                reduce_results=False,  # shared expert added first, one all-reduce after
                renormalize=False,  # routing fn returns final softmax weights
                quant_config=quant_config,
                custom_routing_function=self._select_experts,
                prefix=f"{prefix}.experts",
            )
            self.shared_experts = MokMoeMLP(
                config.hidden_size,
                config.intermediate_size,
                quant_config,
                prefix=f"{prefix}.shared_experts",
            )

        def _select_experts(
            self,
            hidden_states: torch.Tensor,
            gating_output: torch.Tensor,
            topk: int,
            renormalize: bool,  # noqa: ARG002 — weights are already normalized
            **_kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """MokMoeGate.forward: bias steers WHICH experts win; the mixture
            weights are a softmax over the selected UNBIASED fp32 logits."""
            logits = gating_output.float()
            biased = logits + self.gate.e_score_correction_bias
            _, expert_ids = torch.topk(biased, topk, dim=-1)
            weights = torch.softmax(logits.gather(-1, expert_ids), dim=-1)
            return weights, expert_ids.to(torch.int32)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            router_logits, _ = self.gate(hidden_states.to(torch.float32))
            output = self.experts(hidden_states=hidden_states, router_logits=router_logits)
            output = output + self.shared_experts(hidden_states)
            if self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output)
            return output

    class MokMoeAttention(nn.Module):
        def __init__(
            self, config: Any, cache_config: Any, quant_config: Any, prefix: str
        ) -> None:
            super().__init__()
            tp_size = get_tensor_model_parallel_world_size()
            total_heads = config.num_attention_heads
            total_kv_heads = config.num_key_value_heads
            if total_heads % tp_size != 0:
                raise ValueError(f"num heads {total_heads} not divisible by tp={tp_size}")
            self.head_dim = config.head_dim
            self.num_heads = total_heads // tp_size
            self.num_kv_heads = max(1, total_kv_heads // tp_size)
            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.qkv_proj = QKVParallelLinear(
                config.hidden_size,
                self.head_dim,
                total_heads,
                total_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
            self.o_proj = RowParallelLinear(
                total_heads * self.head_dim,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.o_proj",
            )
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=True,  # half-split rotation == mok_core apply_rope
            )
            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.head_dim**-0.5,
                num_kv_heads=self.num_kv_heads,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.attn",
            )

        def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

    class MokMoeDecoderLayer(nn.Module):
        """Pre-norm block via vLLM's fused add-RMSNorm residual convention."""

        def __init__(
            self, config: Any, cache_config: Any, quant_config: Any, prefix: str
        ) -> None:
            super().__init__()
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.self_attn = MokMoeAttention(
                config, cache_config, quant_config, prefix=f"{prefix}.self_attn"
            )
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            # First num_dense_layers blocks are dense SwiGLU (no router/experts),
            # matching the HF class and the training-side MoKBlock.is_dense.
            layer_idx = int(prefix.rsplit(".", 1)[-1])
            if layer_idx < getattr(config, "num_dense_layers", 0):
                self.mlp: nn.Module = MokMoeMLP(
                    config.hidden_size,
                    config.dense_intermediate_size,
                    quant_config,
                    prefix=f"{prefix}.mlp",
                )
            else:
                self.mlp = MokMoeSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            residual: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(positions, hidden_states)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual

    class MokMoeModel(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()
            config = vllm_config.model_config.hf_config
            cache_config = vllm_config.cache_config
            quant_config = vllm_config.quant_config
            self.config = config
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "embed_tokens")
            )
            self.layers = nn.ModuleList(
                MokMoeDecoderLayer(
                    config, cache_config, quant_config, prefix=maybe_prefix(prefix, f"layers.{i}")
                )
                for i in range(config.num_hidden_layers)
            )
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Any = None,  # noqa: ARG002 — no pipeline parallelism
            inputs_embeds: torch.Tensor | None = None,
        ) -> torch.Tensor:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
            residual: torch.Tensor | None = None
            for layer in self.layers:
                hidden_states, residual = layer(positions, hidden_states, residual)
            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states

    class MokMoeForCausalLM_vLLM(nn.Module):
        """vLLM engine class for the `MokMoeForCausalLM` architecture."""

        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }

        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()
            config = vllm_config.model_config.hf_config
            self.config = config
            self.model = MokMoeModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
            self.lm_head = ParallelLMHead(  # untied, like training/HF
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            self.logits_processor = LogitsProcessor(config.vocab_size)

        def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.model.embed_tokens(input_ids)

        def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Any = None,
            inputs_embeds: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

        def compute_logits(
            self, hidden_states: torch.Tensor, sampling_metadata: Any = None
        ) -> torch.Tensor | None:
            try:
                return self.logits_processor(self.lm_head, hidden_states)
            except TypeError:  # pre-0.10 signature keeps sampling_metadata
                return self.logits_processor(self.lm_head, hidden_states, sampling_metadata)

        def load_weights(self, weights: Any) -> set[str]:
            params_dict = dict(self.named_parameters())
            expert_mapping = expert_params_mapping(self.config.num_experts)
            loaded: set[str] = set()
            for name, weight in weights:
                if is_routed_expert_weight(name):
                    for target, source, expert_id, shard_id in expert_mapping:
                        if source not in name:
                            continue
                        vllm_name = name.replace(source, target)
                        param = params_dict[vllm_name]
                        param.weight_loader(
                            param, weight, vllm_name, shard_id=shard_id, expert_id=expert_id
                        )
                        loaded.add(vllm_name)
                        break
                    else:
                        raise KeyError(f"unmapped routed expert weight {name!r}")
                    continue
                vllm_name, shard_id = map_dense_name(name)
                param = params_dict.get(vllm_name)
                if param is None:
                    raise KeyError(f"unexpected checkpoint tensor {name!r} (-> {vllm_name!r})")
                if shard_id is None:
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, weight)
                else:
                    param.weight_loader(param, weight, shard_id)
                loaded.add(vllm_name)
            return loaded

    _CLASS_CACHE["cls"] = MokMoeForCausalLM_vLLM
    return MokMoeForCausalLM_vLLM


def __getattr__(name: str) -> Any:
    """PEP 562: `MokMoeForCausalLM_vLLM` materializes on first access, so the
    string-path registration works in vLLM workers while plain imports of
    this module never touch vllm (and fail loudly if it is missing)."""
    if name == "MokMoeForCausalLM_vLLM":
        return _build_vllm_model_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HF_ARCHITECTURE",
    "STACKED_PARAMS_MAPPING",
    "VLLM_CLASS_PATH",
    "MokMoeForCausalLM_vLLM",  # noqa: F822 — provided by module __getattr__ (needs vllm)
    "expert_params_mapping",
    "is_routed_expert_weight",
    "map_dense_name",
    "register_mok_moe",
    "vllm_param_of",
]
