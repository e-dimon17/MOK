"""Convert a CHECKPOINT LAYOUT CONTRACT directory to a HuggingFace model dir.

Input (produced by C/core/checkpoint.py — the contract both sides follow):

    checkpoints/w{window:08d}/
      model/            torch.distributed.checkpoint (DCP) of the master state
                        dict, param names from MoKTransformer.iter_master_params()
                        (incl. the router balance_bias buffers)
      outer_state.pt    outer-optimizer momentum (not needed for SFT; ignored)
      meta.json         canonical JSON {window, global_step, tokens_consumed,
                        state_root, manifest_hash, spec_version}

Output: sharded safetensors + config.json (auto_map/trust_remote_code) +
generation_config.json + the modeling/configuration files + tokenizer files +
ChatML chat template.

NAME MAPPING (mok_core master name -> HF name; `{i}` = layer, `{e}` = expert):

    embed.weight                        model.embed_tokens.weight
    blocks.{i}.attn_norm.weight         model.layers.{i}.input_layernorm.weight
    blocks.{i}.attn.qkv.weight          model.layers.{i}.self_attn.q_proj.weight   [rows 0 : nq*hd]
                                        model.layers.{i}.self_attn.k_proj.weight   [rows nq*hd : (nq+nkv)*hd]
                                        model.layers.{i}.self_attn.v_proj.weight   [rows (nq+nkv)*hd : end]
    blocks.{i}.attn.o_proj.weight       model.layers.{i}.self_attn.o_proj.weight
    blocks.{i}.moe_norm.weight          model.layers.{i}.post_attention_layernorm.weight

  dense blocks (i < num_dense_layers — DenseSwiGLU, no router/experts):
    blocks.{i}.moe.w_gate.weight [Id,H] model.layers.{i}.mlp.gate_proj.weight
    blocks.{i}.moe.w_up.weight   [Id,H] model.layers.{i}.mlp.up_proj.weight
    blocks.{i}.moe.w_down.weight [H,Id] model.layers.{i}.mlp.down_proj.weight

  MoE blocks (i >= num_dense_layers):
    blocks.{i}.moe.router.proj.weight   model.layers.{i}.mlp.gate.weight                    (fp32 kept)
    blocks.{i}.moe.router.balance_bias  model.layers.{i}.mlp.gate.e_score_correction_bias   (fp32 kept)
    blocks.{i}.moe.shared_gate          model.layers.{i}.mlp.shared_experts.gate_proj.weight
    blocks.{i}.moe.shared_up            model.layers.{i}.mlp.shared_experts.up_proj.weight
    blocks.{i}.moe.shared_down          model.layers.{i}.mlp.shared_experts.down_proj.weight
    blocks.{i}.moe.routed_gate [E,I,H]  model.layers.{i}.mlp.experts.{e}.gate_proj.weight  [I,H] per e
    blocks.{i}.moe.routed_up   [E,I,H]  model.layers.{i}.mlp.experts.{e}.up_proj.weight    [I,H] per e
    blocks.{i}.moe.routed_down [E,H,I]  model.layers.{i}.mlp.experts.{e}.down_proj.weight  [H,I] per e
    final_norm.weight                   model.norm.weight
    lm_head.weight                      lm_head.weight

DTYPE POLICY: the fp32 masters of the router projection and its selection bias
stay fp32 (the training stack's "fp32 router" rule); everything else — the
bf16 masters AND the fp32 lm_head master — is cast to the requested dtype
(default bfloat16). The modeling file pins the router back to fp32 on load via
`_keep_in_fp32_modules_strict`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from F.data_prep import CHAT_TEMPLATE
from mok_core.determinism import hash_named_tensors

if TYPE_CHECKING:  # transformers stays lazy at runtime
    from F.hf_model.configuration_mok_moe import MokMoeConfig
    from mok_core.config import ModelConfig

META_KEYS = ("window", "global_step", "tokens_consumed", "state_root", "manifest_hash", "spec_version")

# Consensus token strings — mirror A/pipeline/tokenizer_train.py (F must not import A).
PAD_TOKEN, PAD_ID = "<|pad|>", 0
BOS_TOKEN, BOS_ID = "<|bos|>", 1
EOS_TOKEN, EOS_ID = "<|eos|>", 2

TORCH_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

# HF names (suffix match) whose fp32 masters survive the dtype cast.
FP32_KEEP_SUFFIXES = ("mlp.gate.weight", "e_score_correction_bias")

DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3

WEIGHTS_NAME = "model.safetensors"
INDEX_NAME = "model.safetensors.index.json"

HF_MODEL_FILES = ("configuration_mok_moe.py", "modeling_mok_moe.py")


class ConversionError(RuntimeError):
    """Checkpoint contract violation or integrity failure during conversion."""


@dataclass(frozen=True)
class ConversionReport:
    out_dir: Path
    weight_files: tuple[str, ...]
    num_tensors: int
    total_bytes: int
    dtype: str
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# DCP + meta loading
# --------------------------------------------------------------------------- #


def read_checkpoint_meta(checkpoint_dir: Path) -> dict[str, Any]:
    """Parse and validate the contract's meta.json."""
    meta_path = Path(checkpoint_dir) / "meta.json"
    if not meta_path.is_file():
        raise ConversionError(f"missing meta.json in {checkpoint_dir} (checkpoint layout contract)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = [k for k in META_KEYS if k not in meta]
    if missing:
        raise ConversionError(f"meta.json missing contract keys: {missing}")
    return meta


def load_dcp_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    """Single-rank CPU load of a DCP directory into a fresh state dict.

    Shapes/dtypes come from the DCP metadata, so no model instance is needed.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ConversionError(f"missing DCP model/ directory: {model_dir}")
    from torch.distributed.checkpoint import FileSystemReader  # noqa: PLC0415

    try:
        from torch.distributed.checkpoint.format_utils import (  # noqa: PLC0415
            _EmptyStateDictLoadPlanner,
        )
        from torch.distributed.checkpoint.state_dict_loader import (  # noqa: PLC0415
            _load_state_dict,
        )

        state: dict[str, torch.Tensor] = {}
        _load_state_dict(
            state,
            storage_reader=FileSystemReader(str(model_dir)),
            planner=_EmptyStateDictLoadPlanner(),
            no_dist=True,
        )
    except ImportError:  # pragma: no cover — older/newer torch: public two-step fallback
        import tempfile  # noqa: PLC0415

        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save  # noqa: PLC0415

        with tempfile.TemporaryDirectory() as tmp:
            tmp_file = Path(tmp) / "state.pt"
            dcp_to_torch_save(str(model_dir), str(tmp_file))
            state = torch.load(tmp_file, weights_only=True)
    if not state:
        raise ConversionError(f"DCP checkpoint at {model_dir} produced an empty state dict")
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


# --------------------------------------------------------------------------- #
# Config inference / name remap / dtype policy (pure)
# --------------------------------------------------------------------------- #


def _num_layers(sd: Mapping[str, torch.Tensor]) -> int:
    layers = {int(name.split(".")[1]) for name in sd if name.startswith("blocks.")}
    if not layers or layers != set(range(max(layers) + 1)):
        raise ConversionError(f"non-contiguous or missing block indices: {sorted(layers)}")
    return max(layers) + 1


def infer_hf_config(sd: Mapping[str, torch.Tensor], *, head_dim: int | None = None) -> MokMoeConfig:
    """Best-effort MokMoeConfig from checkpoint tensor shapes alone.

    Shape-inferable: layers, hidden, vocab, GQA head split (given head_dim),
    expert count and width. NOT inferable (taken from ModelConfig defaults):
    top_k (clamped to num_experts), rope_theta, rms_norm_eps, seq_len. For any
    run whose manifest deviates from those defaults (e.g. step E's rope_theta
    5e5 / 16k seq), pass `model_config=` to `convert` instead.
    """
    from F.hf_model.configuration_mok_moe import MokMoeConfig  # noqa: PLC0415 — transformers
    from mok_core.config import ModelConfig  # noqa: PLC0415

    defaults = ModelConfig.model_fields
    head_dim = head_dim if head_dim is not None else int(defaults["head_dim"].default)
    vocab_size, hidden_size = sd["embed.weight"].shape
    num_layers = _num_layers(sd)
    # Dense-first blocks carry moe.w_gate.weight instead of moe.routed_gate;
    # they must form a contiguous prefix (mirrors ModelConfig.num_dense_layers).
    dense_idx = sorted(
        int(n.split(".")[1]) for n in sd if n.endswith(".moe.w_gate.weight")
    )
    num_dense = len(dense_idx)
    if dense_idx != list(range(num_dense)):
        raise ConversionError(f"dense blocks are not a contiguous prefix: {dense_idx}")
    if num_dense >= num_layers:
        raise ConversionError(f"all {num_layers} blocks are dense — no MoE layer to infer from")
    dense_intermediate = (
        int(sd["blocks.0.moe.w_gate.weight"].shape[0])
        if num_dense
        else int(ModelConfig.model_fields["dense_intermediate_size"].default)
    )
    num_experts, intermediate_size, hidden_check = sd[
        f"blocks.{num_dense}.moe.routed_gate"
    ].shape
    if hidden_check != hidden_size:
        raise ConversionError(
            f"routed_gate hidden dim {hidden_check} != embedding hidden dim {hidden_size}"
        )
    if hidden_size % head_dim != 0:
        raise ConversionError(f"hidden_size {hidden_size} not divisible by head_dim {head_dim}")
    num_q_heads = hidden_size // head_dim
    qkv_rows = sd["blocks.0.attn.qkv.weight"].shape[0]
    kv_rows = qkv_rows - num_q_heads * head_dim
    if kv_rows <= 0 or kv_rows % (2 * head_dim) != 0:
        raise ConversionError(f"qkv rows {qkv_rows} inconsistent with head_dim {head_dim}")
    num_kv_heads = kv_rows // (2 * head_dim)
    return MokMoeConfig(
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        num_hidden_layers=num_layers,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=int(intermediate_size),
        num_dense_layers=num_dense,
        dense_intermediate_size=dense_intermediate,
        num_experts=int(num_experts),
        num_experts_per_tok=min(int(defaults["top_k"].default), int(num_experts)),
        max_position_embeddings=int(defaults["seq_len"].default),
        rope_theta=float(defaults["rope_theta"].default),
        rms_norm_eps=float(defaults["rms_norm_eps"].default),
    )


def remap_state_dict(sd: Mapping[str, torch.Tensor], cfg: Any) -> dict[str, torch.Tensor]:
    """Pure mok_core -> HF name remap per the module-docstring mapping table.

    `cfg` needs HF-style attributes (MokMoeConfig or equivalent). Strict: every
    checkpoint tensor must be consumed and every expected name present.
    Returned tensors are contiguous owned copies where slicing occurred.
    """
    nq, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    consumed: set[str] = set()

    def take(name: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
        if name not in sd:
            raise ConversionError(f"checkpoint missing master tensor {name!r}")
        tensor = sd[name]
        if shape is not None and tuple(tensor.shape) != shape:
            raise ConversionError(f"{name}: expected shape {shape}, got {tuple(tensor.shape)}")
        consumed.add(name)
        return tensor

    out: dict[str, torch.Tensor] = {}
    out["model.embed_tokens.weight"] = take(
        "embed.weight", (cfg.vocab_size, cfg.hidden_size)
    ).contiguous()
    for i in range(cfg.num_hidden_layers):
        src = f"blocks.{i}."
        dst = f"model.layers.{i}."
        out[dst + "input_layernorm.weight"] = take(src + "attn_norm.weight").contiguous()
        qkv = take(src + "attn.qkv.weight", ((nq + 2 * nkv) * hd, cfg.hidden_size))
        q_w, k_w, v_w = torch.split(qkv, [nq * hd, nkv * hd, nkv * hd], dim=0)
        out[dst + "self_attn.q_proj.weight"] = q_w.clone()
        out[dst + "self_attn.k_proj.weight"] = k_w.clone()
        out[dst + "self_attn.v_proj.weight"] = v_w.clone()
        out[dst + "self_attn.o_proj.weight"] = take(src + "attn.o_proj.weight").contiguous()
        out[dst + "post_attention_layernorm.weight"] = take(src + "moe_norm.weight").contiguous()
        if i < cfg.num_dense_layers:
            d_inter = cfg.dense_intermediate_size
            out[dst + "mlp.gate_proj.weight"] = take(
                src + "moe.w_gate.weight", (d_inter, cfg.hidden_size)
            ).contiguous()
            out[dst + "mlp.up_proj.weight"] = take(
                src + "moe.w_up.weight", (d_inter, cfg.hidden_size)
            ).contiguous()
            out[dst + "mlp.down_proj.weight"] = take(
                src + "moe.w_down.weight", (cfg.hidden_size, d_inter)
            ).contiguous()
            continue
        out[dst + "mlp.gate.weight"] = take(
            src + "moe.router.proj.weight", (cfg.num_experts, cfg.hidden_size)
        ).contiguous()
        out[dst + "mlp.gate.e_score_correction_bias"] = take(
            src + "moe.router.balance_bias", (cfg.num_experts,)
        ).contiguous()
        inter = cfg.intermediate_size
        out[dst + "mlp.shared_experts.gate_proj.weight"] = take(
            src + "moe.shared_gate", (inter, cfg.hidden_size)
        ).contiguous()
        out[dst + "mlp.shared_experts.up_proj.weight"] = take(
            src + "moe.shared_up", (inter, cfg.hidden_size)
        ).contiguous()
        out[dst + "mlp.shared_experts.down_proj.weight"] = take(
            src + "moe.shared_down", (cfg.hidden_size, inter)
        ).contiguous()
        routed_gate = take(src + "moe.routed_gate", (cfg.num_experts, inter, cfg.hidden_size))
        routed_up = take(src + "moe.routed_up", (cfg.num_experts, inter, cfg.hidden_size))
        routed_down = take(src + "moe.routed_down", (cfg.num_experts, cfg.hidden_size, inter))
        for e in range(cfg.num_experts):
            prefix = f"{dst}mlp.experts.{e}."
            out[prefix + "gate_proj.weight"] = routed_gate[e].clone()
            out[prefix + "up_proj.weight"] = routed_up[e].clone()
            out[prefix + "down_proj.weight"] = routed_down[e].clone()
    out["model.norm.weight"] = take("final_norm.weight").contiguous()
    out["lm_head.weight"] = take("lm_head.weight", (cfg.vocab_size, cfg.hidden_size)).contiguous()

    leftover = sorted(set(sd) - consumed)
    if leftover:
        raise ConversionError(f"unmapped checkpoint tensors (contract drift?): {leftover[:8]}")
    return out


def apply_dtype_policy(
    hf_sd: Mapping[str, torch.Tensor], dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Router projection + selection bias stay fp32; everything else -> `dtype`."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in hf_sd.items():
        if name.endswith(FP32_KEEP_SUFFIXES):
            out[name] = tensor.to(torch.float32)
        else:
            out[name] = tensor.to(dtype)
    return out


# --------------------------------------------------------------------------- #
# Safetensors writing
# --------------------------------------------------------------------------- #


def write_sharded_safetensors(
    hf_sd: Mapping[str, torch.Tensor],
    out_dir: Path,
    *,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> tuple[str, ...]:
    """Greedy size-capped sharding in insertion order; writes the HF index when
    more than one shard results. Returns the weight file names written."""
    from safetensors.torch import save_file  # noqa: PLC0415 — lazy

    shards: list[dict[str, torch.Tensor]] = [{}]
    shard_bytes = 0
    for name, tensor in hf_sd.items():
        nbytes = tensor.numel() * tensor.element_size()
        if shard_bytes + nbytes > max_shard_bytes and shards[-1]:
            shards.append({})
            shard_bytes = 0
        shards[-1][name] = tensor.contiguous()
        shard_bytes += nbytes

    out_dir = Path(out_dir)
    if len(shards) == 1:
        save_file(shards[0], str(out_dir / WEIGHTS_NAME), metadata={"format": "pt"})
        return (WEIGHTS_NAME,)

    weight_map: dict[str, str] = {}
    files: list[str] = []
    for idx, shard in enumerate(shards, start=1):
        fname = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out_dir / fname), metadata={"format": "pt"})
        files.append(fname)
        for name in shard:
            weight_map[name] = fname
    total = sum(t.numel() * t.element_size() for t in hf_sd.values())
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / INDEX_NAME).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return tuple(files)


# --------------------------------------------------------------------------- #
# Tokenizer / template files
# --------------------------------------------------------------------------- #


def write_tokenizer_files(tokenizer_path: Path, out_dir: Path, *, model_max_length: int) -> None:
    """Copy the step-A tokenizer.json and write HF tokenizer configs + template."""
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.is_file():
        raise ConversionError(f"tokenizer file not found: {tokenizer_path}")
    shutil.copyfile(tokenizer_path, Path(out_dir) / "tokenizer.json")
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": model_max_length,
        "pad_token": PAD_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "clean_up_tokenization_spaces": False,
        "chat_template": CHAT_TEMPLATE,
    }
    (Path(out_dir) / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True), encoding="utf-8"
    )
    special_map = {"pad_token": PAD_TOKEN, "bos_token": BOS_TOKEN, "eos_token": EOS_TOKEN}
    (Path(out_dir) / "special_tokens_map.json").write_text(
        json.dumps(special_map, indent=2, sort_keys=True), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# The converter
# --------------------------------------------------------------------------- #


def convert(
    checkpoint_dir: Path,
    out_dir: Path,
    tokenizer_path: Path | None = None,
    *,
    dtype: str = "bfloat16",
    model_config: ModelConfig | None = None,
    check_state_root: bool = True,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> ConversionReport:
    """Checkpoint-contract dir -> HF model dir. See module docstring for layout,
    mapping table and dtype policy.

    `model_config` pins architecture fields that shapes cannot determine
    (top_k, rope_theta, seq_len, ...); without it, shape inference +
    ModelConfig defaults are used (correct for the bulk-phase 54B run only).
    `check_state_root=True` recomputes the state_root over the loaded masters
    and requires it to equal meta.json's — the same hash the miners committed
    on-chain, so a passing conversion is provenance-verified.
    """
    if dtype not in TORCH_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(TORCH_DTYPES)}, got {dtype!r}")
    checkpoint_dir, out_dir = Path(checkpoint_dir), Path(out_dir)
    meta = read_checkpoint_meta(checkpoint_dir)
    sd = load_dcp_state_dict(checkpoint_dir / "model")

    if check_state_root:
        actual_root = hash_named_tensors(sd.items())
        if actual_root != meta["state_root"]:
            raise ConversionError(
                f"state_root mismatch: meta.json {meta['state_root']} != recomputed {actual_root}"
            )

    from transformers import GenerationConfig  # noqa: PLC0415

    from F.hf_model.configuration_mok_moe import MokMoeConfig  # noqa: PLC0415 — transformers

    hf_cfg = (
        MokMoeConfig.from_model_config(model_config) if model_config is not None else infer_hf_config(sd)
    )
    hf_cfg.mok_provenance = dict(meta)
    hf_cfg.architectures = ["MokMoeForCausalLM"]
    hf_cfg.auto_map = {
        "AutoConfig": "configuration_mok_moe.MokMoeConfig",
        "AutoModelForCausalLM": "modeling_mok_moe.MokMoeForCausalLM",
    }
    hf_cfg.dtype = dtype

    hf_sd = apply_dtype_policy(remap_state_dict(sd, hf_cfg), TORCH_DTYPES[dtype])

    out_dir.mkdir(parents=True, exist_ok=True)
    weight_files = write_sharded_safetensors(hf_sd, out_dir, max_shard_bytes=max_shard_bytes)
    hf_cfg.save_pretrained(out_dir)
    GenerationConfig(
        pad_token_id=PAD_ID,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        stop_strings=["<|im_end|>"],
    ).save_pretrained(out_dir)

    hf_model_src = Path(__file__).parent / "hf_model"
    for fname in HF_MODEL_FILES:
        shutil.copyfile(hf_model_src / fname, out_dir / fname)

    (out_dir / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")
    if tokenizer_path is not None:
        write_tokenizer_files(
            Path(tokenizer_path), out_dir, model_max_length=hf_cfg.max_position_embeddings
        )

    total_bytes = sum(t.numel() * t.element_size() for t in hf_sd.values())
    return ConversionReport(
        out_dir=out_dir,
        weight_files=weight_files,
        num_tensors=len(hf_sd),
        total_bytes=total_bytes,
        dtype=dtype,
        meta=dict(meta),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-convert-hf",
        description="Convert a MoK DCP checkpoint directory to a HuggingFace model directory.",
    )
    parser.add_argument("checkpoint_dir", type=Path, help="checkpoints/w{window:08d}/ contract dir")
    parser.add_argument("out_dir", type=Path, help="output HF model directory")
    parser.add_argument("--tokenizer", type=Path, default=None, help="step-A tokenizer.json to bundle")
    parser.add_argument("--dtype", choices=sorted(TORCH_DTYPES), default="bfloat16")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="YAML with a `model:` section (e.g. C/configs/toy4L.yaml) pinning non-inferable fields",
    )
    parser.add_argument(
        "--no-state-root-check",
        action="store_true",
        help="skip recomputing state_root against meta.json (NOT recommended)",
    )
    parser.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = None
    if args.model_config is not None:
        from mok_core.config import ModelConfig  # noqa: PLC0415
        from mok_core.config.loader import load_yaml  # noqa: PLC0415

        data = load_yaml(args.model_config)
        model_config = ModelConfig(**data.get("model", data))
    report = convert(
        args.checkpoint_dir,
        args.out_dir,
        args.tokenizer,
        dtype=args.dtype,
        model_config=model_config,
        check_state_root=not args.no_state_root_check,
        max_shard_bytes=args.max_shard_bytes,
    )
    print(
        f"converted window={report.meta['window']} global_step={report.meta['global_step']} "
        f"-> {report.out_dir} ({report.num_tensors} tensors, "
        f"{report.total_bytes / 1e9:.2f} GB as {report.dtype}, "
        f"files: {', '.join(report.weight_files)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
