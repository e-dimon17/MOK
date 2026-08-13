"""Step F: DCP-checkpoint-contract -> HF conversion round-trip + parity gate.

The checkpoint dir is created here exactly per the CHECKPOINT LAYOUT CONTRACT
(model/ DCP of iter_master_params(), outer_state.pt, canonical meta.json) so
this test also pins the layout the checkpoint implementer must produce.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from C.core.outer_opt import ReplicatedOuterStep
from F.convert_dcp_to_hf import (
    ConversionError,
    convert,
    infer_hf_config,
    load_dcp_state_dict,
    read_checkpoint_meta,
)
from F.data_prep import CHAT_TEMPLATE
from F.verify_conversion import VerificationError, verify
from mok_core.config import ModelConfig, OuterOptConfig
from mok_core.config.canonical import canonical_bytes
from mok_core.determinism import hash_named_tensors
from mok_core.model import build_reference_model

WINDOW = 42


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


def make_checkpoint_dir(root: Path, cfg: ModelConfig, seed: int = 5) -> Path:
    """Write a contract-conformant checkpoints/w{window:08d}/ directory."""
    import torch.distributed.checkpoint as dcp  # noqa: PLC0415

    model = build_reference_model(cfg, seed=seed)
    for i, moe in enumerate(model.moe_layers()):
        with torch.no_grad():
            moe.router.balance_bias.copy_(torch.linspace(-0.1, 0.1, cfg.num_experts).roll(i))
    state = {name: tensor.detach().clone() for name, tensor in model.iter_master_params()}

    ckpt_dir = root / f"w{WINDOW:08d}"
    ckpt_dir.mkdir(parents=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # single-process DCP notice
        dcp.save(state, checkpoint_id=str(ckpt_dir / "model"))

    outer = ReplicatedOuterStep(
        OuterOptConfig(), {name: tensor.shape for name, tensor in state.items()}
    )
    torch.save({"outer": outer.state_dict()}, ckpt_dir / "outer_state.pt")

    meta = {
        "window": WINDOW,
        "global_step": WINDOW * 20,
        "tokens_consumed": 1_234_567,
        "state_root": hash_named_tensors(state.items()),
        "manifest_hash": "ab" * 32,
        "spec_version": 1,
    }
    (ckpt_dir / "meta.json").write_bytes(canonical_bytes(meta))
    return ckpt_dir


def make_tokenizer_json(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: PLC0415

    vocab = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2}
    for i, word in enumerate(["hello", "world", "hi", "there", "<|im_start|>", "<|im_end|>"]):
        vocab[word] = i + 3
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<|pad|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))
    return path


@pytest.fixture(scope="module")
def converted(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("stepF_convert")
    cfg = tiny_cfg()
    ckpt_dir = make_checkpoint_dir(root / "checkpoints", cfg)
    tok_path = make_tokenizer_json(root / "tokenizer.json")
    out_dir = root / "hf"
    report = convert(ckpt_dir, out_dir, tok_path, dtype="bfloat16", model_config=cfg)
    return {"cfg": cfg, "ckpt_dir": ckpt_dir, "out_dir": out_dir, "report": report, "root": root}


def test_checkpoint_contract_loads(converted) -> None:
    meta = read_checkpoint_meta(converted["ckpt_dir"])
    assert meta["window"] == WINDOW and meta["spec_version"] == 1
    state = load_dcp_state_dict(converted["ckpt_dir"] / "model")
    assert "blocks.1.moe.router.balance_bias" in state
    assert hash_named_tensors(state.items()) == meta["state_root"]
    outer = torch.load(converted["ckpt_dir"] / "outer_state.pt", weights_only=True)
    assert set(outer) == {"outer"}


def test_output_files_and_config(converted) -> None:
    out = converted["out_dir"]
    for fname in (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "configuration_mok_moe.py",
        "modeling_mok_moe.py",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        assert (out / fname).is_file(), fname
    config = json.loads((out / "config.json").read_text())
    assert config["model_type"] == "mok_moe"
    assert config["auto_map"]["AutoModelForCausalLM"] == "modeling_mok_moe.MokMoeForCausalLM"
    assert config["mok_provenance"]["window"] == WINDOW
    assert config["mok_provenance"]["state_root"] == read_checkpoint_meta(
        converted["ckpt_dir"]
    )["state_root"]
    assert config["num_experts"] == 8 and config["num_experts_per_tok"] == 2
    assert (out / "chat_template.jinja").read_text() == CHAT_TEMPLATE
    tok_cfg = json.loads((out / "tokenizer_config.json").read_text())
    assert tok_cfg["chat_template"] == CHAT_TEMPLATE
    gen_cfg = json.loads((out / "generation_config.json").read_text())
    assert (gen_cfg["bos_token_id"], gen_cfg["eos_token_id"], gen_cfg["pad_token_id"]) == (1, 2, 0)


def test_dtype_policy(converted) -> None:
    from safetensors.torch import load_file  # noqa: PLC0415

    sd = load_file(str(converted["out_dir"] / "model.safetensors"))
    assert sd["model.layers.1.mlp.gate.weight"].dtype == torch.float32
    assert sd["model.layers.1.mlp.gate.e_score_correction_bias"].dtype == torch.float32
    assert sd["model.layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16  # dense block
    assert sd["model.layers.1.mlp.experts.0.gate_proj.weight"].dtype == torch.bfloat16
    assert sd["model.embed_tokens.weight"].dtype == torch.bfloat16
    assert sd["lm_head.weight"].dtype == torch.bfloat16  # fp32 master cast per policy


def test_parity_gate_passes(converted) -> None:
    report = verify(converted["ckpt_dir"], converted["out_dir"], n_tokens=256, seq=32, seed=0)
    assert report.ok
    assert report.n_positions == 256
    assert report.max_abs_diff < 2e-2
    assert report.argmax_agreement > 0.99


def test_parity_gate_catches_tampering(converted, tmp_path) -> None:
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    tampered = tmp_path / "hf_tampered"
    tampered.mkdir()
    for f in converted["out_dir"].iterdir():
        if f.is_file():
            (tampered / f.name).write_bytes(f.read_bytes())
    sd = load_file(str(tampered / "model.safetensors"))
    sd["lm_head.weight"] = sd["lm_head.weight"] + 1.0
    save_file(sd, str(tampered / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(VerificationError, match="parity FAILED"):
        verify(converted["ckpt_dir"], tampered, n_tokens=64, seq=32, seed=0)


def test_state_root_gate(converted, tmp_path) -> None:
    ckpt = converted["ckpt_dir"]
    bad = tmp_path / f"w{WINDOW:08d}"
    bad.mkdir()
    (bad / "model").symlink_to(ckpt / "model")
    meta = read_checkpoint_meta(ckpt)
    meta["state_root"] = "0" * 64
    (bad / "meta.json").write_bytes(canonical_bytes(meta))
    with pytest.raises(ConversionError, match="state_root mismatch"):
        convert(bad, tmp_path / "out", dtype="bfloat16", model_config=converted["cfg"])


def test_sharded_write_and_from_pretrained(converted, tmp_path) -> None:
    out2 = tmp_path / "hf_sharded"
    convert(
        converted["ckpt_dir"],
        out2,
        dtype="bfloat16",
        model_config=converted["cfg"],
        max_shard_bytes=1_000_000,
    )
    assert (out2 / "model.safetensors.index.json").is_file()
    index = json.loads((out2 / "model.safetensors.index.json").read_text())
    shard_files = sorted(set(index["weight_map"].values()))
    assert len(shard_files) > 1
    for fname in shard_files:
        assert (out2 / fname).is_file()

    from F.hf_model.modeling_mok_moe import MokMoeForCausalLM  # noqa: PLC0415

    model = MokMoeForCausalLM.from_pretrained(out2, dtype=torch.bfloat16)
    # _keep_in_fp32_modules_strict pins the router back to fp32 in a bf16 load.
    assert model.model.layers[1].mlp.gate.weight.dtype == torch.float32
    assert model.model.layers[1].mlp.gate.e_score_correction_bias.dtype == torch.float32
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    out = model(input_ids=torch.randint(0, 512, (1, 8)))
    assert out.logits.shape == (1, 8, 512)


def test_infer_hf_config_from_shapes(converted) -> None:
    state = load_dcp_state_dict(converted["ckpt_dir"] / "model")
    inferred = infer_hf_config(state)  # head_dim defaults to 128 == tiny cfg's
    cfg = converted["cfg"]
    assert inferred.hidden_size == cfg.hidden_size
    assert inferred.vocab_size == cfg.vocab_size
    assert inferred.num_hidden_layers == cfg.num_layers
    assert inferred.num_attention_heads == cfg.num_q_heads
    assert inferred.num_key_value_heads == cfg.num_kv_heads
    assert inferred.num_experts == cfg.num_experts
    assert inferred.intermediate_size == cfg.intermediate_size
    # top_k is NOT shape-inferable: defaults clamp (documented; pass
    # model_config= for any run whose manifest deviates from defaults).
    assert inferred.num_experts_per_tok == min(8, cfg.num_experts)


def test_chat_template_renders_chatml(converted) -> None:
    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    tokenizer = PreTrainedTokenizerFast.from_pretrained(converted["out_dir"])
    messages = [
        {"role": "user", "content": "hello world"},
        {"role": "assistant", "content": "hi there"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert text == (
        "<|im_start|>user\nhello world<|im_end|>\n"
        "<|im_start|>assistant\nhi there<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
