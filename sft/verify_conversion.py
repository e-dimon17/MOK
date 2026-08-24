"""THE PARITY GATE: converted HF model vs the mok_core reference backend.

F/G run on standard HF kernels; this gate is what makes that safe. It rebuilds
the training-side reference model directly from the DCP checkpoint masters,
loads the converted HF directory, runs both on the same random tokens, and
requires bf16-level agreement of the logits. Release procedure: no SFT job
starts on a conversion that has not passed `verify`.

Both forwards run under `mok_core.model.attention.sdpa_backend()` so the SDPA
kernel is pinned identically on both sides (math on CPU, the deterministic
kernel on CUDA) — the gate compares the model math, not kernel selection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from mok_core.model.attention import sdpa_backend
from sft.convert_dcp_to_hf import load_dcp_state_dict

MAX_ABS_LOGIT_DIFF = 2e-2   # bf16 tolerance
MIN_ARGMAX_AGREEMENT = 0.99


class VerificationError(RuntimeError):
    """The converted model does not reproduce the reference model's logits."""


@dataclass(frozen=True)
class VerificationReport:
    max_abs_diff: float
    argmax_agreement: float
    n_positions: int
    seq: int
    dtype: str

    @property
    def ok(self) -> bool:
        return self.max_abs_diff < MAX_ABS_LOGIT_DIFF and self.argmax_agreement > MIN_ARGMAX_AGREEMENT


def verify(
    checkpoint_dir: Path,
    hf_dir: Path,
    *,
    n_tokens: int = 64,
    seq: int = 32,
    seed: int = 0,
) -> VerificationReport:
    """Compare reference-backend and HF logits on `n_tokens` random positions.

    Gate: max |logit diff| < 2e-2 (bf16 tolerance) AND argmax agreement > 99%.
    Raises VerificationError on failure, returns the report on success.
    """
    if n_tokens < seq or seq < 2:
        raise ValueError(f"need n_tokens >= seq >= 2, got n_tokens={n_tokens} seq={seq}")
    checkpoint_dir, hf_dir = Path(checkpoint_dir), Path(hf_dir)

    # Lazy heavy imports (transformers via the modeling file).
    from mok_core.model import MoKTransformer, reference_config  # noqa: PLC0415
    from sft.hf_model.configuration_mok_moe import MokMoeConfig  # noqa: PLC0415
    from sft.hf_model.modeling_mok_moe import MokMoeForCausalLM  # noqa: PLC0415

    hf_cfg = MokMoeConfig.from_pretrained(hf_dir)
    dtype_name = str(getattr(hf_cfg, "dtype", None) or "bfloat16").removeprefix("torch.")
    torch_dtype = getattr(torch, dtype_name)

    # Reference model straight from the checkpoint masters (bf16 masters +
    # fp32 router/head, exactly as trained).
    model_cfg = hf_cfg.to_model_config()
    reference = MoKTransformer(reference_config(model_cfg), backend="reference")
    state = load_dcp_state_dict(checkpoint_dir / "model")
    reference.load_state_dict(state, strict=True)
    reference.eval()

    hf_model = MokMoeForCausalLM.from_pretrained(hf_dir, dtype=torch_dtype)
    hf_model.eval()

    batch = max(1, n_tokens // seq)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tokens = torch.randint(0, model_cfg.vocab_size, (batch, seq), generator=generator)

    with torch.no_grad(), sdpa_backend(device=tokens.device):
        ref_logits = reference(tokens).logits.float()
        hf_logits = hf_model(input_ids=tokens).logits.float()

    max_abs_diff = float((ref_logits - hf_logits).abs().max())
    agreement = float((ref_logits.argmax(dim=-1) == hf_logits.argmax(dim=-1)).float().mean())
    report = VerificationReport(
        max_abs_diff=max_abs_diff,
        argmax_agreement=agreement,
        n_positions=batch * seq,
        seq=seq,
        dtype=dtype_name,
    )
    if not report.ok:
        raise VerificationError(
            f"conversion parity FAILED: max|logit diff|={max_abs_diff:.6f} "
            f"(gate {MAX_ABS_LOGIT_DIFF}), argmax agreement={agreement:.4f} "
            f"(gate > {MIN_ARGMAX_AGREEMENT}) over {report.n_positions} positions"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-verify-hf",
        description="Parity gate: converted HF model vs the mok_core reference backend.",
    )
    parser.add_argument("checkpoint_dir", type=Path, help="checkpoints/w{window:08d}/ contract dir")
    parser.add_argument("hf_dir", type=Path, help="converted HF model directory")
    parser.add_argument("--n-tokens", type=int, default=64)
    parser.add_argument("--seq", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify(
            args.checkpoint_dir, args.hf_dir, n_tokens=args.n_tokens, seq=args.seq, seed=args.seed
        )
    except VerificationError as err:
        print(str(err))
        return 1
    print(
        f"parity OK: max|logit diff|={report.max_abs_diff:.6f} < {MAX_ABS_LOGIT_DIFF}, "
        f"argmax agreement={report.argmax_agreement:.4f} > {MIN_ARGMAX_AGREEMENT} "
        f"({report.n_positions} positions, dtype {report.dtype})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
