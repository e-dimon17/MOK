"""GPU milestone 1a: the MoK autograd wrapper vs the pure-PyTorch reference.

One θ, two backends: a full-expert (ep_size=1) reference replica is built with
`build_reference_model` and RESHARDED into the EP-8 mok-backend model — the
explicit resharding helper below is the same layout transform fleet bring-up's
init-publish uses (owner initializes all experts, each rank takes its
contiguous expert block).

Parity contract (per rank, on that rank's tokens — MoE math per token depends
only on the full expert set and that token's routing, so an ep=1 replica of
rank r's batch is the exact reference for EP-8 rank r):

  - forward:  |loss_mok − loss_ref| < tolerance
  - backward: cosine(embed.grad_mok, embed.grad_ref) > threshold, and every
    weight-grad l2 norm within the relative tolerance. Routed expert grads are
    compared after SUM-reducing the per-rank reference grads (mok's kernel
    accumulates every rank's tokens into the local experts; each rank's
    reference replica only saw its own).

bf16 routed precision isolates the wrapper (strict: cos > 0.999, norms ±2%);
mxfp8 adds quantization noise (documented looser bounds).
"""

from __future__ import annotations

import _synthetic as synth
import pytest
import torch
import torch.nn.functional as F

from mok_core.config import RunConfig
from mok_core.model import MoKTransformer, build_reference_model, init_model, is_expert_local
from mok_core.model.losses import loss_head
from subnet.core.inner_loop import IGNORE_INDEX

TOLERANCES = {
    # routed_precision -> (loss abs tol, d_x cosine min, weight-grad-norm rel tol)
    "bf16": (0.05, 0.999, 0.02),
    "mxfp8": (0.25, 0.99, 0.10),
}


# --------------------------------------------------------------------------- #
# The explicit resharding helper (reference ep=1 layout -> mok EP layout)
# --------------------------------------------------------------------------- #


def reshard_reference_to_mok(ref_model: MoKTransformer, mok_model: MoKTransformer, rank: int) -> None:
    """Copy θ from a full-expert reference model into an EP-sharded mok model.

    Replicated tensors (attention, norms, router incl. balance_bias, shared
    expert, embeddings, LM head) copy verbatim; every `.routed_` tensor takes
    this rank's contiguous expert block ``[rank*E_local, (rank+1)*E_local)``
    of the reference's full ``[E, ...]`` tensor — the protocol EP geometry
    (experts blocked contiguously by rank).
    """
    e_local = mok_model.cfg.num_local_experts
    if ref_model.cfg.num_local_experts != mok_model.cfg.num_experts:
        raise ValueError("reference model must hold ALL experts (ep_size == 1)")
    reference = dict(ref_model.iter_master_params())
    with torch.no_grad():
        for name, param in mok_model.iter_master_params():
            src = reference[name]
            if is_expert_local(name):
                src = src[rank * e_local : (rank + 1) * e_local]
            if tuple(src.shape) != tuple(param.shape):
                raise ValueError(f"{name}: reshard shape {tuple(src.shape)} != target {tuple(param.shape)}")
            param.copy_(src.to(device=param.device, dtype=param.dtype))


def _paired_models(
    toy_cfg: RunConfig, precision: str, device: torch.device, rank: int
) -> tuple[MoKTransformer, MoKTransformer]:
    model_cfg = toy_cfg.model.model_copy(update={"routed_precision": precision})
    ref = build_reference_model(model_cfg, synth.INIT_SEED, device=device)
    mok_model = init_model(
        model_cfg, synth.INIT_SEED, device=device, backend="mok", mok_runtime=toy_cfg.mok
    )
    reshard_reference_to_mok(ref, mok_model, rank)
    synth.prepare_mok_model(mok_model)  # quantize AFTER resharding
    return ref, mok_model


def _token_batch(cfg: RunConfig, device: torch.device, rank: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(1234 + rank)
    batch = 8192 // cfg.model.seq_len  # T = B*S == tokens_per_rank_microbatch
    tokens = torch.randint(0, cfg.model.vocab_size, (batch, cfg.model.seq_len), generator=generator)
    return tokens.to(device)


def _loss(model: MoKTransformer, tokens: torch.Tensor, cfg: RunConfig) -> torch.Tensor:
    output = model(tokens)
    targets = torch.full_like(tokens, IGNORE_INDEX)
    targets[:, :-1] = tokens[:, 1:]  # within-sequence shift, last column ignored
    return loss_head(output.logits, targets, output.loss_inputs, cfg.model).total


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def _norm(t: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(t.float()))


# --------------------------------------------------------------------------- #
# The parity gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_wrapper_parity(dist_ctx, mok_available, toy_cfg, precision: str) -> None:
    import torch.distributed as dist

    if dist_ctx.world_size != toy_cfg.model.ep_size:
        pytest.skip(f"toy4L pins ep_size={toy_cfg.model.ep_size}; world_size={dist_ctx.world_size}")
    loss_tol, cos_min, norm_rtol = TOLERANCES[precision]

    ref, mok_model = _paired_models(toy_cfg, precision, dist_ctx.device, dist_ctx.rank)

    # Resharding sanity before any kernel runs: replicated tensors are bitwise
    # shared, expert tensors are this rank's block of the full set.
    ref_params = dict(ref.iter_master_params())
    e_local = mok_model.cfg.num_local_experts
    for name, param in mok_model.iter_master_params():
        if is_expert_local(name):
            block = ref_params[name][dist_ctx.rank * e_local : (dist_ctx.rank + 1) * e_local]
            assert torch.equal(param.detach(), block.detach())
        else:
            assert torch.equal(param.detach(), ref_params[name].detach())

    tokens = _token_batch(toy_cfg, dist_ctx.device, dist_ctx.rank)

    # ---- forward -----------------------------------------------------------
    loss_ref = _loss(ref, tokens, toy_cfg)
    loss_mok = _loss(mok_model, tokens, toy_cfg)
    diff = abs(float(loss_mok) - float(loss_ref))
    assert diff < loss_tol, (
        f"[{precision}] loss parity broken: ref={float(loss_ref):.6f} "
        f"mok={float(loss_mok):.6f} |Δ|={diff:.6f} >= {loss_tol}"
    )

    # ---- backward ----------------------------------------------------------
    loss_ref.backward()
    loss_mok.backward()

    # d_x at the model input: the embedding weight gradient.
    cos_embed = _cosine(mok_model.embed.weight.grad, ref.embed.weight.grad)
    assert cos_embed > cos_min, f"[{precision}] embed-grad cosine {cos_embed:.6f} <= {cos_min}"

    mok_grads = {n: p.grad for n, p in mok_model.named_parameters() if p.grad is not None}
    for name, grad in mok_grads.items():
        ref_grad = ref_params[name].grad
        assert ref_grad is not None, f"reference produced no grad for {name}"
        if is_expert_local(name):
            # mok accumulated all 8 ranks' tokens into the local experts; the
            # equivalent reference grad is the SUM over ranks, sliced locally.
            full = ref_grad.float().contiguous()
            dist.all_reduce(full, op=dist.ReduceOp.SUM)
            ref_grad = full[dist_ctx.rank * e_local : (dist_ctx.rank + 1) * e_local]
        n_ref, n_mok = _norm(ref_grad), _norm(grad)
        if n_ref < 1e-12 and n_mok < 1e-12:
            continue
        rel = abs(n_mok - n_ref) / max(n_ref, 1e-12)
        assert rel < norm_rtol, (
            f"[{precision}] {name}: grad-norm mismatch ref={n_ref:.6e} mok={n_mok:.6e} rel={rel:.4f}"
        )
        assert _cosine(grad, ref_grad) > cos_min, f"[{precision}] {name}: grad direction diverged"

    dist_ctx.barrier()
