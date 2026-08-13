"""Tests for C/core/zero1.py — ZeRO-1 AdamW: torch parity, bucketing, reductions."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from C.core.zero1 import SingleProcessComm, Zero1Adam, flat_grad_all_reduce
from mok_core.config import InnerOptConfig


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    """Tiny-op tests thrash 16-way intra-op parallelism; pin one thread here
    (restored afterwards so other modules keep the session default)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


def _expert(name: str) -> bool:
    return ".routed_" in name


def _make_params(seed: int = 0) -> dict[str, nn.Parameter]:
    """Mixed bf16/fp32 params with expert-local and replicated names."""
    torch.manual_seed(seed)
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "blocks.0.attn.qkv.weight": ((8, 16), torch.bfloat16),
        "blocks.0.moe.routed_gate": ((2, 4, 8), torch.bfloat16),
        "blocks.0.moe.routed_up": ((2, 4, 8), torch.bfloat16),
        "blocks.0.moe.router.proj.weight": ((4, 8), torch.float32),
        "embed.weight": ((16, 8), torch.bfloat16),
        "lm_head.weight": ((16, 8), torch.float32),
    }
    return {n: nn.Parameter(torch.randn(s, dtype=dt)) for n, (s, dt) in shapes.items()}


def _set_grads(params: dict[str, nn.Parameter], seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    grads = {n: torch.randn_like(params[n]) for n in sorted(params)}
    for n, g in grads.items():
        params[n].grad = g.clone()
    return grads


class RecordingComm:
    """Captures collective calls without touching the tensors."""

    def __init__(self) -> None:
        self.broadcasts: list[tuple[int, torch.Tensor]] = []
        self.reduces: int = 0

    def broadcast(self, tensor: torch.Tensor, src_rank: int) -> None:
        self.broadcasts.append((src_rank, tensor))

    def all_reduce(self, tensor: torch.Tensor) -> None:
        self.reduces += 1


# --------------------------------------------------------------------------- #
# AdamW math parity
# --------------------------------------------------------------------------- #


class TestAdamWParity:
    def test_bitwise_match_vs_torch_adamw(self) -> None:
        """Multiple steps with a varying LR match torch.optim.AdamW run on FP32
        masters (the optimizer's dtype policy), cast back to the param dtype."""
        betas, eps, wd = (0.9, 0.95), 1e-8, 0.1
        ours = _make_params(1)
        ref = {
            n: nn.Parameter(p.detach().to(torch.float32).clone())
            for n, p in _make_params(1).items()
        }
        opt = Zero1Adam(
            ours, rank=0, world_size=1, is_expert_local=_expert, betas=betas, eps=eps, weight_decay=wd
        )
        names = sorted(ref)
        torch_opt = torch.optim.AdamW(
            [ref[n] for n in names], lr=1.0, betas=betas, eps=eps, weight_decay=wd, foreach=False
        )
        for step, lr in enumerate([3e-4, 1e-3, 2.5e-4, 7e-4]):
            torch.manual_seed(100 + step)
            grads = {n: torch.randn_like(ours[n]) for n in names}
            for n in names:
                ours[n].grad = grads[n].clone()
                ref[n].grad = grads[n].detach().to(torch.float32)
            opt.step(lr)
            for group in torch_opt.param_groups:
                group["lr"] = lr
            torch_opt.step()
            for n in names:
                assert torch.equal(
                    ours[n].detach(), ref[n].detach().to(ours[n].dtype)
                ), (step, n)

    def test_zero_weight_decay_matches_torch(self) -> None:
        ours, ref = _make_params(2), _make_params(2)
        opt = Zero1Adam(ours, rank=0, world_size=1, is_expert_local=_expert, weight_decay=0.0)
        names = sorted(ref)
        torch_opt = torch.optim.AdamW(
            [ref[n] for n in names],
            lr=1e-3,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0,
            foreach=False,
        )
        torch.manual_seed(7)
        for n in names:
            g = torch.randn_like(ours[n])
            ours[n].grad = g.clone()
            ref[n].grad = g.clone()
        opt.step(1e-3)
        torch_opt.step()
        for n in names:
            assert torch.equal(ours[n].detach(), ref[n].detach()), n

    def test_param_without_grad_untouched(self) -> None:
        params = _make_params(3)
        _set_grads(params, 5)
        frozen = "embed.weight"
        params[frozen].grad = None
        before = params[frozen].detach().clone()
        opt = Zero1Adam(params, rank=0, world_size=1, is_expert_local=_expert)
        opt.step(1e-3)
        assert torch.equal(params[frozen].detach(), before)

    def test_two_steps_deterministic(self) -> None:
        """Same inputs -> same bytes, run twice from scratch."""
        results = []
        for _ in range(2):
            params = _make_params(4)
            opt = Zero1Adam(params, rank=0, world_size=1, is_expert_local=_expert)
            for s in range(2):
                _set_grads(params, 50 + s)
                opt.step(1e-3)
            results.append({n: p.detach().clone() for n, p in params.items()})
        for n in results[0]:
            assert torch.equal(results[0][n], results[1][n])


# --------------------------------------------------------------------------- #
# bucketing + broadcast schedule
# --------------------------------------------------------------------------- #


class TestBucketing:
    def test_round_robin_over_sorted_replicated_names(self) -> None:
        params = _make_params()
        world = 3
        replicated = sorted(n for n in params if not _expert(n))
        experts = sorted(n for n in params if _expert(n))
        for rank in range(world):
            opt = Zero1Adam(
                params, rank=rank, world_size=world, is_expert_local=_expert, comm=RecordingComm()
            )
            assert opt.replicated_names == tuple(replicated)
            assert opt.expert_names == tuple(experts)
            for i, name in enumerate(replicated):
                assert opt.owner_of(name) == i % world
            # every rank owns ALL expert-local params plus exactly its bucket
            expected = sorted(experts + [n for i, n in enumerate(replicated) if i % world == rank])
            assert opt.owned_names == tuple(expected)

    def test_expert_names_have_no_owner_entry(self) -> None:
        opt = Zero1Adam(
            _make_params(), rank=0, world_size=2, is_expert_local=_expert, comm=RecordingComm()
        )
        with pytest.raises(KeyError):
            opt.owner_of("blocks.0.moe.routed_gate")

    def test_broadcast_schedule_sorted_replicated_only(self) -> None:
        params = _make_params()
        comm = RecordingComm()
        opt = Zero1Adam(params, rank=0, world_size=2, is_expert_local=_expert, comm=comm)
        _set_grads(params, 6)
        opt.step(1e-3)
        replicated = sorted(n for n in params if not _expert(n))
        assert [src for src, _ in comm.broadcasts] == [i % 2 for i in range(len(replicated))]
        sent = [t for _, t in comm.broadcasts]
        for name, tensor in zip(replicated, sent, strict=True):
            assert tensor.data_ptr() == params[name].data_ptr()  # broadcast of the live storage
        expert_ptrs = {params[n].data_ptr() for n in params if _expert(n)}
        assert all(t.data_ptr() not in expert_ptrs for t in sent)

    def test_two_rank_partition_equals_full_adamw(self) -> None:
        """rank0 bucket + rank1 bucket (+ owner sync) == plain full AdamW."""
        world = 2
        rank_params = [_make_params(11) for _ in range(world)]
        # torch reference runs on FP32 masters, mirroring Zero1Adam's dtype policy
        full = {
            n: nn.Parameter(p.detach().to(torch.float32).clone())
            for n, p in _make_params(11).items()
        }
        opts = [
            Zero1Adam(
                rank_params[r], rank=r, world_size=world, is_expert_local=_expert, comm=RecordingComm()
            )
            for r in range(world)
        ]
        names = sorted(full)
        torch_opt = torch.optim.AdamW(
            [full[n] for n in names], lr=1.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, foreach=False
        )
        for step, lr in enumerate([1e-3, 4e-4]):
            torch.manual_seed(200 + step)
            grads = {n: torch.randn_like(rank_params[0][n]) for n in names}
            for n in names:
                full[n].grad = grads[n].detach().to(torch.float32)
                for r in range(world):
                    rank_params[r][n].grad = grads[n].clone()
            for r in range(world):
                opts[r].step(lr)
            for group in torch_opt.param_groups:
                group["lr"] = lr
            torch_opt.step()
            # emulate the broadcast: copy each replicated param from its owner
            for name in opts[0].replicated_names:
                owner = opts[0].owner_of(name)
                for r in range(world):
                    if r != owner:
                        rank_params[r][name].data.copy_(rank_params[owner][name].data)
            for n in names:
                for r in range(world):
                    assert torch.equal(
                        rank_params[r][n].detach(),
                        full[n].detach().to(rank_params[r][n].dtype),
                    ), (step, n, r)


# --------------------------------------------------------------------------- #
# fresh()
# --------------------------------------------------------------------------- #


class TestFresh:
    def test_fresh_reads_inner_config(self) -> None:
        inner = InnerOptConfig(betas=(0.8, 0.9), eps=1e-9, weight_decay=0.05)
        opt = Zero1Adam.fresh(
            _make_params(), inner, rank=0, world_size=1, is_expert_local=_expert
        )
        assert opt.betas == (0.8, 0.9)
        assert opt.eps == 1e-9
        assert opt.weight_decay == 0.05

    def test_fresh_state_starts_zero(self) -> None:
        """A window-reset optimizer steps exactly like a brand-new one."""
        inner = InnerOptConfig()
        a_params, b_params = _make_params(21), _make_params(21)
        stale = Zero1Adam.fresh(a_params, inner, rank=0, world_size=1, is_expert_local=_expert)
        _set_grads(a_params, 30)
        stale.step(1e-3)
        # reset: fresh optimizer over the SAME (already-stepped) params
        for n in b_params:
            b_params[n].data.copy_(a_params[n].data)
        fresh_a = Zero1Adam.fresh(a_params, inner, rank=0, world_size=1, is_expert_local=_expert)
        fresh_b = Zero1Adam.fresh(b_params, inner, rank=0, world_size=1, is_expert_local=_expert)
        _set_grads(a_params, 31)
        _set_grads(b_params, 31)
        fresh_a.step(1e-3)
        fresh_b.step(1e-3)
        for n in a_params:
            assert torch.equal(a_params[n].detach(), b_params[n].detach()), n


# --------------------------------------------------------------------------- #
# flat_grad_all_reduce
# --------------------------------------------------------------------------- #


class TestFlatAllReduce:
    def test_single_process_round_trip_exact(self) -> None:
        """world_size=1: fp32 flatten -> reduce -> unflatten is bitwise lossless."""
        params = _make_params(5)
        _set_grads(params, 7)
        before = {n: params[n].grad.detach().clone() for n in params}
        flat_grad_all_reduce(params, SingleProcessComm(), 1)
        for n in params:
            assert torch.equal(params[n].grad, before[n]), n

    def test_two_rank_mean(self) -> None:
        params = _make_params(5)
        mine = _set_grads(params, 11)
        torch.manual_seed(13)
        theirs = {n: torch.randn_like(params[n]) for n in sorted(params)}
        other_flat = torch.cat([theirs[n].reshape(-1).to(torch.float32) for n in sorted(params)])

        class PlusComm:
            def broadcast(self, tensor: torch.Tensor, src_rank: int) -> None:
                raise AssertionError("no broadcast expected")

            def all_reduce(self, tensor: torch.Tensor) -> None:
                tensor.add_(other_flat)

        flat_grad_all_reduce(params, PlusComm(), 2)
        for n in params:
            expected = ((mine[n].to(torch.float32) + theirs[n].to(torch.float32)) / 2).to(
                params[n].dtype
            )
            assert torch.equal(params[n].grad, expected), n

    def test_missing_grad_becomes_zero_contribution(self) -> None:
        params = _make_params(5)
        _set_grads(params, 17)
        params["embed.weight"].grad = None
        flat_grad_all_reduce(params, SingleProcessComm(), 1)
        grad = params["embed.weight"].grad
        assert grad is not None
        assert grad.dtype == params["embed.weight"].dtype
        assert torch.equal(grad, torch.zeros_like(params["embed.weight"]))

    def test_reduces_exactly_once(self) -> None:
        params = _make_params(5)
        _set_grads(params, 19)
        comm = RecordingComm()
        flat_grad_all_reduce(params, comm, 2)
        assert comm.reduces == 1

    def test_empty_params_no_collective(self) -> None:
        comm = RecordingComm()
        flat_grad_all_reduce({}, comm, 2)
        assert comm.reduces == 0


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #


class TestValidation:
    def test_rank_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            Zero1Adam(_make_params(), rank=2, world_size=2, is_expert_local=_expert, comm=RecordingComm())

    def test_bad_world_size(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            Zero1Adam(_make_params(), rank=0, world_size=0, is_expert_local=_expert)

    def test_multi_rank_requires_comm(self) -> None:
        with pytest.raises(ValueError, match="comm"):
            Zero1Adam(_make_params(), rank=0, world_size=2, is_expert_local=_expert)

    def test_single_process_comm_rejects_nonzero_src(self) -> None:
        with pytest.raises(ValueError, match="rank 0"):
            SingleProcessComm().broadcast(torch.zeros(1), 1)

    def test_flat_all_reduce_bad_world_size(self) -> None:
        with pytest.raises(ValueError, match="world_size"):
            flat_grad_all_reduce(_make_params(), SingleProcessComm(), 0)
