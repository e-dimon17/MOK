"""In-node ZeRO-1 AdamW with deterministic name-sorted bucketing.

The inner optimizer of the window protocol (playbook step C). Two parameter
classes, split by the `.routed_` ownership marker (`is_expert_local`):

  - EXPERT-LOCAL params (EP-sharded routed experts): every rank holds a
    different tensor, so each rank optimizes its own shard fully locally —
    no state sharding, no broadcast.
  - REPLICATED params (attention, norms, router, shared expert, embeddings,
    LM head): identical on all ranks after the fixed-order gradient
    all-reduce. Adam state is ZeRO-1 partitioned by a DETERMINISTIC
    name-sorted round-robin — replicated name i (in sorted order) is owned
    by rank `i % world_size`. Each rank steps only its bucket, then every
    updated tensor is broadcast from its owner in sorted-name order, so all
    ranks re-converge bitwise before the next microbatch.

The AdamW math replicates `torch.optim.AdamW`'s single-tensor non-capturable
path operation-for-operation (lerp_ first moment, mul_/addcmul_ second moment,
python-float bias corrections, addcdiv_ update) on an FP32 MASTER copy, then
writes the result back into the parameter's dtype — bitwise identical to
`torch.optim.AdamW(..., foreach=False)` applied to FP32 parameters (pinned by
test). Master + moments are fp32 because a bf16 weight-decay factor rounds to a
no-op and bf16 moments cost measurable convergence quality.

Protocol decision #1: the optimizer is constructed FRESH each window
(`Zero1Adam.fresh`) — Adam state starts at zero, making a window a pure
function of (θ_start, uid, window, manifest).

Collectives go through the injected `Comm` protocol: `SingleProcessComm` is
the pure-logic default for world_size == 1 / CPU tests, `TorchDistComm` wraps
torch.distributed for the real 8-rank node. No DDP hooks anywhere — gradient
reduction is the explicit fixed-order `flat_grad_all_reduce`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from mok_core.config import InnerOptConfig

__all__ = [
    "Comm",
    "SingleProcessComm",
    "TorchDistComm",
    "Zero1Adam",
    "flat_grad_all_reduce",
]


@runtime_checkable
class Comm(Protocol):
    """The two collectives the inner loop needs. Implementations must be
    deterministic given deterministic inputs (fixed reduction order)."""

    def broadcast(self, tensor: torch.Tensor, src_rank: int) -> None:
        """Overwrite `tensor` in place with rank `src_rank`'s copy."""
        ...  # pragma: no cover — protocol

    def all_reduce(self, tensor: torch.Tensor) -> None:
        """Sum `tensor` across all ranks, in place."""
        ...  # pragma: no cover — protocol


class SingleProcessComm:
    """world_size == 1: both collectives are exact identities (CPU tests, replay)."""

    def broadcast(self, tensor: torch.Tensor, src_rank: int) -> None:
        if src_rank != 0:
            raise ValueError(f"single-process comm has only rank 0, got src_rank={src_rank}")

    def all_reduce(self, tensor: torch.Tensor) -> None:  # noqa: B027 — intentional no-op
        pass


class TorchDistComm:
    """torch.distributed-backed comm for the in-node 8-rank group.

    `src_rank` is the rank WITHIN `group` (translated to the global rank via
    `get_global_rank`, matching the protocol's group-relative bucketing).
    """

    def __init__(self, group: Any | None = None) -> None:
        self.group = group

    def broadcast(self, tensor: torch.Tensor, src_rank: int) -> None:
        import torch.distributed as dist  # noqa: PLC0415 — needs an initialized process group

        group = self.group if self.group is not None else dist.group.WORLD
        dist.broadcast(tensor, src=dist.get_global_rank(group, src_rank), group=group)

    def all_reduce(self, tensor: torch.Tensor) -> None:
        import torch.distributed as dist  # noqa: PLC0415

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)


# --------------------------------------------------------------------------- #
# fixed-order gradient reduction
# --------------------------------------------------------------------------- #


def flat_grad_all_reduce(
    named_params: Mapping[str, torch.Tensor],
    comm: Comm,
    world_size: int,
) -> None:
    """Average `p.grad` across ranks via ONE flat fp32 buffer in sorted-name order.

    This is the protocol's deterministic gradient reduction (no DDP hooks, no
    per-tensor reductions): grads are copied into a single fp32 flat buffer in
    sorted-name order (missing grads contribute zeros), summed with exactly one
    `comm.all_reduce`, divided by `world_size`, and written back into `p.grad`
    in the parameter's grad dtype. The buffer layout is a pure function of the
    name set, so every rank reduces byte-identical buffers in the same order.

    For world_size == 1 the fp32 round-trip is exact (divide-by-1 and
    bf16->fp32->bf16 are lossless), so the single-process path is bitwise
    identical to not reducing at all — one code path everywhere.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    names = sorted(named_params)
    if not names:
        return
    device = named_params[names[0]].device
    numels = [named_params[n].numel() for n in names]
    flat = torch.zeros(sum(numels), dtype=torch.float32, device=device)

    offset = 0
    for name, numel in zip(names, numels, strict=True):
        grad = named_params[name].grad
        if grad is not None:
            flat[offset : offset + numel].copy_(grad.detach().reshape(-1).to(torch.float32))
        offset += numel

    comm.all_reduce(flat)
    flat.div_(world_size)

    offset = 0
    for name, numel in zip(names, numels, strict=True):
        param = named_params[name]
        chunk = flat[offset : offset + numel].reshape(param.shape)
        if param.grad is None:
            param.grad = chunk.to(param.dtype).clone()
        else:
            param.grad.copy_(chunk)
        offset += numel


# --------------------------------------------------------------------------- #
# ZeRO-1 AdamW
# --------------------------------------------------------------------------- #


@dataclass
class _AdamState:
    step: int
    exp_avg: torch.Tensor      # fp32, regardless of the parameter dtype
    exp_avg_sq: torch.Tensor   # fp32
    master: torch.Tensor       # fp32 master copy the update is applied to


class Zero1Adam:
    """See module docstring. All mutation happens in `step`; construction is cheap."""

    def __init__(
        self,
        named_params: Mapping[str, torch.nn.Parameter],
        *,
        rank: int,
        world_size: int,
        is_expert_local: Callable[[str], bool],
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        comm: Comm | None = None,
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")
        if comm is None:
            if world_size != 1:
                raise ValueError("world_size > 1 requires an explicit comm")
            comm = SingleProcessComm()
        self.rank = rank
        self.world_size = world_size
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.comm = comm

        self._params: dict[str, torch.nn.Parameter] = dict(named_params)
        names = sorted(self._params)
        self._expert_names = tuple(n for n in names if is_expert_local(n))
        self._replicated_names = tuple(n for n in names if not is_expert_local(n))
        # Deterministic name-sorted round-robin: replicated name i -> rank i % world_size.
        self._owner = {n: i % world_size for i, n in enumerate(self._replicated_names)}
        self._owned_names = tuple(
            n for n in names if is_expert_local(n) or self._owner[n] == rank
        )
        # Fresh states start at zero (protocol decision #1). Moments AND the
        # master copy are fp32 whatever the parameter dtype: in bf16 the decay
        # factor (1 - lr*wd) = 0.99997 rounds straight back to 1.0, silently
        # disabling weight decay, and bf16 second moments lose small updates
        # (measured 2.7x worse convergence on a controlled fit).
        self._state = {
            n: _AdamState(
                step=0,
                exp_avg=torch.zeros_like(
                    self._params[n], dtype=torch.float32, memory_format=torch.preserve_format
                ),
                exp_avg_sq=torch.zeros_like(
                    self._params[n], dtype=torch.float32, memory_format=torch.preserve_format
                ),
                master=self._params[n].detach().to(torch.float32).clone(),
            )
            for n in self._owned_names
        }

    @classmethod
    def fresh(
        cls,
        named_params: Mapping[str, torch.nn.Parameter],
        inner: InnerOptConfig,
        *,
        rank: int,
        world_size: int,
        is_expert_local: Callable[[str], bool],
        comm: Comm | None = None,
    ) -> Zero1Adam:
        """The per-window Adam reset: a brand-new optimizer with zero state."""
        return cls(
            named_params,
            rank=rank,
            world_size=world_size,
            is_expert_local=is_expert_local,
            betas=inner.betas,
            eps=inner.eps,
            weight_decay=inner.weight_decay,
            comm=comm,
        )

    # -- introspection (tests / telemetry) -----------------------------------

    @property
    def owned_names(self) -> tuple[str, ...]:
        """Names this rank steps: all expert-local + its replicated bucket (sorted)."""
        return self._owned_names

    @property
    def expert_names(self) -> tuple[str, ...]:
        return self._expert_names

    @property
    def replicated_names(self) -> tuple[str, ...]:
        return self._replicated_names

    def owner_of(self, name: str) -> int:
        """Owning rank of a REPLICATED param (expert-local params raise KeyError)."""
        return self._owner[name]

    # -- the step ------------------------------------------------------------

    @torch.no_grad()
    def step(self, lr: float) -> None:
        """AdamW-update this rank's bucket at `lr`, then broadcast every
        replicated param from its owner in sorted-name order.

        The update replicates torch.optim.AdamW's single-tensor non-capturable
        path bitwise; params without a grad are skipped (their step count does
        not advance — same as torch's lazy state init).
        """
        beta1, beta2 = self.betas
        for name in self._owned_names:
            param = self._params[name]
            grad = param.grad
            if grad is None:
                continue
            state = self._state[name]
            state.step += 1
            master = state.master
            grad32 = grad.detach().to(torch.float32)
            if self.weight_decay != 0:
                master.mul_(1 - lr * self.weight_decay)
            state.exp_avg.lerp_(grad32, 1 - beta1)
            state.exp_avg_sq.mul_(beta2).addcmul_(grad32, grad32, value=1 - beta2)
            step_f = float(state.step)
            bias_correction1 = 1 - beta1**step_f
            bias_correction2 = 1 - beta2**step_f
            step_size = lr / bias_correction1
            denom = (state.exp_avg_sq.sqrt() / bias_correction2**0.5).add_(self.eps)
            master.addcdiv_(state.exp_avg, denom, value=-step_size)
            param.data.copy_(master.to(param.dtype))

        for name in self._replicated_names:  # fixed sorted order on every rank
            self.comm.broadcast(self._params[name].data, self._owner[name])
