"""The window runner — the full per-window protocol loop (playbook step C).

`WindowRunner.run_window` executes one outer-loop window end to end, in the
playbook's phase order:

  0. `resolve_phase` — a phase transition whose workspace shape differs from
     the running model requires a clean relaunch (`restart_required`).
  1. Build the `WindowBatchPlan`, prefetch + hash-verify its shards.
  2-5. `run_training_phase` — the PURE training core (see below).
  6. Two-phase payload publication inside the upload gate (`put_window_payload`),
     skipped with a `late_upload` outcome when the gate has already closed.
  7. Await the leader's window certificate and gather EXACTLY the certified
     peer set; any certified payload that cannot be fetched is a `desync`
     outcome (the caller runs `checkpoint.catch_up`).
  8. `build_outer_inputs` + `ReplicatedOuterStep.apply` — the deterministic
     replicated outer step; then a non-fatal sync sanity check against the
     leader's debug slices.
  9. Metrics + periodic checkpoint (`window % checkpoint_every_windows == 0`).

SEPARATION FOR AUDIT — `run_training_phase` is the bitwise-replayable core:
phases 1-4 plus the deterministic derivation of the payload BYTES, with NO
storage/chain/clock dependencies. `C.core.replay.WindowReplayer` calls THE
SAME function with the miner's uid, so the audited computation and the mined
computation share one code path by construction.

Multi-rank contract: every rank runs the training phase and the outer step;
only rank 0 touches storage/chain. Cross-rank coordination uses the injected
`RunnerComm` object collectives exclusively (`gather_object` for compressed
shards and state-root digests, `broadcast_object` for the certificate verdict,
`barrier` at the window edge). Payload merging requires every gathered
parameter name to be globally unique across ranks (replicated names are
round-robin-partitioned by `assign_owned_params`; EP-shard names must be
disjoint by naming) — a duplicate raises rather than silently corrupting the
payload.

The wall clock is injected (`WindowClock`) so gate arithmetic is testable;
the chain client is only touched through `exchange.put_window_payload`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

from mok_core.config import RunConfig
from mok_core.config.manifest import RunManifest
from mok_core.config.schemas import BucketCreds
from mok_core.data import ShardCache, ShardReader, WindowBatchPlan, verify_index_matches_ref
from mok_core.determinism import per_tensor_digests
from mok_core.determinism.hashing import hash_bytes
from mok_core.model import MoKTransformer
from mok_core.storage import StorageClient, StorageError
from mok_core.telemetry import get_logger

from .certificate import WindowCertificate, build_certificate
from .checkpoint import Checkpointer, CheckpointMeta, build_outer_inputs
from .compress import ErrorFeedback, TopKCompressor
from .exchange import (
    ExchangeError,
    UploadReceipt,
    gather_certified,
    get_certificate,
    get_debug_slices,
    put_certificate,
    put_debug_slices,
    put_window_payload,
)
from .inner_loop import InnerLoop, WindowResult
from .outer_opt import OuterReport, ReplicatedOuterStep
from .payload import PayloadMeta, WindowPayload, assign_owned_params, serialize
from .phase import PhaseConfig, accum_at, resolve_phase
from .pseudo_grad import CpuSnapshot, restore_and_extract_delta
from .window_state import _combine_digest_pairs  # framing pinned by test_window_state.py
from .zero1 import SingleProcessComm, TorchDistComm

__all__ = [
    "DENSE_SUFFIX",
    "RunState",
    "RunnerComm",
    "SingleNodeComm",
    "TorchDistRunnerComm",
    "TrainingArtifacts",
    "WindowClock",
    "WindowOutcome",
    "WindowRunner",
    "await_certificate",
    "build_window_plan",
    "run_state_at",
    "run_training_phase",
    "shared_master_root",
]

log = get_logger("core.window_runner")

# Master tensors with this suffix travel as DENSE payload entries (router
# balance biases — tiny fp32 vectors); everything else is compressed. The same
# rule is used by checkpoint.catch_up's default dense_names.
DENSE_SUFFIX = "balance_bias"

FetchFn = Callable[[int], Any]  # async shard-bytes downloader (mok_core.data.download.FetchFn)
SignFn = Callable[[bytes], bytes]


# --------------------------------------------------------------------------- #
# Injected protocols
# --------------------------------------------------------------------------- #


@runtime_checkable
class WindowClock(Protocol):
    """Wall-clock view of the window schedule (chain-derived in production)."""

    def boundary_ts(self, window: int) -> float:
        """Epoch seconds of `window`'s boundary block."""
        ...  # pragma: no cover — protocol

    def now(self) -> float:
        """Current epoch seconds."""
        ...  # pragma: no cover — protocol


@runtime_checkable
class RunnerComm(Protocol):
    """The collectives the window protocol needs beyond zero1.Comm."""

    def broadcast(self, tensor: torch.Tensor, src_rank: int) -> None: ...  # pragma: no cover

    def all_reduce(self, tensor: torch.Tensor) -> None: ...  # pragma: no cover

    def gather_object(self, obj: Any) -> list[Any] | None:
        """All ranks' objects in rank order on rank 0; None elsewhere."""
        ...  # pragma: no cover — protocol

    def broadcast_object(self, obj: Any, src_rank: int) -> Any:
        """Rank `src_rank`'s object, returned on every rank."""
        ...  # pragma: no cover — protocol

    def barrier(self) -> None: ...  # pragma: no cover


class SingleNodeComm(SingleProcessComm):
    """world_size == 1 RunnerComm — exact identities (CPU tests, replay, calibration)."""

    def gather_object(self, obj: Any) -> list[Any]:
        return [obj]

    def broadcast_object(self, obj: Any, src_rank: int = 0) -> Any:
        if src_rank != 0:
            raise ValueError(f"single-process comm has only rank 0, got src_rank={src_rank}")
        return obj

    def barrier(self) -> None:  # noqa: B027 — intentional no-op
        pass


class TorchDistRunnerComm(TorchDistComm):
    """torch.distributed-backed RunnerComm for the in-node rank group."""

    def _group(self) -> Any:
        import torch.distributed as dist  # noqa: PLC0415 — needs an initialized process group

        return self.group if self.group is not None else dist.group.WORLD

    def gather_object(self, obj: Any) -> list[Any] | None:
        import torch.distributed as dist  # noqa: PLC0415

        group = self._group()
        is_dst = dist.get_rank(group) == 0
        out: list[Any] | None = [None] * dist.get_world_size(group) if is_dst else None
        dist.gather_object(obj, out, dst=dist.get_global_rank(group, 0), group=group)
        return out

    def broadcast_object(self, obj: Any, src_rank: int) -> Any:
        import torch.distributed as dist  # noqa: PLC0415

        group = self._group()
        buf = [obj]
        dist.broadcast_object_list(buf, src=dist.get_global_rank(group, src_rank), group=group)
        return buf[0]

    def barrier(self) -> None:
        import torch.distributed as dist  # noqa: PLC0415

        dist.barrier(group=self.group)


# --------------------------------------------------------------------------- #
# Run accounting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RunState:
    """The three global counters a window is parameterized by (consensus values)."""

    global_step: int          # applied outer steps == non-void windows completed
    global_inner_step: int    # total inner steps taken (drives the closed-form LR)
    tokens_consumed: int      # run tokens consumed across all ranks (drives accum ramp)


def run_state_at(cfg: RunConfig, manifest: RunManifest, window: int, *, world_size: int) -> RunState:
    """The consensus `RunState` at the START of `window` — pure arithmetic.

    Folds every non-void window before `window`, mirroring the inner loop's
    token accounting exactly (per-step `accum_at` clamp included), so auditors
    can derive the LR/accum inputs of any window without miner state. Void
    windows contribute nothing (they are excised from the lineage).
    """
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    global_step = 0
    inner_steps = 0
    tokens = 0
    for w in range(window):
        if manifest.is_void(w):
            continue
        phase = resolve_phase(manifest, cfg, w)
        for _ in range(phase.inner_steps):
            accum = max(1, min(accum_at(cfg.window, tokens), phase.grad_accum))
            tokens += accum * phase.tokens_per_rank_microbatch * world_size
        inner_steps += phase.inner_steps
        global_step += 1
    return RunState(global_step=global_step, global_inner_step=inner_steps, tokens_consumed=tokens)


# --------------------------------------------------------------------------- #
# State roots across ranks
# --------------------------------------------------------------------------- #


def _combined_root(digests: Mapping[str, bytes], comm: RunnerComm) -> str | None:
    """Combine per-rank tensor digests into the global state_root (rank 0 only).

    Byte-compatible with `window_state.rank_parallel_state_root` /
    `hash_named_tensors` — the combine framing is imported, not re-derived.
    """
    gathered = comm.gather_object(sorted(digests.items()))
    if gathered is None:
        return None
    merged: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    for rank_pairs in gathered:
        for name, digest in rank_pairs:
            if name in seen:
                raise ValueError(f"tensor {name!r} hashed by more than one rank")
            seen.add(name)
            merged.append((name, digest))
    return _combine_digest_pairs(merged)


def shared_master_root(
    model: MoKTransformer, *, rank: int, world_size: int, comm: RunnerComm
) -> str | None:
    """The model's global state_root, hashed cooperatively (rank 0 gets it).

    Ownership for hashing is `assign_owned_params` — each replicated tensor is
    hashed by exactly one rank; expert-local tensors by their local rank. For
    world_size == 1 this equals `window_state.state_root(iter_master_params())`.
    """
    master = dict(model.iter_master_params())
    owned = assign_owned_params(master, rank, world_size, model.is_expert_local)
    digests = per_tensor_digests((n, master[n]) for n in sorted(owned))
    return _combined_root(digests, comm)


# --------------------------------------------------------------------------- #
# The pure training phase (phases 1-5) — replay imports THIS
# --------------------------------------------------------------------------- #


def build_window_plan(
    manifest: RunManifest,
    phase: PhaseConfig,
    *,
    run_seed: bytes,
    uid: int,
    window: int,
    rank: int,
    world_size: int,
) -> WindowBatchPlan:
    """The consensus batch plan of (uid, window) under `phase` — pure function."""
    return WindowBatchPlan.build(
        manifest,
        run_seed=run_seed,
        uid=uid,
        window=window,
        rank=rank,
        world_size=world_size,
        tokens_per_rank_microbatch=phase.tokens_per_rank_microbatch,
        grad_accum=phase.grad_accum,
        inner_steps=phase.inner_steps,
        seq_len=phase.seq_len,
        dataset=phase.data,
    )


@dataclass(frozen=True, eq=False)
class TrainingArtifacts:
    """Everything the pure training phase of one (uid, window) produces.

    `state_root_start` / `theta_end_root` are the GLOBAL roots (str on rank 0,
    None on other ranks); `theta_end_digests` are this rank's per-tensor θ_end
    digests (audit evidence); `deltas` are this rank's fp32 pseudo-gradients
    Δ = θ_start − θ_end for every local master tensor. The payload triple is
    populated on rank 0 only, and only when compression state was provided.
    """

    uid: int
    window: int
    state_root_start: str | None
    theta_end_root: str | None
    theta_end_digests: dict[str, bytes]
    deltas: dict[str, torch.Tensor]
    result: WindowResult
    sample_digest: str
    payload: WindowPayload | None = None
    payload_bytes: bytes | None = None
    payload_hash: str | None = None


def run_training_phase(
    model: MoKTransformer,
    cfg: RunConfig,
    manifest: RunManifest,
    phase: PhaseConfig,
    *,
    uid: int,
    window: int,
    rank: int,
    world_size: int,
    comm: RunnerComm,
    shard_lookup: Callable[[int], ShardReader],
    global_state: RunState,
    compressor: TopKCompressor | None = None,
    error_feedback: ErrorFeedback | None = None,
    device: str | torch.device = "cpu",
    plan: WindowBatchPlan | None = None,
    run_seed: bytes | None = None,
) -> TrainingArtifacts:
    """Phases 1-5 of the window protocol — the bitwise-replayable core.

    NO storage/chain/clock dependencies: given θ_start-loaded `model` and the
    consensus inputs (uid, window, manifest, cfg, global_state), the θ_end
    root and — when `compressor`+`error_feedback` are given — the payload
    BYTES are deterministic. `C.core.replay` runs THE SAME function (with
    fresh/absent compression state) to audit a miner's window.

    Afterwards the model is back at θ_start bitwise (asserted on per-tensor
    digests) — the outer step is the only thing allowed to move the masters.
    """
    if (compressor is None) != (error_feedback is None):
        raise ValueError("compressor and error_feedback must be provided together (or neither)")
    seed = bytes.fromhex(manifest.prf.run_seed_hex) if run_seed is None else run_seed
    if plan is None:
        plan = build_window_plan(
            manifest, phase, run_seed=seed, uid=uid, window=window, rank=rank, world_size=world_size
        )

    master = dict(model.iter_master_params())
    owned = assign_owned_params(master, rank, world_size, model.is_expert_local)
    owned_pairs = [(n, master[n]) for n in sorted(owned)]

    start_digests = per_tensor_digests(owned_pairs)
    state_root_start = _combined_root(start_digests, comm)

    snapshot = CpuSnapshot.take(master)
    inner = InnerLoop(
        model, cfg, phase, rank=rank, world_size=world_size, comm=comm, device=device
    )
    result = inner.run_window(
        plan,
        shard_lookup,
        window,
        global_inner_step0=global_state.global_inner_step,
        tokens_consumed0=global_state.tokens_consumed,
    )

    theta_end_digests = per_tensor_digests(owned_pairs)  # same tensor objects, mutated in place
    theta_end_root = _combined_root(theta_end_digests, comm)

    deltas = restore_and_extract_delta(master, snapshot)
    restored_digests = per_tensor_digests(owned_pairs)
    damaged = sorted(n for n in start_digests if restored_digests[n] != start_digests[n])
    if damaged:
        raise RuntimeError(f"θ_start restore is not bitwise for {len(damaged)} tensors: {damaged[:8]}")

    payload: WindowPayload | None = None
    payload_bytes: bytes | None = None
    payload_hash: str | None = None
    if compressor is not None and error_feedback is not None:
        shard: dict[str, Any] = {}
        dense_shard: dict[str, torch.Tensor] = {}
        for name, _ in owned_pairs:
            if name.endswith(DENSE_SUFFIX):
                dense_shard[name] = deltas[name].to(torch.float32)
                continue
            buffered = error_feedback.update(name, deltas[name])
            ct = compressor.compress(name, buffered)
            error_feedback.subtract_transmitted(name, compressor.decompress(name, ct))
            shard[name] = ct
        gathered = comm.gather_object((shard, dense_shard))
        if gathered is not None:  # rank 0
            merged: dict[str, Any] = {}
            merged_dense: dict[str, torch.Tensor] = {}
            for rank_comp, rank_dense in gathered:
                for name, ct in rank_comp.items():
                    if name in merged:
                        raise ValueError(
                            f"parameter {name!r} compressed by more than one rank — "
                            "EP shard names must be globally unique"
                        )
                    merged[name] = ct
                for name, t in rank_dense.items():
                    if name in merged_dense:
                        raise ValueError(f"dense parameter {name!r} contributed by more than one rank")
                    merged_dense[name] = t
            assert state_root_start is not None and theta_end_root is not None
            meta = PayloadMeta(
                sample_digest=plan.sample_digest(),
                sample_count=plan.total_sequences,
                theta_end_hash=theta_end_root,
                state_root=state_root_start,
                global_step=global_state.global_step,
                spec_version=manifest.spec_version,
            )
            payload = WindowPayload(
                uid=uid, window=window, compressed=merged, dense=merged_dense, metadata=meta
            )
            payload_bytes = serialize(payload)
            payload_hash = hash_bytes(payload_bytes)

    return TrainingArtifacts(
        uid=uid,
        window=window,
        state_root_start=state_root_start,
        theta_end_root=theta_end_root,
        theta_end_digests=theta_end_digests,
        deltas=deltas,
        result=result,
        sample_digest=plan.sample_digest(),
        payload=payload,
        payload_bytes=payload_bytes,
        payload_hash=payload_hash,
    )


# --------------------------------------------------------------------------- #
# Certificate polling
# --------------------------------------------------------------------------- #


async def await_certificate(
    storage: StorageClient,
    leader_bucket: BucketCreds,
    window: int,
    *,
    timeout_s: float,
    poll_s: float,
) -> WindowCertificate | None:
    """Poll the leader's bucket for the window certificate until `timeout_s`."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            return await get_certificate(storage, leader_bucket, window)
        except (TimeoutError, StorageError, ExchangeError) as e:
            if loop.time() + poll_s > deadline:
                log.warning("certificate poll timed out", window=window, error=str(e))
                return None
            await asyncio.sleep(poll_s)


# --------------------------------------------------------------------------- #
# Outcome
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowOutcome:
    """What one `run_window` call did. Exactly one of the flag combinations:
    `restart_required` (nothing ran), `desync` (trained but could not apply —
    caller runs catch_up), or a completed window (flags False; `late_upload`
    marks a completed window whose own payload missed the gate)."""

    window: int
    state_after: RunState
    restart_required: bool = False
    desync: bool = False
    late_upload: bool = False
    reason: str = ""
    state_root_start: str | None = None
    theta_end_root: str | None = None
    state_root_after: str | None = None
    payload_hash: str | None = None
    upload_key: str | None = None
    gather_uids: tuple[int, ...] = ()
    outer_report: OuterReport | None = None
    train_result: WindowResult | None = None
    checkpoint_saved: bool = False
    sync_divergences: tuple[str, ...] = ()


@dataclass(frozen=True)
class _SelfCommit:
    """CommitLike for the self-leader bootstrap path."""

    uid: int
    payload_hash: str
    in_gate: bool
    valid: bool


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


class WindowRunner:
    """Owns the per-window protocol loop for one node (all ranks construct one).

    All I/O objects are injected; `peer_buckets` / `leader_bucket` are
    window-keyed callables so bucket rotation and leader election stay outside
    this module. `self_leader=True` is the single-node bootstrap mode (step B
    calibration, loopback tests): after its own upload, rank 0 builds, signs
    and publishes the window certificate over its own commit.
    """

    def __init__(
        self,
        model: MoKTransformer,
        cfg: RunConfig,
        manifest: RunManifest,
        *,
        uid: int,
        rank: int,
        world_size: int,
        comm: RunnerComm,
        storage: StorageClient,
        chain: Any,
        shard_cache: ShardCache,
        fetch_fn: FetchFn,
        compressor: TopKCompressor,
        error_feedback: ErrorFeedback,
        outer_step: ReplicatedOuterStep,
        checkpointer: Checkpointer | None = None,
        metrics: Any | None = None,
        clock: WindowClock,
        peer_buckets: Callable[[int], Mapping[int, BucketCreds]],
        leader_bucket: Callable[[int], BucketCreds],
        payload_version: int = 1,
        device: str | torch.device = "cpu",
        self_leader: bool = False,
        sign_fn: SignFn | None = None,
        cert_poll_s: float = 2.0,
        cert_timeout_s: float = 180.0,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range [0, {world_size})")
        self.model = model
        self.cfg = cfg
        self.manifest = manifest
        self.uid = uid
        self.rank = rank
        self.world_size = world_size
        self.comm = comm
        self.storage = storage
        self.chain = chain
        self.shard_cache = shard_cache
        self.fetch_fn = fetch_fn
        self.compressor = compressor
        self.error_feedback = error_feedback
        self.outer_step = outer_step
        self.checkpointer = checkpointer
        self.metrics = metrics
        self.clock = clock
        self.peer_buckets = peer_buckets
        self.leader_bucket = leader_bucket
        self.payload_version = int(payload_version)
        self.device = torch.device(device)
        self.self_leader = bool(self_leader)
        self.sign_fn: SignFn = sign_fn if sign_fn is not None else (lambda _msg: b"")
        self.cert_poll_s = float(cert_poll_s)
        self.cert_timeout_s = float(cert_timeout_s)
        self.run_seed = bytes.fromhex(manifest.prf.run_seed_hex)

        shapes = model.param_shapes()
        self._dense_shapes = {n: s for n, s in shapes.items() if n.endswith(DENSE_SUFFIX)}
        self._comp_shapes = {n: s for n, s in shapes.items() if not n.endswith(DENSE_SUFFIX)}
        self._comp_names = sorted(self._comp_shapes)

    # ------------------------------------------------------------------ #

    def _needs_restart(self, phase: PhaseConfig) -> bool:
        """True iff `phase` changes a workspace shape the process was built with."""
        current_theta = float(self.model.blocks[0].attn.rope_theta)
        return phase.seq_len != self.model.cfg.seq_len or phase.rope_theta != current_theta

    def _gate_deadline(self, window: int) -> float:
        """Gate close for `window`'s payload: next boundary + the grace period."""
        return self.clock.boundary_ts(window + 1) + self.cfg.window.upload_grace_s

    # ------------------------------------------------------------------ #

    async def run_window(self, window: int, global_state: RunState) -> WindowOutcome:
        """Run one full window from θ_start(window); see the module docstring."""
        phase = resolve_phase(self.manifest, self.cfg, window)
        if phase.requires_restart and self._needs_restart(phase):
            return WindowOutcome(
                window=window,
                state_after=global_state,
                restart_required=True,
                reason=f"phase {phase.name!r} changes workspace shapes (seq_len/rope_theta)",
            )

        # Phase 1 — plan + verified shard prefetch.
        t_window = time.monotonic()
        if self.rank == 0:
            log.info(
                "window start",
                window=window,
                phase=phase.name,
                data=phase.data,
                inner_steps=phase.inner_steps,
                global_step=global_state.global_step,
                tokens_consumed=global_state.tokens_consumed,
            )
        verify_index_matches_ref(self.shard_cache.index, self.manifest.dataset(phase.data))
        plan = build_window_plan(
            self.manifest,
            phase,
            run_seed=self.run_seed,
            uid=self.uid,
            window=window,
            rank=self.rank,
            world_size=self.world_size,
        )
        t0 = time.monotonic()
        await self.shard_cache.prefetch(set(plan.shard_ids), self.fetch_fn)
        if self.rank == 0:
            log.info(
                "shards ready",
                window=window,
                shards=len(set(plan.shard_ids)),
                sequences=plan.total_sequences,
                prefetch_s=round(time.monotonic() - t0, 1),
            )
            log.info("training", window=window, inner_steps=phase.inner_steps, device=str(self.device))
        t0 = time.monotonic()

        # Phases 2-5 — the pure training core (replay-shared).
        readers: dict[int, ShardReader] = {}
        try:
            for i in set(plan.shard_ids):
                readers[i] = ShardReader(self.shard_cache.path_for(i), phase.seq_len)
            artifacts = run_training_phase(
                self.model,
                self.cfg,
                self.manifest,
                phase,
                uid=self.uid,
                window=window,
                rank=self.rank,
                world_size=self.world_size,
                comm=self.comm,
                shard_lookup=readers.__getitem__,
                global_state=global_state,
                compressor=self.compressor,
                error_feedback=self.error_feedback,
                device=self.device,
                plan=plan,
                run_seed=self.run_seed,
            )
        finally:
            for reader in readers.values():
                reader.close()

        state_after = RunState(
            global_step=global_state.global_step + 1,
            global_inner_step=artifacts.result.global_inner_steps_done,
            tokens_consumed=global_state.tokens_consumed + artifacts.result.tokens,
        )
        if self.rank == 0:
            r = artifacts.result
            log.info(
                "training done",
                window=window,
                train_s=round(time.monotonic() - t0, 1),
                entry_loss=r.entry_loss,
                final_loss=r.final_loss,
                grad_norm=r.grad_norm_mean,
                capacity_util=r.capacity_util_max,
                tokens=r.tokens,
                theta_end=artifacts.theta_end_root,
                payload_bytes=len(artifacts.payload_bytes) if artifacts.payload_bytes else 0,
            )

        # Phases 6-7 — upload gate, certificate, certified gather (rank 0 I/O).
        verdict = await self._publish_and_gather(window, artifacts)
        verdict = self.comm.broadcast_object(verdict, 0)
        if verdict["status"] == "desync":
            self.comm.barrier()
            return WindowOutcome(
                window=window,
                state_after=global_state,
                desync=True,
                late_upload=verdict["late"],
                reason=verdict["reason"],
                state_root_start=artifacts.state_root_start,
                theta_end_root=artifacts.theta_end_root,
                payload_hash=artifacts.payload_hash,
                train_result=artifacts.result,
            )
        cert: WindowCertificate = verdict["cert"]
        gather = verdict["gather"]

        # Phase 8 — the deterministic replicated outer step, on every rank.
        master = dict(self.model.iter_master_params())
        sparse, dense, norms = build_outer_inputs(gather.payloads, self.compressor, self._comp_names)
        report = self.outer_step.apply(master, sparse, dense, norms)
        state_root_after = shared_master_root(
            self.model, rank=self.rank, world_size=self.world_size, comm=self.comm
        )
        if self.rank == 0:
            log.info(
                "outer step applied",
                window=window,
                certified_peers=list(cert.included_uids),
                applied_peers=report.applied_peers,
                outer_grad_l2=report.global_grad_l2,
                state_root_after=state_root_after,
            )

        sync_divergences: tuple[str, ...] = ()
        checkpoint_saved = False
        if self.rank == 0:
            if self.self_leader:
                await put_debug_slices(self.storage, window, self.uid, master)
            sync_divergences = await self._sync_check(window, cert, master)

            # Phase 9 — metrics + periodic checkpoint.
            if self.metrics is not None:
                self.metrics.emit(
                    "window",
                    window=window,
                    uid=self.uid,
                    entry_loss=artifacts.result.entry_loss,
                    final_loss=artifacts.result.final_loss,
                    mean_loss=artifacts.result.mean_loss,
                    grad_norm_mean=artifacts.result.grad_norm_mean,
                    capacity_util_max=artifacts.result.capacity_util_max,
                    tokens=artifacts.result.tokens,
                    applied_peers=report.applied_peers,
                    outer_grad_l2=report.global_grad_l2,
                    late_upload=verdict["late"],
                    sync_divergences=len(sync_divergences),
                )
            if (
                self.checkpointer is not None
                and window % self.cfg.window.checkpoint_every_windows == 0
            ):
                assert state_root_after is not None
                meta = CheckpointMeta(
                    window=window,
                    global_step=state_after.global_step,
                    tokens_consumed=state_after.tokens_consumed,
                    state_root=state_root_after,
                    manifest_hash=self.manifest.manifest_hash(),
                    spec_version=self.manifest.spec_version,
                )
                await self.checkpointer.save(window, master, self.outer_step.state_dict(), meta)
                checkpoint_saved = True
                log.info("checkpoint saved", window=window, state_root=state_root_after)
            log.info(
                "window done",
                window=window,
                window_s=round(time.monotonic() - t_window, 1),
                late_upload=verdict["late"],
                sync_divergences=len(sync_divergences),
                global_step=state_after.global_step,
                tokens_consumed=state_after.tokens_consumed,
            )

        self.comm.barrier()
        return WindowOutcome(
            window=window,
            state_after=state_after,
            late_upload=verdict["late"],
            state_root_start=artifacts.state_root_start,
            theta_end_root=artifacts.theta_end_root,
            state_root_after=state_root_after,
            payload_hash=artifacts.payload_hash,
            upload_key=verdict["upload_key"],
            gather_uids=tuple(gather.uids),
            outer_report=report,
            train_result=artifacts.result,
            checkpoint_saved=checkpoint_saved,
            sync_divergences=sync_divergences,
        )

    # ------------------------------------------------------------------ #

    async def _publish_and_gather(self, window: int, artifacts: TrainingArtifacts) -> dict[str, Any]:
        """Rank 0: two-phase upload inside the gate, certificate await, certified
        gather. Returns the verdict dict to broadcast; rank > 0 returns None."""
        if self.rank != 0:
            return {}

        late = False
        receipt: UploadReceipt | None = None
        assert artifacts.payload is not None
        gate_left = self._gate_deadline(window) - self.clock.now()
        if gate_left <= 0:
            late = True
            log.warning(
                "upload gate already closed — skipping upload",
                window=window,
                missed_by_s=round(-gate_left, 1),
            )
        else:
            log.info(
                "publishing: phase 1 chain commit (WindowCommit) then phase 2 upload",
                window=window,
                gate_left_s=round(gate_left, 1),
                payload_hash=artifacts.payload_hash,
            )
            t0 = time.monotonic()
            receipt = await put_window_payload(
                self.storage, self.chain, artifacts.payload, version=self.payload_version
            )
            log.info(
                "published",
                window=window,
                key=receipt.key,
                committed=receipt.committed,
                publish_s=round(time.monotonic() - t0, 1),
            )

        if self.self_leader:
            commits: dict[int, _SelfCommit] = {}
            if receipt is not None:
                commits[self.uid] = _SelfCommit(
                    uid=self.uid, payload_hash=receipt.payload_hash, in_gate=True, valid=True
                )
            assert artifacts.state_root_start is not None
            cert = build_certificate(
                window,
                commits,
                {self.uid: 1.0},
                gather_count=self.cfg.window.gather_peer_count,
                reserve_count=self.cfg.window.reserve_peer_count,
                theta_start_root=artifacts.state_root_start,
                leader_uid=self.uid,
                sign=self.sign_fn,
            )
            await put_certificate(self.storage, cert)

        base = {
            "late": late,
            "upload_key": receipt.key if receipt is not None else None,
            "cert": None,
            "gather": None,
        }
        log.info(
            "awaiting leader certificate",
            window=window,
            leader_bucket=self.leader_bucket(window).bucket_name,
            timeout_s=self.cert_timeout_s,
        )
        t0 = time.monotonic()
        cert = await await_certificate(
            self.storage,
            self.leader_bucket(window),
            window,
            timeout_s=self.cert_timeout_s,
            poll_s=self.cert_poll_s,
        )
        if cert is None:
            log.warning("certificate timeout — desync", window=window, waited_s=round(time.monotonic() - t0, 1))
            return {**base, "status": "desync", "reason": "certificate timeout"}
        log.info(
            "certificate received",
            window=window,
            leader_uid=cert.leader_uid,
            included_uids=list(cert.included_uids),
            wait_s=round(time.monotonic() - t0, 1),
        )
        if artifacts.state_root_start is not None and cert.theta_start_root != artifacts.state_root_start:
            return {
                **base,
                "status": "desync",
                "reason": (
                    f"certificate theta_start_root {cert.theta_start_root} != "
                    f"local {artifacts.state_root_start}"
                ),
            }
        gather = await gather_certified(
            self.storage,
            cert,
            self.peer_buckets(window),
            expected_param_shapes=self._comp_shapes,
            expected_dense=self._dense_shapes,
            topk=self.cfg.compression.topk,
            version=self.payload_version,
            deadline_s=self.cfg.storage.gather_timeout_s,
            max_bytes=self.cfg.storage.max_payload_bytes,
            target_chunk=self.cfg.compression.target_chunk,
            leader_bucket=self.leader_bucket(window),
        )
        if gather.missing:
            log.warning("certified payloads unavailable — desync", window=window, missing=gather.missing)
            return {
                **base,
                "cert": cert,
                "status": "desync",
                "reason": f"certified payloads unavailable: {gather.missing}",
            }
        log.info("gathered certified payloads", window=window, uids=list(gather.uids))
        return {**base, "status": "ok", "reason": "", "cert": cert, "gather": gather}

    async def _sync_check(
        self, window: int, cert: WindowCertificate, master: Mapping[str, torch.Tensor]
    ) -> tuple[str, ...]:
        """Non-fatal lockstep sanity: compare our post-outer-step parameter heads
        against the leader's published debug slices. Absence is not an error."""
        try:
            slices = await get_debug_slices(
                self.storage, self.leader_bucket(window), window, cert.leader_uid
            )
        except (TimeoutError, StorageError, ExchangeError) as e:
            log.info("no leader debug slices for sync check", window=window, error=str(e))
            return ()
        divergent: list[str] = []
        for name in sorted(set(slices) & set(master)):
            head = master[name].detach().reshape(-1)[: len(slices[name])]
            ours = [float(v) for v in head.to(device="cpu", dtype=torch.float32).tolist()]
            if ours != slices[name]:
                divergent.append(name)
        if divergent:
            log.warning("sync divergence vs leader debug slices", window=window, names=divergent[:8])
        return tuple(divergent)
