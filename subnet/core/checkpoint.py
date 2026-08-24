"""DCP checkpointing + bitwise catch-up.

Catch-up here is BITWISE — after replaying each window's certified outer step
the resulting ``state_root`` must equal the consensus value miners committed
on-chain: a proof of lockstep rather than a steps-behind heuristic.

CHECKPOINT LAYOUT CONTRACT (consumed by the sft converter and by every node):

    checkpoints/w{window:08d}/
      model/            torch.distributed.checkpoint of the master state dict
                        (names from MoKTransformer.iter_master_params(),
                        including balance_bias buffers)
      outer_state.pt    torch.save({'outer': ReplicatedOuterStep.state_dict()})
      meta.json         canonical JSON {window, global_step, tokens_consumed,
                        state_root, manifest_hash, spec_version}

Remote layout mirrors it with the DCP directory tarred into one object:
``keys.checkpoint_key(window, kind)`` for kinds ``model.tar``,
``outer_state.pt``, ``meta.json``.

Window convention: the checkpoint saved "at window w" holds θ_start(w+1) — the
master state after window w's outer step — so ``meta.state_root`` equals the
``WindowCommit.state_root`` miners commit for window w+1, and
``catch_up(from_window=w, to_window=h)`` replays windows w+1 … h.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
import tarfile
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from mok_core.chain.schemas import WindowCommit
from mok_core.config.canonical import canonical_bytes
from mok_core.config.manifest import RunManifest
from mok_core.config.schemas import BucketCreds, RunConfig
from mok_core.storage import KeyFormatError, ObjectMissingError, StorageClient, StorageError, keys
from mok_core.telemetry import get_logger

from . import exchange as _default_exchange
from .compress import (
    ChunkingTransformer,
    CompressedTensor,
    Quantizer,
    TopKCompressor,
    unpack_2bit_values,
    unpack_12bit_indices,
)
from .exchange import CertifiedGather
from .outer_opt import ReplicatedOuterStep
from .payload import WindowPayload
from .window_state import state_root

__all__ = [
    "CatchUpDivergence",
    "CatchUpError",
    "CatchUpReport",
    "CheckpointError",
    "CheckpointMeta",
    "Checkpointer",
    "build_outer_inputs",
    "catch_up",
    "consensus_state_root",
    "sparse_pairs_from_compressed",
]

log = get_logger("core.checkpoint")

MODEL_DIRNAME = "model"
OUTER_STATE_FILENAME = "outer_state.pt"
META_FILENAME = "meta.json"

KIND_MODEL_TAR = "model.tar"
KIND_OUTER_STATE = "outer_state.pt"
KIND_META = "meta.json"
_REMOTE_KINDS = frozenset({KIND_MODEL_TAR, KIND_OUTER_STATE, KIND_META})

_WINDOW_DIR_RE = re.compile(r"^w(\d{8})$")
_META_KEYS = frozenset(
    {"window", "global_step", "tokens_consumed", "state_root", "manifest_hash", "spec_version"}
)


class CheckpointError(RuntimeError):
    """A checkpoint failed to save, load, or validate."""


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckpointMeta:
    """meta.json contents — canonical JSON, byte-stable across nodes."""

    window: int
    global_step: int
    tokens_consumed: int
    state_root: str      # root of the saved master state == θ_start(window+1)
    manifest_hash: str   # canonical hash of the governing RunManifest
    spec_version: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "window": int(self.window),
            "global_step": int(self.global_step),
            "tokens_consumed": int(self.tokens_consumed),
            "state_root": str(self.state_root),
            "manifest_hash": str(self.manifest_hash),
            "spec_version": int(self.spec_version),
        }

    def canonical(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> CheckpointMeta:
        if set(obj) != _META_KEYS:
            raise CheckpointError(f"meta fields {sorted(obj)} != expected {sorted(_META_KEYS)}")
        for name in ("window", "global_step", "tokens_consumed", "spec_version"):
            if type(obj[name]) is not int or obj[name] < 0:
                raise CheckpointError(f"meta field {name!r} must be a non-negative integer")
        for name in ("state_root", "manifest_hash"):
            if type(obj[name]) is not str:
                raise CheckpointError(f"meta field {name!r} must be a string")
        if obj["spec_version"] < 1:
            raise CheckpointError("meta spec_version must be >= 1")
        return cls(**obj)  # type: ignore[arg-type]


def _load_meta(path: Path) -> CheckpointMeta:
    try:
        obj = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise CheckpointError(f"cannot read meta {path}: {e}") from e
    if not isinstance(obj, dict):
        raise CheckpointError(f"meta {path} must be a JSON object")
    return CheckpointMeta.from_dict(obj)


# --------------------------------------------------------------------------- #
# DCP helpers (single-rank CPU path works without init_process_group)
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _quiet_single_process(no_dist: bool):
    """Silence torch's 'assuming single process' UserWarning — on the no_dist
    path that IS the intent (tests, sft conversion, rank-0 saves), not a problem."""
    with warnings.catch_warnings():
        if no_dist:
            warnings.filterwarnings(
                "ignore", message=".*assuming the intent is to (save|load) in a single process.*"
            )
        yield


def _dcp_save(sd: dict[str, torch.Tensor], model_dir: Path, *, no_dist: bool) -> None:
    with _quiet_single_process(no_dist):
        dcp.save(dict(sd), storage_writer=FileSystemWriter(os.fspath(model_dir)), no_dist=no_dist)


def _dcp_load(model_dir: Path, *, no_dist: bool) -> dict[str, torch.Tensor]:
    """Rebuild the full state dict from DCP metadata alone (no template needed)."""
    reader = FileSystemReader(os.fspath(model_dir))
    try:
        md = reader.read_metadata()
    except Exception as e:
        raise CheckpointError(f"cannot read DCP metadata under {model_dir}: {e}") from e
    template: dict[str, torch.Tensor] = {}
    for name, item in md.state_dict_metadata.items():
        size = getattr(item, "size", None)
        props = getattr(item, "properties", None)
        if size is None or props is None:
            raise CheckpointError(f"DCP entry {name!r} is not a tensor — layout contract violated")
        template[name] = torch.empty(size, dtype=props.dtype)
    with _quiet_single_process(no_dist):
        dcp.load(template, storage_reader=FileSystemReader(os.fspath(model_dir)), no_dist=no_dist)
    return template


# --------------------------------------------------------------------------- #
# Deterministic tar (remote model object)
# --------------------------------------------------------------------------- #


def _write_deterministic_tar(src_dir: Path, out_path: Path) -> None:
    """Byte-stable tar: sorted paths, zeroed owner/mtime, fixed modes."""
    with tarfile.open(out_path, "w", format=tarfile.USTAR_FORMAT) as tar:
        for p in sorted(src_dir.rglob("*")):
            info = tar.gettarinfo(os.fspath(p), arcname=p.relative_to(src_dir).as_posix())
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if p.is_dir() else 0o644
            if p.is_file():
                with open(p, "rb") as f:
                    tar.addfile(info, f)
            else:
                tar.addfile(info)


def _extract_tar(tar_path: Path, dest: Path) -> None:
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter="data")  # py3.12 filter: no traversal/abs paths


# --------------------------------------------------------------------------- #
# Checkpointer
# --------------------------------------------------------------------------- #


class Checkpointer:
    """Local DCP checkpoints with optional R2 mirroring and pruned retention.

    ``storage`` writes go to this node's own bucket; reads (``load_latest``
    remote fallback) can target any peer's bucket. ``no_dist=True`` is the
    single-process path (tests, sft conversion); a torchrun rank group passes False.
    """

    def __init__(
        self,
        storage: StorageClient | None,
        local_dir: str | os.PathLike[str],
        keep_local: int = 2,
        *,
        no_dist: bool = True,
    ) -> None:
        if keep_local < 1:
            raise ValueError(f"keep_local must be >= 1, got {keep_local}")
        self.storage = storage
        self.local_dir = Path(local_dir)
        self.keep_local = int(keep_local)
        self.no_dist = bool(no_dist)
        self.local_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #

    def window_dir(self, window: int) -> Path:
        if not 0 <= window < keys.MAX_WINDOW:
            raise ValueError(f"window must be in [0, {keys.MAX_WINDOW}), got {window}")
        return self.local_dir / f"w{window:08d}"

    def save_local(
        self,
        window: int,
        named_params: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        outer_state: Mapping[str, torch.Tensor],
        meta: CheckpointMeta,
    ) -> Path:
        """Write the full checkpoint layout locally (atomic: tmp dir + rename)."""
        if meta.window != window:
            raise ValueError(f"meta.window {meta.window} != window {window}")
        pairs = named_params.items() if isinstance(named_params, Mapping) else named_params
        sd: dict[str, torch.Tensor] = {}
        for name, tensor in pairs:
            if name in sd:
                raise ValueError(f"duplicate parameter name {name!r}")
            sd[name] = tensor.detach()
        if not sd:
            raise ValueError("refusing to checkpoint an empty state dict")

        final = self.window_dir(window)
        tmp = self.local_dir / f".tmp-w{window:08d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            _dcp_save(sd, tmp / MODEL_DIRNAME, no_dist=self.no_dist)
            outer = {name: t.detach().cpu().clone() for name, t in outer_state.items()}
            torch.save({"outer": outer}, tmp / OUTER_STATE_FILENAME)
            (tmp / META_FILENAME).write_bytes(meta.canonical())
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)
        self.prune_local()
        return final

    async def save(
        self,
        window: int,
        named_params: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
        outer_state: Mapping[str, torch.Tensor],
        meta: CheckpointMeta,
    ) -> Path:
        """save_local + upload of the three remote objects when storage is wired."""
        path = self.save_local(window, named_params, outer_state, meta)
        if self.storage is not None:
            await self.upload(window)
        return path

    async def upload(self, window: int) -> None:
        """Mirror a saved local checkpoint to this node's bucket (tar'd DCP dir)."""
        if self.storage is None:
            raise CheckpointError("no StorageClient configured for upload")
        d = self.window_dir(window)
        if not (d / META_FILENAME).is_file():
            raise CheckpointError(f"no local checkpoint at {d}")
        tar_path = self.local_dir / f".upload-w{window:08d}.tar"
        _write_deterministic_tar(d / MODEL_DIRNAME, tar_path)
        try:
            await self.storage.upload_file(keys.checkpoint_key(window, KIND_MODEL_TAR), tar_path)
        finally:
            tar_path.unlink(missing_ok=True)
        await self.storage.upload_file(
            keys.checkpoint_key(window, KIND_OUTER_STATE), d / OUTER_STATE_FILENAME
        )
        await self.storage.upload_file(keys.checkpoint_key(window, KIND_META), d / META_FILENAME)

    # ------------------------------------------------------------------ #
    # Retention
    # ------------------------------------------------------------------ #

    def local_windows(self) -> list[int]:
        """Windows with a complete-looking local checkpoint dir, ascending."""
        out = []
        for entry in self.local_dir.iterdir() if self.local_dir.is_dir() else ():
            m = _WINDOW_DIR_RE.match(entry.name)
            if m and entry.is_dir() and (entry / META_FILENAME).is_file():
                out.append(int(m.group(1)))
        return sorted(out)

    def prune_local(self) -> list[int]:
        """Delete all but the newest ``keep_local`` local checkpoints."""
        windows = self.local_windows()
        removed = windows[: -self.keep_local] if len(windows) > self.keep_local else []
        for w in removed:
            shutil.rmtree(self.window_dir(w), ignore_errors=True)
        return removed

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    def load_local(
        self, window: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], CheckpointMeta]:
        d = self.window_dir(window)
        meta = _load_meta(d / META_FILENAME)
        if meta.window != window:
            raise CheckpointError(f"meta.window {meta.window} != directory window {window}")
        state = _dcp_load(d / MODEL_DIRNAME, no_dist=self.no_dist)
        try:
            outer_obj = torch.load(d / OUTER_STATE_FILENAME, weights_only=True)
        except Exception as e:
            raise CheckpointError(f"cannot load {OUTER_STATE_FILENAME}: {e}") from e
        outer = outer_obj.get("outer") if isinstance(outer_obj, dict) else None
        if not isinstance(outer, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in outer.items()
        ):
            raise CheckpointError(f"{OUTER_STATE_FILENAME} must hold {{'outer': name -> tensor}}")
        return state, outer, meta

    async def load_latest(
        self, *, bucket: BucketCreds | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], CheckpointMeta] | None:
        """Newest local checkpoint; falls back to the newest complete remote one
        in ``bucket`` (downloaded into local_dir) when nothing exists locally."""
        windows = self.local_windows()
        window = windows[-1] if windows else None
        if window is None and bucket is not None and self.storage is not None:
            window = await self._download_latest(bucket)
        if window is None:
            return None
        return self.load_local(window)

    async def _download_latest(self, bucket: BucketCreds) -> int | None:
        if self.storage is None:
            return None
        by_window: dict[int, set[str]] = {}
        for key in await self.storage.list_keys(bucket, "checkpoints/"):
            try:
                ref = keys.parse_checkpoint_key(key)
            except KeyFormatError:
                continue
            by_window.setdefault(ref.window, set()).add(ref.kind)
        complete = sorted((w for w, kinds in by_window.items() if kinds >= _REMOTE_KINDS), reverse=True)
        for window in complete:
            try:
                await self._download_window(bucket, window)
            except (StorageError, CheckpointError, OSError, tarfile.TarError) as e:
                log.warning("remote checkpoint unusable", window=window, error=str(e))
                continue
            return window
        return None

    async def _download_window(self, bucket: BucketCreds, window: int) -> None:
        assert self.storage is not None
        tmp = self.local_dir / f".dl-w{window:08d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            tar_path = tmp / KIND_MODEL_TAR
            await self.storage.download_file(bucket, keys.checkpoint_key(window, KIND_MODEL_TAR), tar_path)
            (tmp / MODEL_DIRNAME).mkdir(exist_ok=True)
            _extract_tar(tar_path, tmp / MODEL_DIRNAME)
            tar_path.unlink()
            await self.storage.download_file(
                bucket, keys.checkpoint_key(window, KIND_OUTER_STATE), tmp / OUTER_STATE_FILENAME
            )
            await self.storage.download_file(
                bucket, keys.checkpoint_key(window, KIND_META), tmp / META_FILENAME
            )
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        final = self.window_dir(window)
        if final.exists():
            shutil.rmtree(final)
        os.replace(tmp, final)


# --------------------------------------------------------------------------- #
# Payloads -> outer-step inputs (shared with window_runner — the ONE conversion)
# --------------------------------------------------------------------------- #


def sparse_pairs_from_compressed(
    name: str,
    ct: CompressedTensor,
    compressor: TopKCompressor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One peer's (flat original indices, dequantized fp32 values) for a parameter.

    Chunk-space positions map back through the registered ``ChunkGeometry``;
    positions that fall in zero-padding (partial edge chunks) are dropped, so
    scattering the pairs into a zeroed tensor reproduces
    ``compressor.decompress`` bitwise — pinned by tests.
    """
    g = compressor.transformer.geometry(name)
    if (ct.n_chunks, ct.chunk_elems) != (g.n_chunks, g.chunk_elems) or ct.orig_shape != g.orig_shape:
        raise ValueError(
            f"{name}: compressed geometry ({ct.n_chunks}, {ct.chunk_elems}, {ct.orig_shape}) "
            f"!= registered ({g.n_chunks}, {g.chunk_elems}, {g.orig_shape})"
        )
    n = ct.n_values
    codes = unpack_2bit_values(ct.codes_packed, n)
    vals = compressor.quantizer.dequantize(codes, ct.qparams)
    local = unpack_12bit_indices(ct.idxs_packed, n)
    if n and int(local.max().item()) >= g.chunk_elems:
        raise ValueError(f"{name}: chunk-local index out of range")
    chunk_ids = torch.arange(g.n_chunks, dtype=torch.int64).repeat_interleave(ct.topk)

    tc = compressor.transformer.target_chunk
    if g.mode == "grid":
        blocks_per_row = g.pad_cols // tc
        row = (chunk_ids // blocks_per_row) * tc + local // tc
        col = (chunk_ids % blocks_per_row) * tc + local % tc
        keep = (row < g.rows) & (col < g.cols)
        flat = row * g.cols + col
    else:
        flat = chunk_ids * g.chunk_elems + local
        keep = flat < g.numel
    return flat[keep], vals[keep].to(torch.float32)


def build_outer_inputs(
    payloads: Mapping[int, WindowPayload],
    compressor: TopKCompressor,
    param_names: Sequence[str],
) -> tuple[
    dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
    dict[str, list[torch.Tensor]],
    dict[str, torch.Tensor],
]:
    """Certified payloads -> ``ReplicatedOuterStep.apply`` inputs, in payload order.

    ``payloads`` MUST be uid-ascending (certificate order — CertifiedGather
    guarantees it); the returned per-peer lists preserve that order, which is
    consensus-bearing for the deterministic merge. Per-peer pre-clip norms are
    the l2 norms of what actually enters the merge (identical to the norm of
    the decompressed dense contribution). window_runner imports this function —
    catch-up and live training share the exact conversion.
    """
    uids = list(payloads)
    if uids != sorted(uids):
        raise ValueError("payloads must be uid-ascending (certificate order)")
    names = sorted(set(param_names))
    peer_sparse: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    dense_contribs: dict[str, list[torch.Tensor]] = {}
    norm_lists: dict[str, list[float]] = {}
    if not uids:
        return {}, {}, {}

    dense_names = sorted(payloads[uids[0]].dense)
    for uid in uids:
        p = payloads[uid]
        for name in names:
            ct = p.compressed.get(name)
            if ct is None:
                raise ValueError(f"uid {uid}: payload missing compressed param {name!r}")
            idx, vals = sparse_pairs_from_compressed(name, ct, compressor)
            peer_sparse.setdefault(name, []).append((idx, vals))
            norm_lists.setdefault(name, []).append(float(torch.linalg.vector_norm(vals)))
        if sorted(p.dense) != dense_names:
            raise ValueError(f"uid {uid}: dense name set differs from certificate cohort")
        for name in dense_names:
            t = p.dense[name].detach().to(torch.float32)
            dense_contribs.setdefault(name, []).append(t)
            norm_lists.setdefault(name, []).append(float(torch.linalg.vector_norm(t)))
    peer_norms = {n: torch.tensor(v, dtype=torch.float32) for n, v in norm_lists.items()}
    return peer_sparse, dense_contribs, peer_norms


# --------------------------------------------------------------------------- #
# Bitwise catch-up
# --------------------------------------------------------------------------- #


def consensus_state_root(commits: Mapping[int, WindowCommit]) -> str | None:
    """The state_root most miners committed for a window; None with no commits.

    Ties break to the lexicographically smallest root — any deterministic rule
    works because an honest majority makes ties impossible in practice.
    """
    if not commits:
        return None
    counts = Counter(c.state_root for c in commits.values())
    return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


@dataclass(frozen=True)
class CatchUpDivergence:
    """Evidence attached to a failed bitwise catch-up."""

    window: int          # the window whose θ_start disagreed
    expected_root: str   # consensus / certificate value
    actual_root: str     # locally recomputed state_root
    detail: str = ""


class CatchUpError(RuntimeError):
    """Catch-up cannot proceed in lockstep; ``divergence`` carries the report."""

    def __init__(self, message: str, divergence: CatchUpDivergence | None = None) -> None:
        super().__init__(message)
        self.divergence = divergence


class CertificatePendingError(CatchUpError):
    """A window has on-chain commits but no leader certificate YET.

    A liveness condition (leader down/lagging), not corruption: non-leader
    nodes must WAIT and retry — the certificate is the only thing that can
    define this window's outer step. Only the leader itself should treat this
    as an error (it is the one that must publish)."""


@dataclass(frozen=True)
class CatchUpReport:
    applied_windows: tuple[int, ...]
    skipped_void: tuple[int, ...]
    unverified_windows: tuple[int, ...]  # no chain commits found to check against
    final_root: str


async def catch_up(
    model_params: dict[str, torch.Tensor],
    outer_step: ReplicatedOuterStep,
    exchange: Any,
    storage: StorageClient,
    chain: Any,
    manifest: RunManifest,
    cfg: RunConfig,
    from_window: int,
    to_window: int,
    apply_fn: Callable[[CertifiedGather], None] | None = None,
    *,
    leader_bucket: BucketCreds,
    dense_names: Iterable[str] | None = None,
    max_bytes: int | None = None,
) -> CatchUpReport:
    """Replay windows ``(from_window, to_window]`` bitwise from certified artifacts.

    For each non-void window w: verify the current ``state_root`` against the
    consensus ``WindowCommit.state_root`` for w (and the certificate's
    ``theta_start_root``), fetch the certificate + aggregator mirror from the
    leader's bucket, rebuild the sparse contributions with
    :func:`build_outer_inputs` (the same code the live window_runner uses), and
    apply the outer step. Any root mismatch or missing certified payload raises
    :class:`CatchUpError` with a divergence report. ``manifest.is_void`` windows
    are skipped entirely. ``apply_fn`` replaces the default outer application
    (window_runner injects its GPU-side path); the state-root verification
    brackets it either way.

    ``exchange`` is the protocol I/O namespace (pass ``subnet.core.exchange`` or a
    test double providing ``get_certificate`` / ``gather_from_aggregator``);
    None selects the real module.
    """
    if to_window < from_window:
        raise ValueError(f"to_window {to_window} < from_window {from_window}")
    ex = _default_exchange if exchange is None else exchange
    budget = cfg.storage.max_payload_bytes if max_bytes is None else max_bytes

    dn = (
        frozenset(dense_names)
        if dense_names is not None
        else frozenset(n for n in model_params if n.endswith("balance_bias"))
    )
    unknown = dn - set(model_params)
    if unknown:
        raise ValueError(f"dense_names not in model_params: {sorted(unknown)}")
    comp_names = sorted(set(model_params) - dn)
    comp_shapes = {n: tuple(model_params[n].shape) for n in comp_names}
    dense_shapes = {n: tuple(model_params[n].shape) for n in sorted(dn)}

    ccfg = cfg.compression
    compressor = TopKCompressor(
        ChunkingTransformer(
            {n: model_params[n].shape for n in comp_names},
            target_chunk=ccfg.target_chunk,
            use_dct=ccfg.use_dct,
        ),
        Quantizer(bins=ccfg.quant_bins, range_sigmas=ccfg.quant_range_sigmas),
        topk=ccfg.topk,
    )

    applied: list[int] = []
    skipped: list[int] = []
    unverified: list[int] = []

    for w in range(from_window + 1, to_window + 1):
        if manifest.is_void(w):
            skipped.append(w)
            continue

        try:
            commits: dict[int, WindowCommit] = await asyncio.to_thread(chain.get_window_commits, w)
        except Exception as e:  # noqa: BLE001 — chain/RPC failures are retryable at the app layer
            raise CatchUpError(f"window {w}: chain read failed: {e}") from e
        current_root = state_root(model_params.items())
        want = consensus_state_root(commits)
        if want is None:
            unverified.append(w)
        elif want != current_root:
            raise CatchUpError(
                f"window {w}: local θ_start root {current_root} != chain consensus {want}",
                divergence=CatchUpDivergence(
                    window=w,
                    expected_root=want,
                    actual_root=current_root,
                    detail=f"consensus over {len(commits)} on-chain commits",
                ),
            )

        try:
            cert = await ex.get_certificate(storage, leader_bucket, w)
        except ObjectMissingError:
            if commits:
                # Miners committed but the leader has not certified (yet):
                # typically the leader is down or lagging. A WAIT condition for
                # everyone except the leader itself — never data corruption.
                raise CertificatePendingError(
                    f"window {w}: {len(commits)} on-chain commits but no leader certificate"
                ) from None
            # EMPTY window: no commits, no certificate — e.g. pre-launch windows,
            # fleet-wide downtime, or no eligible miners. The certified set is
            # empty, so the outer step is the identity: θ is unchanged. (The live
            # runner asserts the same: an empty certified set leaves state
            # bitwise unchanged.)
            if w not in unverified:
                unverified.append(w)
            log.info("empty window — no commits, no certificate; θ unchanged", window=w)
            continue
        for uid in cert.included_uids:
            commit = commits.get(uid)
            if commit is not None and not commit.binds_payload_hash(cert.payload_hashes.get(uid, "")):
                raise CatchUpError(
                    f"window {w}: certificate payload hash for uid {uid} contradicts chain commit"
                )
        if cert.theta_start_root != current_root:
            raise CatchUpError(
                f"window {w}: certificate theta_start_root disagrees with local state",
                divergence=CatchUpDivergence(
                    window=w,
                    expected_root=cert.theta_start_root,
                    actual_root=current_root,
                    detail="certificate theta_start_root mismatch",
                ),
            )

        gather = await ex.gather_from_aggregator(
            storage,
            cert,
            leader_bucket,
            expected_param_shapes=comp_shapes,
            expected_dense=dense_shapes,
            topk=ccfg.topk,
            target_chunk=ccfg.target_chunk,
            max_bytes=budget,
        )
        if gather.missing:
            raise CatchUpError(f"window {w}: certified payloads unavailable: {gather.missing}")

        if apply_fn is not None:
            apply_fn(gather)
        else:
            sparse, dense, norms = build_outer_inputs(gather.payloads, compressor, comp_names)
            outer_step.apply(model_params, sparse, dense, norms)
        applied.append(w)

    final_root = state_root(model_params.items())
    head_commits: dict[int, WindowCommit] = await asyncio.to_thread(
        chain.get_window_commits, to_window + 1
    )
    head_want = consensus_state_root(head_commits)
    if head_want is not None and head_want != final_root:
        raise CatchUpError(
            f"window {to_window}: post-catch-up root {final_root} != consensus θ_start "
            f"of window {to_window + 1} ({head_want})",
            divergence=CatchUpDivergence(
                window=to_window + 1,
                expected_root=head_want,
                actual_root=final_root,
                detail="post-apply verification against the next window's commits",
            ),
        )
    return CatchUpReport(
        applied_windows=tuple(applied),
        skipped_void=tuple(skipped),
        unverified_windows=tuple(unverified),
        final_root=final_root,
    )
