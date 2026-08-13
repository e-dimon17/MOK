"""Window payload: ownership assignment, deterministic wire format, structural validation.

Gradient-dict assembly with round-robin parameter ownership, strict
compressed-index validation, and an explicit byte-deterministic serialization
(canonical JSON header + ordered raw tensor blobs inside a single-threaded
zstd level-3 frame) instead of pickle-based payloads,
with all bounds checked before any allocation.

Wire format v1 (consensus surface — any change requires a SPEC_VERSION bump):

    frame  := b"MOKP" | u8 wire_version=1 | zstd(body)     (zstd: level 3, no checksum,
                                                            single thread, content size on)
    body   := u32le header_len | header_json | blobs
    header := canonical JSON (mok_core.config.canonical_bytes) listing uid, window,
              metadata, and per-tensor entries sorted by name with exact blob sizes
    blobs  := per compressed entry (header order): 12-bit-packed indices, then
              2-bit-packed codes; per dense entry (header order): fp32 LE bytes
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import torch
import zstandard

from mok_core.config.canonical import canonical_bytes
from mok_core.determinism.hashing import hash_bytes, tensor_bytes

from .compress import (
    SCALE_CEIL,
    SCALE_FLOOR,
    CompressedTensor,
    chunk_geometry,
    packed_nbytes_2bit,
    packed_nbytes_12bit,
    unpack_2bit_values,
    unpack_12bit_indices,
)

MAGIC = b"MOKP"
WIRE_VERSION = 1
ZSTD_LEVEL = 3

_MAX_SHAPE_DIMS = 8
_MAX_N_CHUNKS = 1 << 24
_MAX_CHUNK_ELEMS = 4096  # 12-bit index packing bound
_HEX_DIGITS = frozenset("0123456789abcdef")


class PayloadError(ValueError):
    """A payload failed structural or bounds validation."""


# --------------------------------------------------------------------------- #
# Ownership assignment
# --------------------------------------------------------------------------- #


def assign_owned_params(
    names: Iterable[str],
    rank: int,
    world_size: int,
    is_expert_local: Callable[[str], bool],
) -> set[str]:
    """Which parameters this rank compresses and uploads.

    Expert-local parameters (each EP rank holds its own expert shard) are always
    owned by the local rank. All other (replicated) parameters are partitioned
    round-robin over sorted names: sorted index % world == rank.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    name_list = list(names)
    if len(set(name_list)) != len(name_list):
        raise ValueError("duplicate parameter names")
    owned = {name for name in name_list if is_expert_local(name)}
    shared = sorted(name for name in name_list if name not in owned)
    owned.update(name for i, name in enumerate(shared) if i % world_size == rank)
    return owned


# --------------------------------------------------------------------------- #
# Payload objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PayloadMeta:
    """Per-window commitments carried alongside the compressed tensors."""

    sample_digest: str      # hex: PRF-assigned data sample commitment
    sample_count: int
    theta_end_hash: str     # hex: H(theta_end) after the inner loop
    state_root: str         # hex: master-weight state root
    global_step: int
    spec_version: int


@dataclass(eq=False)
class WindowPayload:
    """One miner-window upload: sparse pseudo-gradients + small dense tensors."""

    uid: int
    window: int
    compressed: dict[str, CompressedTensor]
    dense: dict[str, torch.Tensor]          # fp32 router balance biases — small
    metadata: PayloadMeta


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #


def _qparams_wire(name: str, qparams: Mapping[str, float | torch.Tensor]) -> dict[str, object]:
    try:
        shift = float(qparams["shift"])  # type: ignore[arg-type]
        scale = float(qparams["scale"])  # type: ignore[arg-type]
        lookup = qparams["lookup"]
    except (KeyError, TypeError) as e:
        raise PayloadError(f"{name}: malformed qparams: {e}") from e
    if not isinstance(lookup, torch.Tensor):
        lookup = torch.tensor(lookup, dtype=torch.float32)
    lookup_vals = [float(x) for x in lookup.detach().to(torch.float32).flatten().tolist()]
    for label, values in (("shift", [shift]), ("scale", [scale]), ("lookup", lookup_vals)):
        if not all(math.isfinite(v) for v in values):
            raise PayloadError(f"{name}: non-finite qparams field {label!r}")
    return {"shift": shift, "scale": scale, "lookup": lookup_vals}


def serialize(payload: WindowPayload) -> bytes:
    """Deterministic wire bytes; identical payloads serialize to identical bytes."""
    comp_entries: list[dict[str, object]] = []
    blobs: list[bytes] = []
    for name in sorted(payload.compressed):
        ct = payload.compressed[name]
        if ct.idxs_packed.dtype != torch.uint8 or ct.codes_packed.dtype != torch.uint8:
            raise PayloadError(f"{name}: packed tensors must be uint8")
        n = ct.n_chunks * ct.topk
        idxs_raw = tensor_bytes(ct.idxs_packed)
        codes_raw = tensor_bytes(ct.codes_packed)
        if len(idxs_raw) != packed_nbytes_12bit(n) or len(codes_raw) != packed_nbytes_2bit(n):
            raise PayloadError(f"{name}: packed byte length inconsistent with n_chunks*topk={n}")
        comp_entries.append(
            {
                "name": name,
                "n_chunks": int(ct.n_chunks),
                "chunk_elems": int(ct.chunk_elems),
                "topk": int(ct.topk),
                "orig_shape": [int(s) for s in ct.orig_shape],
                "qparams": _qparams_wire(name, ct.qparams),
                "idxs_bytes": len(idxs_raw),
                "codes_bytes": len(codes_raw),
            }
        )
        blobs.append(idxs_raw)
        blobs.append(codes_raw)

    dense_entries: list[dict[str, object]] = []
    for name in sorted(payload.dense):
        t = payload.dense[name]
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
            raise PayloadError(f"{name}: dense tensors must be fp32")
        if not bool(torch.isfinite(t).all()):
            raise PayloadError(f"{name}: dense tensor has non-finite values")
        raw = tensor_bytes(t)
        dense_entries.append(
            {"name": name, "shape": [int(s) for s in t.shape], "numel_bytes": len(raw)}
        )
        blobs.append(raw)

    if set(payload.compressed) & set(payload.dense):
        raise PayloadError("compressed and dense name sets must be disjoint")

    meta = payload.metadata
    header = {
        "uid": int(payload.uid),
        "window": int(payload.window),
        "meta": {
            "sample_digest": str(meta.sample_digest),
            "sample_count": int(meta.sample_count),
            "theta_end_hash": str(meta.theta_end_hash),
            "state_root": str(meta.state_root),
            "global_step": int(meta.global_step),
            "spec_version": int(meta.spec_version),
        },
        "compressed": comp_entries,
        "dense": dense_entries,
    }
    header_raw = canonical_bytes(header)
    body = len(header_raw).to_bytes(4, "little") + header_raw + b"".join(blobs)
    cctx = zstandard.ZstdCompressor(level=ZSTD_LEVEL, write_checksum=False, threads=0)
    return MAGIC + bytes([WIRE_VERSION]) + cctx.compress(body)


def canonical_payload_hash(payload: WindowPayload) -> str:
    """Hex blake2b-256 over the serialized bytes — the on-chain H(payload) commitment."""
    return hash_bytes(serialize(payload))


# --------------------------------------------------------------------------- #
# Deserialization (bounds checked before allocation)
# --------------------------------------------------------------------------- #


def _req(entry: Mapping[str, object], key: str, ctx: str) -> object:
    if key not in entry:
        raise PayloadError(f"{ctx}: missing field {key!r}")
    return entry[key]


def _req_int(entry: Mapping[str, object], key: str, ctx: str, lo: int, hi: int) -> int:
    v = _req(entry, key, ctx)
    if type(v) is not int:  # bool is an int subclass — rejected on purpose
        raise PayloadError(f"{ctx}: field {key!r} must be an integer")
    if not lo <= v <= hi:
        raise PayloadError(f"{ctx}: field {key!r}={v} outside [{lo}, {hi}]")
    return v


def _req_str(entry: Mapping[str, object], key: str, ctx: str) -> str:
    v = _req(entry, key, ctx)
    if type(v) is not str:
        raise PayloadError(f"{ctx}: field {key!r} must be a string")
    return v


def _req_float(entry: Mapping[str, object], key: str, ctx: str) -> float:
    v = _req(entry, key, ctx)
    if type(v) not in (int, float):
        raise PayloadError(f"{ctx}: field {key!r} must be a number")
    f = float(v)  # type: ignore[arg-type]
    if not math.isfinite(f):
        raise PayloadError(f"{ctx}: field {key!r} is not finite")
    return f


def _req_keys(entry: Mapping[str, object], keys: frozenset[str], ctx: str) -> None:
    if set(entry) != keys:
        raise PayloadError(f"{ctx}: fields {sorted(entry)} != expected {sorted(keys)}")


def _req_shape(entry: Mapping[str, object], key: str, ctx: str) -> tuple[int, ...]:
    v = _req(entry, key, ctx)
    if not isinstance(v, list) or len(v) > _MAX_SHAPE_DIMS:
        raise PayloadError(f"{ctx}: field {key!r} must be a shape list of <= {_MAX_SHAPE_DIMS} dims")
    for s in v:
        if type(s) is not int or s < 1:
            raise PayloadError(f"{ctx}: shape dims must be positive integers, got {v}")
    return tuple(v)


_COMP_KEYS = frozenset(
    {"name", "n_chunks", "chunk_elems", "topk", "orig_shape", "qparams", "idxs_bytes", "codes_bytes"}
)
_DENSE_KEYS = frozenset({"name", "shape", "numel_bytes"})
_META_KEYS = frozenset(
    {"sample_digest", "sample_count", "theta_end_hash", "state_root", "global_step", "spec_version"}
)
_HEADER_KEYS = frozenset({"uid", "window", "meta", "compressed", "dense"})
_QPARAM_KEYS = frozenset({"shift", "scale", "lookup"})
_I64 = (1 << 63) - 1


def deserialize(
    data: bytes,
    *,
    max_bytes: int,
    max_decompressed_bytes: int | None = None,
) -> WindowPayload:
    """Parse wire bytes into a WindowPayload with strict bounds checks before allocating.

    `max_bytes` bounds the compressed frame; the decompressed body is bounded by
    `max_decompressed_bytes` (default 4 * max_bytes) and the zstd frame must declare
    its content size upfront. Every blob length is verified against the header, and
    the header itself must re-serialize canonically (byte-for-byte).
    """
    if max_bytes < 1:
        raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
    data = bytes(data)
    if len(data) > max_bytes:
        raise PayloadError(f"payload {len(data)} bytes exceeds max_bytes {max_bytes}")
    if len(data) < 6:
        raise PayloadError("payload too short to be a MOKP frame")
    if data[:4] != MAGIC:
        raise PayloadError(f"bad magic {data[:4]!r}")
    if data[4] != WIRE_VERSION:
        raise PayloadError(f"unsupported wire version {data[4]}")

    budget = 4 * max_bytes if max_decompressed_bytes is None else max_decompressed_bytes
    frame = data[5:]
    try:
        content_size = zstandard.frame_content_size(frame)
    except zstandard.ZstdError as e:
        raise PayloadError(f"invalid zstd frame: {e}") from e
    if content_size < 0 or content_size > budget:
        raise PayloadError(f"declared content size {content_size} outside (0, {budget}]")
    try:
        body = zstandard.ZstdDecompressor().decompress(frame, max_output_size=budget)
    except zstandard.ZstdError as e:
        raise PayloadError(f"zstd decompression failed: {e}") from e

    if len(body) < 4:
        raise PayloadError("body too short for header length")
    header_len = int.from_bytes(body[:4], "little")
    if not 2 <= header_len <= len(body) - 4:
        raise PayloadError(f"header length {header_len} out of range for body {len(body)}")
    header_raw = body[4 : 4 + header_len]
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PayloadError(f"header is not valid JSON: {e}") from e
    if not isinstance(header, dict):
        raise PayloadError("header must be a JSON object")
    try:
        recanonical = canonical_bytes(header)
    except (TypeError, ValueError) as e:
        raise PayloadError(f"header not canonically serializable: {e}") from e
    if recanonical != header_raw:
        raise PayloadError("header bytes are not in canonical form")

    _req_keys(header, _HEADER_KEYS, "header")
    uid = _req_int(header, "uid", "header", 0, _I64)
    window = _req_int(header, "window", "header", 0, _I64)

    meta_raw = _req(header, "meta", "header")
    if not isinstance(meta_raw, dict):
        raise PayloadError("header.meta must be an object")
    _req_keys(meta_raw, _META_KEYS, "meta")
    meta = PayloadMeta(
        sample_digest=_req_str(meta_raw, "sample_digest", "meta"),
        sample_count=_req_int(meta_raw, "sample_count", "meta", 0, _I64),
        theta_end_hash=_req_str(meta_raw, "theta_end_hash", "meta"),
        state_root=_req_str(meta_raw, "state_root", "meta"),
        global_step=_req_int(meta_raw, "global_step", "meta", 0, _I64),
        spec_version=_req_int(meta_raw, "spec_version", "meta", 1, _I64),
    )

    comp_raw = _req(header, "compressed", "header")
    dense_raw = _req(header, "dense", "header")
    if not isinstance(comp_raw, list) or not isinstance(dense_raw, list):
        raise PayloadError("header.compressed and header.dense must be lists")

    # ---- validate every entry and total blob size BEFORE building any tensor ----
    comp_parsed: list[tuple[str, int, int, int, tuple[int, ...], dict[str, object], int, int]] = []
    total_blob = 0
    prev_name = ""
    for entry in comp_raw:
        if not isinstance(entry, dict):
            raise PayloadError("compressed entries must be objects")
        _req_keys(entry, _COMP_KEYS, "compressed entry")
        name = _req_str(entry, "name", "compressed entry")
        ctx = f"compressed[{name}]"
        if name <= prev_name:
            raise PayloadError(f"{ctx}: names must be strictly sorted")
        prev_name = name
        n_chunks = _req_int(entry, "n_chunks", ctx, 1, _MAX_N_CHUNKS)
        chunk_elems = _req_int(entry, "chunk_elems", ctx, 1, _MAX_CHUNK_ELEMS)
        topk = _req_int(entry, "topk", ctx, 1, chunk_elems)
        orig_shape = _req_shape(entry, "orig_shape", ctx)
        if math.prod(orig_shape) > n_chunks * chunk_elems:
            raise PayloadError(f"{ctx}: orig_shape {orig_shape} exceeds chunk capacity")
        n = n_chunks * topk
        idxs_bytes = _req_int(entry, "idxs_bytes", ctx, 0, _I64)
        codes_bytes = _req_int(entry, "codes_bytes", ctx, 0, _I64)
        if idxs_bytes != packed_nbytes_12bit(n) or codes_bytes != packed_nbytes_2bit(n):
            raise PayloadError(f"{ctx}: declared blob sizes inconsistent with n_chunks*topk={n}")
        qp_raw = _req(entry, "qparams", ctx)
        if not isinstance(qp_raw, dict):
            raise PayloadError(f"{ctx}: qparams must be an object")
        _req_keys(qp_raw, _QPARAM_KEYS, f"{ctx}.qparams")
        shift = _req_float(qp_raw, "shift", f"{ctx}.qparams")
        scale = _req_float(qp_raw, "scale", f"{ctx}.qparams")
        lookup_raw = _req(qp_raw, "lookup", f"{ctx}.qparams")
        if not isinstance(lookup_raw, list) or not 1 <= len(lookup_raw) <= 256:
            raise PayloadError(f"{ctx}: qparams.lookup must be a list of 1..256 numbers")
        lookup_vals: list[float] = []
        for x in lookup_raw:
            if type(x) not in (int, float) or not math.isfinite(float(x)):
                raise PayloadError(f"{ctx}: qparams.lookup entries must be finite numbers")
            lookup_vals.append(float(x))
        qparams: dict[str, object] = {"shift": shift, "scale": scale, "lookup": lookup_vals}
        comp_parsed.append((name, n_chunks, chunk_elems, topk, orig_shape, qparams, idxs_bytes, codes_bytes))
        total_blob += idxs_bytes + codes_bytes

    dense_parsed: list[tuple[str, tuple[int, ...], int]] = []
    prev_name = ""
    comp_names = {c[0] for c in comp_parsed}
    for entry in dense_raw:
        if not isinstance(entry, dict):
            raise PayloadError("dense entries must be objects")
        _req_keys(entry, _DENSE_KEYS, "dense entry")
        name = _req_str(entry, "name", "dense entry")
        ctx = f"dense[{name}]"
        if name <= prev_name:
            raise PayloadError(f"{ctx}: names must be strictly sorted")
        prev_name = name
        if name in comp_names:
            raise PayloadError(f"{ctx}: name collides with a compressed entry")
        shape = _req_shape(entry, "shape", ctx)
        numel_bytes = _req_int(entry, "numel_bytes", ctx, 4, _I64)
        if numel_bytes != math.prod(shape) * 4:
            raise PayloadError(f"{ctx}: numel_bytes {numel_bytes} != 4 * prod{shape}")
        dense_parsed.append((name, shape, numel_bytes))
        total_blob += numel_bytes

    if len(body) != 4 + header_len + total_blob:
        raise PayloadError(
            f"body length {len(body)} != header ({4 + header_len}) + declared blobs ({total_blob})"
        )

    # ---- all bounds verified: build tensors ----
    off = 4 + header_len
    compressed: dict[str, CompressedTensor] = {}
    for name, n_chunks, chunk_elems, topk, orig_shape, qparams, idxs_bytes, codes_bytes in comp_parsed:
        idxs = torch.frombuffer(bytearray(body[off : off + idxs_bytes]), dtype=torch.uint8)
        off += idxs_bytes
        codes = torch.frombuffer(bytearray(body[off : off + codes_bytes]), dtype=torch.uint8)
        off += codes_bytes
        compressed[name] = CompressedTensor(
            idxs_packed=idxs,
            codes_packed=codes,
            qparams={
                "shift": qparams["shift"],
                "scale": qparams["scale"],
                "lookup": torch.tensor(qparams["lookup"], dtype=torch.float32),
            },
            n_chunks=n_chunks,
            chunk_elems=chunk_elems,
            orig_shape=orig_shape,
            topk=topk,
        )

    dense: dict[str, torch.Tensor] = {}
    for name, shape, numel_bytes in dense_parsed:
        arr = np.frombuffer(body[off : off + numel_bytes], dtype="<f4").astype(np.float32)
        off += numel_bytes
        dense[name] = torch.from_numpy(arr).reshape(shape)

    return WindowPayload(uid=uid, window=window, compressed=compressed, dense=dense, metadata=meta)


# --------------------------------------------------------------------------- #
# Structural validation (protocol level, after deserialize)
# --------------------------------------------------------------------------- #


def _is_hex_digest(s: str) -> bool:
    return len(s) == 64 and set(s) <= _HEX_DIGITS


def validate_structure(
    payload: WindowPayload,
    expected_param_shapes: Mapping[str, tuple[int, ...]],
    expected_dense: Mapping[str, tuple[int, ...]] | set[str] | frozenset[str],
    topk: int,
    *,
    target_chunk: int = 64,
) -> None:
    """Reject a payload whose contents cannot enter the outer step.

    Checks: exact name sets, exact chunk
    geometry per parameter, exact per-chunk top-k with strictly increasing in-bounds
    indices, canonical zero pad bits, code/lookup consistency, finite qparams with
    scale in [SCALE_FLOOR, SCALE_CEIL], dense fp32/finite (+ shapes when a mapping is
    given), and well-formed metadata digests. Raises PayloadError on the first failure.
    """
    if set(payload.compressed) != set(expected_param_shapes):
        missing = sorted(set(expected_param_shapes) - set(payload.compressed))
        extra = sorted(set(payload.compressed) - set(expected_param_shapes))
        raise PayloadError(f"compressed name set mismatch: missing={missing} extra={extra}")
    expected_dense_names = set(expected_dense)
    if set(payload.dense) != expected_dense_names:
        missing = sorted(expected_dense_names - set(payload.dense))
        extra = sorted(set(payload.dense) - expected_dense_names)
        raise PayloadError(f"dense name set mismatch: missing={missing} extra={extra}")

    for name in sorted(payload.compressed):
        ct = payload.compressed[name]
        ctx = f"compressed[{name}]"
        expected_shape = tuple(int(s) for s in expected_param_shapes[name])
        if tuple(ct.orig_shape) != expected_shape:
            raise PayloadError(f"{ctx}: orig_shape {tuple(ct.orig_shape)} != expected {expected_shape}")
        geom = chunk_geometry(expected_shape, target_chunk)
        if (ct.n_chunks, ct.chunk_elems) != (geom.n_chunks, geom.chunk_elems):
            raise PayloadError(
                f"{ctx}: geometry ({ct.n_chunks}, {ct.chunk_elems}) "
                f"!= expected ({geom.n_chunks}, {geom.chunk_elems})"
            )
        expected_topk = min(topk, geom.chunk_elems)
        if ct.topk != expected_topk:
            raise PayloadError(f"{ctx}: topk {ct.topk} != expected {expected_topk}")
        n = ct.n_chunks * ct.topk

        if ct.idxs_packed.dtype != torch.uint8 or ct.idxs_packed.dim() != 1:
            raise PayloadError(f"{ctx}: idxs_packed must be a 1-D uint8 tensor")
        if ct.codes_packed.dtype != torch.uint8 or ct.codes_packed.dim() != 1:
            raise PayloadError(f"{ctx}: codes_packed must be a 1-D uint8 tensor")
        if ct.idxs_packed.numel() != packed_nbytes_12bit(n):
            raise PayloadError(f"{ctx}: idxs_packed length {ct.idxs_packed.numel()} wrong for {n} indices")
        if ct.codes_packed.numel() != packed_nbytes_2bit(n):
            raise PayloadError(f"{ctx}: codes_packed length {ct.codes_packed.numel()} wrong for {n} codes")
        if n % 2 and int(ct.idxs_packed[-1].item()) & 0xF0:
            raise PayloadError(f"{ctx}: nonzero pad bits in final index byte")
        if n % 4 and int(ct.codes_packed[-1].item()) >> (2 * (n % 4)):
            raise PayloadError(f"{ctx}: nonzero pad bits in final code byte")

        idx = unpack_12bit_indices(ct.idxs_packed, n).reshape(ct.n_chunks, ct.topk)
        if int(idx.max().item()) >= ct.chunk_elems:
            bad = int(idx.max().item())
            raise PayloadError(f"{ctx}: index {bad} out of bounds (chunk_elems={ct.chunk_elems})")
        if ct.topk > 1 and not bool((idx[:, 1:] > idx[:, :-1]).all()):
            raise PayloadError(f"{ctx}: per-chunk indices must be strictly increasing (exact top-k)")

        qparams = ct.qparams
        try:
            shift = float(qparams["shift"])  # type: ignore[arg-type]
            scale = float(qparams["scale"])  # type: ignore[arg-type]
            lookup = qparams["lookup"]
        except (KeyError, TypeError) as e:
            raise PayloadError(f"{ctx}: malformed qparams: {e}") from e
        if not math.isfinite(shift):
            raise PayloadError(f"{ctx}: non-finite qparams shift")
        if not math.isfinite(scale) or not SCALE_FLOOR <= scale <= SCALE_CEIL:
            raise PayloadError(f"{ctx}: scale {scale} outside [{SCALE_FLOOR}, {SCALE_CEIL}]")
        if not isinstance(lookup, torch.Tensor):
            raise PayloadError(f"{ctx}: qparams lookup must be a tensor")
        if lookup.dtype != torch.float32 or lookup.dim() != 1 or not 1 <= lookup.numel() <= 4:
            raise PayloadError(f"{ctx}: lookup must be a 1-D fp32 tensor of 1..4 bins")
        if not bool(torch.isfinite(lookup).all()):
            raise PayloadError(f"{ctx}: non-finite lookup values")
        codes = unpack_2bit_values(ct.codes_packed, n)
        if int(codes.max().item()) >= lookup.numel():
            raise PayloadError(f"{ctx}: code {int(codes.max().item())} >= lookup bins {lookup.numel()}")

    dense_shapes = expected_dense if isinstance(expected_dense, Mapping) else None
    for name in sorted(payload.dense):
        t = payload.dense[name]
        ctx = f"dense[{name}]"
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float32:
            raise PayloadError(f"{ctx}: must be an fp32 tensor")
        if dense_shapes is not None:
            expected_shape = tuple(int(s) for s in dense_shapes[name])
            if tuple(t.shape) != expected_shape:
                raise PayloadError(f"{ctx}: shape {tuple(t.shape)} != expected {expected_shape}")
        if not bool(torch.isfinite(t).all()):
            raise PayloadError(f"{ctx}: non-finite values")

    meta = payload.metadata
    for field_name, value in (
        ("sample_digest", meta.sample_digest),
        ("theta_end_hash", meta.theta_end_hash),
        ("state_root", meta.state_root),
    ):
        if not _is_hex_digest(value):
            raise PayloadError(f"metadata.{field_name} is not a 64-char lowercase hex digest")
    if meta.sample_count < 0 or meta.global_step < 0 or meta.spec_version < 1:
        raise PayloadError("metadata counters out of range")
