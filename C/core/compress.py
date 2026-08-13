"""SparseLoCo compression: chunking, top-k selection, 2-bit quantization, error feedback.

Chunked top-k compression in the DeMo/SparseLoCo family, with these
protocol-level properties:

  - Chunk geometry is a pure function of (shape, target_chunk): tensors are zero-padded
    to full ``target_chunk x target_chunk`` blocks (no divisor search),
    so every parameter encodes to a uniform ``[n_chunks, target_chunk**2]`` layout.
  - 12-bit index packing handles odd counts.
  - Values travel as 2-bit codes (4 per byte).
  - The DCT path is accepted as a config flag but must be False (never implemented here);
    quantization uses mean-shift + sigma-scaled uniform bins + per-bin-mean dequant.

Determinism: all consensus-relevant arithmetic is fp32 with a fixed reduction order.
Compression runs only on the producing miner and its auditor replay, which share a
pinned container — the wire bytes, not the float ops, are the consensus surface.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from mok_core.determinism.hashing import tensor_bytes

# fp32-rounded floor for the quantizer scale; the wire validator uses the same constant.
SCALE_FLOOR: float = float(torch.tensor(1e-12, dtype=torch.float32).item())
SCALE_CEIL: float = 1e4

_MERKLE_CHUNK_BYTES = 1 << 20


# --------------------------------------------------------------------------- #
# Bit packing
# --------------------------------------------------------------------------- #


def packed_nbytes_12bit(count: int) -> int:
    """Wire size of `count` 12-bit indices: 3 bytes per pair, 2 for an odd tail."""
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    return (count // 2) * 3 + (2 if count % 2 else 0)


def packed_nbytes_2bit(count: int) -> int:
    """Wire size of `count` 2-bit codes: 4 per byte, zero-padded tail."""
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    return (count + 3) // 4


def pack_12bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack integer indices in [0, 4096) into uint8, 2 indices per 3 bytes.

    Pair layout: byte0 = idx1[7:0]; byte1 = idx1[11:8] | idx2[3:0] << 4;
    byte2 = idx2[11:4]. An odd tail index packs into 2 bytes: byte0 = idx[7:0],
    byte1 = idx[11:8] (high nibble zero).
    """
    flat = indices.detach().flatten()
    n = flat.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.uint8, device=indices.device)
    if not flat.dtype.is_floating_point and flat.dtype != torch.bool:
        flat = flat.to(torch.int64)
    else:
        raise ValueError(f"indices must be an integer tensor, got {indices.dtype}")
    lo, hi = int(flat.min().item()), int(flat.max().item())
    if lo < 0 or hi >= 4096:
        raise ValueError(f"indices must be in [0, 4096), got range [{lo}, {hi}]")

    packed = torch.zeros(packed_nbytes_12bit(n), dtype=torch.uint8, device=flat.device)
    n_pairs = n // 2
    if n_pairs > 0:
        pairs = flat[: 2 * n_pairs].reshape(-1, 2)
        idx1, idx2 = pairs[:, 0], pairs[:, 1]
        body = packed[: 3 * n_pairs]
        body[0::3] = (idx1 & 0xFF).to(torch.uint8)
        body[1::3] = (((idx1 >> 8) & 0x0F) | ((idx2 & 0x0F) << 4)).to(torch.uint8)
        body[2::3] = ((idx2 >> 4) & 0xFF).to(torch.uint8)
    if n % 2:
        tail = int(flat[-1].item())
        packed[-2] = tail & 0xFF
        packed[-1] = (tail >> 8) & 0x0F
    return packed


def unpack_12bit_indices(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack `count` 12-bit indices from uint8 wire bytes to a 1-D int64 tensor."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed indices must be uint8, got {packed.dtype}")
    expected = packed_nbytes_12bit(count)
    if packed.numel() != expected:
        raise ValueError(f"packed length {packed.numel()} != expected {expected} for count {count}")
    out = torch.zeros(count, dtype=torch.int64, device=packed.device)
    if count == 0:
        return out
    flat = packed.flatten().to(torch.int64)
    n_pairs = count // 2
    if n_pairs > 0:
        body = flat[: 3 * n_pairs]
        byte0, byte1, byte2 = body[0::3], body[1::3], body[2::3]
        out[0 : 2 * n_pairs : 2] = byte0 | ((byte1 & 0x0F) << 8)
        out[1 : 2 * n_pairs : 2] = ((byte1 >> 4) & 0x0F) | (byte2 << 4)
    if count % 2:
        out[-1] = flat[-2] | ((flat[-1] & 0x0F) << 8)
    return out


def pack_2bit_values(codes: torch.Tensor) -> torch.Tensor:
    """Pack uint8 codes in [0, 4) into uint8, 4 codes per byte (code i at bits 2*(i%4))."""
    flat = codes.detach().flatten()
    if flat.dtype != torch.uint8:
        raise ValueError(f"codes must be uint8, got {codes.dtype}")
    n = flat.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.uint8, device=codes.device)
    if int(flat.max().item()) >= 4:
        raise ValueError(f"codes must be in [0, 4), got max {int(flat.max().item())}")
    padded = flat
    if n % 4:
        padded = torch.cat([flat, flat.new_zeros(4 - n % 4)])
    quads = padded.reshape(-1, 4).to(torch.int64)
    packed = quads[:, 0] | (quads[:, 1] << 2) | (quads[:, 2] << 4) | (quads[:, 3] << 6)
    return packed.to(torch.uint8)


def unpack_2bit_values(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack `count` 2-bit codes from uint8 wire bytes to a 1-D uint8 tensor."""
    if packed.dtype != torch.uint8:
        raise ValueError(f"packed codes must be uint8, got {packed.dtype}")
    expected = packed_nbytes_2bit(count)
    if packed.numel() != expected:
        raise ValueError(f"packed length {packed.numel()} != expected {expected} for count {count}")
    if count == 0:
        return torch.zeros(0, dtype=torch.uint8, device=packed.device)
    flat = packed.flatten().to(torch.int64)
    quads = torch.stack([flat & 0x3, (flat >> 2) & 0x3, (flat >> 4) & 0x3, (flat >> 6) & 0x3], dim=1)
    return quads.reshape(-1)[:count].to(torch.uint8)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkGeometry:
    """Pure function of (shape, target_chunk) — replicated by payload validation."""

    orig_shape: tuple[int, ...]
    mode: Literal["flat", "grid"]
    rows: int          # grid: logical row count (prod of leading dims); flat: numel
    cols: int          # grid: last-dim size; flat: 1
    pad_rows: int      # grid: rows padded to target_chunk multiple; flat: padded numel
    pad_cols: int      # grid: cols padded to target_chunk multiple; flat: 1
    n_chunks: int
    chunk_elems: int

    @property
    def numel(self) -> int:
        return self.rows * self.cols

    @property
    def padded_numel(self) -> int:
        return self.n_chunks * self.chunk_elems


def chunk_geometry(shape: tuple[int, ...] | torch.Size, target_chunk: int) -> ChunkGeometry:
    """Chunk layout for a parameter shape: >=2-D tensors tile as (rows, last-dim) into
    target_chunk x target_chunk blocks; 0-D/1-D tensors chunk flat to target_chunk**2."""
    shape_t = tuple(int(s) for s in shape)
    if target_chunk < 1:
        raise ValueError(f"target_chunk must be >= 1, got {target_chunk}")
    if any(s < 1 for s in shape_t):
        raise ValueError(f"zero-sized parameter shape {shape_t} is not compressible")
    chunk_elems = target_chunk * target_chunk
    if len(shape_t) >= 2:
        rows = math.prod(shape_t[:-1])
        cols = shape_t[-1]
        pad_rows = -(-rows // target_chunk) * target_chunk
        pad_cols = -(-cols // target_chunk) * target_chunk
        n_chunks = (pad_rows // target_chunk) * (pad_cols // target_chunk)
        return ChunkGeometry(shape_t, "grid", rows, cols, pad_rows, pad_cols, n_chunks, chunk_elems)
    numel = shape_t[0] if shape_t else 1
    n_chunks = -(-numel // chunk_elems)
    return ChunkGeometry(shape_t, "flat", numel, 1, n_chunks * chunk_elems, 1, n_chunks, chunk_elems)


class ChunkingTransformer:
    """Maps registered parameters to and from a uniform ``[n_chunks, chunk_elems]`` layout.

    `use_dct` mirrors CompressionConfig but the DCT math is not part
    of wire format v1 — passing True is rejected here rather than silently ignored.
    """

    def __init__(
        self,
        param_shapes: Mapping[str, torch.Size | tuple[int, ...]],
        target_chunk: int = 64,
        *,
        use_dct: bool = False,
    ):
        if use_dct:
            raise ValueError("use_dct=True is not part of wire format v1 (SPEC_VERSION bump required)")
        self.use_dct = False
        self.target_chunk = int(target_chunk)
        self._geom: dict[str, ChunkGeometry] = {
            name: chunk_geometry(shape, self.target_chunk) for name, shape in param_shapes.items()
        }

    def geometry(self, name: str) -> ChunkGeometry:
        geom = self._geom.get(name)
        if geom is None:
            raise KeyError(f"parameter {name!r} not registered with ChunkingTransformer")
        return geom

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._geom)

    @torch.no_grad()
    def encode(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Chunk `tensor` to ``[n_chunks, chunk_elems]``, zero-padding partial chunks."""
        g = self.geometry(name)
        if tuple(tensor.shape) != g.orig_shape:
            raise ValueError(f"{name}: shape {tuple(tensor.shape)} != registered {g.orig_shape}")
        t = tensor.detach()
        tc = self.target_chunk
        if g.mode == "grid":
            t2 = t.reshape(g.rows, g.cols)
            t2 = F.pad(t2, (0, g.pad_cols - g.cols, 0, g.pad_rows - g.rows))
            blocks = t2.reshape(g.pad_rows // tc, tc, g.pad_cols // tc, tc)
            return blocks.permute(0, 2, 1, 3).reshape(g.n_chunks, g.chunk_elems)
        flat = t.reshape(-1)
        pad = g.padded_numel - flat.numel()
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        return flat.reshape(g.n_chunks, g.chunk_elems)

    @torch.no_grad()
    def decode(self, name: str, chunked: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`encode`: strip padding and restore the original shape."""
        g = self.geometry(name)
        if tuple(chunked.shape) != (g.n_chunks, g.chunk_elems):
            raise ValueError(
                f"{name}: chunked shape {tuple(chunked.shape)} != ({g.n_chunks}, {g.chunk_elems})"
            )
        tc = self.target_chunk
        if g.mode == "grid":
            blocks = chunked.reshape(g.pad_rows // tc, g.pad_cols // tc, tc, tc)
            full = blocks.permute(0, 2, 1, 3).reshape(g.pad_rows, g.pad_cols)
            return full[: g.rows, : g.cols].reshape(g.orig_shape)
        return chunked.reshape(-1)[: g.numel].reshape(g.orig_shape)


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #


class Quantizer:
    """Mean-shift + sigma-scaled uniform binning with a per-bin-mean dequant lookup.

    Mean-shift sigma-scaled uniform bins with n_bins=4 (2-bit codes). qparams are
    plain floats (exact fp32 values) plus an fp32 lookup tensor so they serialize
    canonically. Degenerate all-equal inputs floor the scale at SCALE_FLOOR and
    dequantize exactly (all values land in the center bin whose mean offset is 0).
    """

    def __init__(self, bins: int = 4, range_sigmas: float = 6.0):
        if not 2 <= bins <= 256:
            raise ValueError(f"bins must be in [2, 256], got {bins}")
        if range_sigmas <= 0:
            raise ValueError(f"range_sigmas must be positive, got {range_sigmas}")
        self.bins = int(bins)
        self.range_sigmas = float(range_sigmas)

    @torch.no_grad()
    def quantize(self, vals: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | torch.Tensor]]:
        """fp32 values [n] -> (codes uint8 [n], {shift, scale, lookup fp32[bins]})."""
        v = vals.detach().to(torch.float32).flatten()
        n = v.numel()
        if n == 0:
            raise ValueError("cannot quantize an empty tensor")
        if not bool(torch.isfinite(v).all()):
            raise ValueError("cannot quantize non-finite values")
        shift_t = v.mean()
        centered = v - shift_t
        if n > 1:
            std = centered.norm() / math.sqrt(n - 1)
            scale = float((std * (self.range_sigmas / self.bins)).item())
        else:
            scale = 0.0
        if not math.isfinite(scale) or scale < SCALE_FLOOR:
            scale = SCALE_FLOOR
        offset = self.bins // 2
        codes = (centered / scale + offset).round().clamp_(0, self.bins - 1).to(torch.uint8)

        bin_idx = codes.to(torch.int64)
        sums = torch.zeros(self.bins, dtype=torch.float32)
        counts = torch.zeros(self.bins, dtype=torch.float32)
        sums.scatter_add_(0, bin_idx, centered)
        counts.scatter_add_(0, bin_idx, torch.ones_like(centered))
        lookup = torch.where(counts > 0, sums / counts.clamp(min=1.0), torch.zeros(()))
        qparams: dict[str, float | torch.Tensor] = {
            "shift": float(shift_t.item()),
            "scale": scale,
            "lookup": lookup,
        }
        return codes, qparams

    @torch.no_grad()
    def dequantize(self, codes: torch.Tensor, qparams: Mapping[str, float | torch.Tensor]) -> torch.Tensor:
        """uint8 codes [n] -> fp32 values [n] via lookup[code] + shift."""
        if codes.dtype != torch.uint8:
            raise ValueError(f"codes must be uint8, got {codes.dtype}")
        lookup = qparams["lookup"]
        if not isinstance(lookup, torch.Tensor):
            lookup = torch.tensor(lookup, dtype=torch.float32)
        lookup = lookup.to(torch.float32).flatten()
        if codes.numel() and int(codes.max().item()) >= lookup.numel():
            raise ValueError(f"code {int(codes.max().item())} out of lookup range {lookup.numel()}")
        return lookup[codes.to(torch.int64)] + float(qparams["shift"])


# --------------------------------------------------------------------------- #
# Top-k compression
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class CompressedTensor:
    """One parameter's wire-ready sparse form: per-chunk top-k, packed and quantized."""

    idxs_packed: torch.Tensor       # uint8, 12-bit-packed indices, n_chunks*topk entries
    codes_packed: torch.Tensor      # uint8, 2-bit-packed codes, n_chunks*topk entries
    qparams: dict[str, float | torch.Tensor]
    n_chunks: int
    chunk_elems: int
    orig_shape: tuple[int, ...]
    topk: int

    @property
    def n_values(self) -> int:
        return self.n_chunks * self.topk


class TopKCompressor:
    """Per-chunk top-k by |value|, canonical ascending-index order, quantized values.

    Wire format v1 packs values as 2-bit codes, so the quantizer must use <= 4 bins.
    Batch merging of many miners' payloads is owned by outer_opt, not here.
    """

    def __init__(self, transformer: ChunkingTransformer, quantizer: Quantizer, topk: int):
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if quantizer.bins > 4:
            raise ValueError(f"wire format v1 packs 2-bit codes; quantizer.bins must be <= 4, got {quantizer.bins}")
        self.transformer = transformer
        self.quantizer = quantizer
        self.topk = int(topk)

    def effective_topk(self, name: str) -> int:
        return min(self.topk, self.transformer.geometry(name).chunk_elems)

    @torch.no_grad()
    def compress(self, name: str, tensor: torch.Tensor) -> CompressedTensor:
        g = self.transformer.geometry(name)
        chunked = self.transformer.encode(name, tensor).to(torch.float32)
        k = min(self.topk, g.chunk_elems)
        idx = torch.topk(chunked.abs(), k=k, dim=-1, largest=True, sorted=False).indices
        idx, _ = idx.sort(dim=-1)  # canonical: strictly increasing per chunk
        vals = torch.gather(chunked, -1, idx)
        codes, qparams = self.quantizer.quantize(vals.reshape(-1))
        return CompressedTensor(
            idxs_packed=pack_12bit_indices(idx.reshape(-1)),
            codes_packed=pack_2bit_values(codes),
            qparams=qparams,
            n_chunks=g.n_chunks,
            chunk_elems=g.chunk_elems,
            orig_shape=g.orig_shape,
            topk=k,
        )

    @torch.no_grad()
    def decompress(self, name: str, ct: CompressedTensor) -> torch.Tensor:
        g = self.transformer.geometry(name)
        if (ct.n_chunks, ct.chunk_elems) != (g.n_chunks, g.chunk_elems):
            raise ValueError(
                f"{name}: compressed geometry ({ct.n_chunks}, {ct.chunk_elems}) "
                f"!= registered ({g.n_chunks}, {g.chunk_elems})"
            )
        if ct.orig_shape != g.orig_shape:
            raise ValueError(f"{name}: compressed shape {ct.orig_shape} != registered {g.orig_shape}")
        n = ct.n_values
        codes = unpack_2bit_values(ct.codes_packed, n)
        vals = self.quantizer.dequantize(codes, ct.qparams)
        idx = unpack_12bit_indices(ct.idxs_packed, n).reshape(ct.n_chunks, ct.topk)
        if n and int(idx.max().item()) >= ct.chunk_elems:
            raise ValueError(f"{name}: index {int(idx.max().item())} out of range for chunk {ct.chunk_elems}")
        dense = torch.zeros(ct.n_chunks, ct.chunk_elems, dtype=torch.float32)
        dense.scatter_(-1, idx, vals.reshape(ct.n_chunks, ct.topk))
        return self.transformer.decode(name, dense)


# --------------------------------------------------------------------------- #
# Error feedback
# --------------------------------------------------------------------------- #


class ErrorFeedback:
    """Per-parameter fp32 CPU momentum/residual buffers: m = beta*m + delta.

    The transmitted (decompressed) part is subtracted after compression so the
    residual re-enters the next window. `merkle_root` commits the buffer contents
    for the A2 spot-check audit (v1.1): fixed 1 MiB chunk leaves, name-sorted.
    """

    def __init__(self, beta: float = 0.95):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = float(beta)
        self._buffers: dict[str, torch.Tensor] = {}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._buffers))

    def buffer(self, name: str) -> torch.Tensor:
        return self._buffers[name]

    @torch.no_grad()
    def update(self, name: str, delta: torch.Tensor) -> torch.Tensor:
        """Fold `delta` into the buffer (m = beta*m + delta); returns a clone to compress."""
        d = delta.detach().to(device="cpu", dtype=torch.float32)
        buf = self._buffers.get(name)
        if buf is None:
            buf = torch.zeros_like(d)
            self._buffers[name] = buf
        elif buf.shape != d.shape:
            raise ValueError(f"{name}: delta shape {tuple(d.shape)} != buffer {tuple(buf.shape)}")
        buf.mul_(self.beta).add_(d)
        return buf.clone()

    @torch.no_grad()
    def subtract_transmitted(self, name: str, decompressed: torch.Tensor) -> None:
        """Remove the transmitted reconstruction, keeping only the residual."""
        buf = self._buffers.get(name)
        if buf is None:
            raise KeyError(f"no error-feedback buffer for {name!r}")
        d = decompressed.detach().to(device="cpu", dtype=torch.float32)
        if buf.shape != d.shape:
            raise ValueError(f"{name}: transmitted shape {tuple(d.shape)} != buffer {tuple(buf.shape)}")
        buf.sub_(d)

    @torch.no_grad()
    def reset(self) -> None:
        """Zero all buffers (null/warmup rounds must not accumulate stale residual)."""
        for buf in self._buffers.values():
            buf.zero_()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: buf.clone() for name, buf in self._buffers.items()}

    @torch.no_grad()
    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        loaded: dict[str, torch.Tensor] = {}
        for name, tensor in state.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name}: expected a tensor, got {type(tensor).__name__}")
            loaded[name] = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
        self._buffers = loaded

    def merkle_root(self) -> str:
        """Hex commitment to all buffers for the A2 audit.

        Leaves are blake2b-256 over (name-len, name, chunk-index, chunk-bytes) per
        1 MiB chunk, in name-sorted order. Rooted via mok_core.data.merkle when
        available, else blake2b over the concatenated leaf digests.
        """
        leaves: list[bytes] = []
        for name in sorted(self._buffers):
            raw = tensor_bytes(self._buffers[name])
            n_chunks = max(1, -(-len(raw) // _MERKLE_CHUNK_BYTES))
            for ci in range(n_chunks):
                h = hashlib.blake2b(digest_size=32)
                h.update(len(name).to_bytes(4, "little"))
                h.update(name.encode("utf-8"))
                h.update(ci.to_bytes(8, "little"))
                h.update(raw[ci * _MERKLE_CHUNK_BYTES : (ci + 1) * _MERKLE_CHUNK_BYTES])
                leaves.append(h.digest())
        return _merkle_root_from_leaves(leaves)


def _merkle_root_from_leaves(leaves: Iterable[bytes]) -> str:
    leaf_list = list(leaves)
    if leaf_list:
        try:
            from mok_core.data.merkle import MerkleTree  # noqa: PLC0415 — heavier import kept lazy

            return MerkleTree(leaf_list).root.hex()
        except ImportError:
            pass
    # No buffers, or merkle module unavailable: blake2b over the concatenated leaf digests.
    h = hashlib.blake2b(digest_size=32)
    for leaf in leaf_list:
        h.update(leaf)
    return h.hexdigest()
