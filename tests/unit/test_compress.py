"""Unit tests for subnet/core/compress.py — SparseLoCo compression primitives."""

from __future__ import annotations

import math

import pytest
import torch

from subnet.core.compress import (
    SCALE_FLOOR,
    ChunkingTransformer,
    ErrorFeedback,
    Quantizer,
    TopKCompressor,
    chunk_geometry,
    pack_2bit_values,
    pack_12bit_indices,
    packed_nbytes_2bit,
    packed_nbytes_12bit,
    unpack_2bit_values,
    unpack_12bit_indices,
)

# --------------------------------------------------------------------------- #
# 12-bit index packing
# --------------------------------------------------------------------------- #


class Test12BitPacking:
    def test_golden_even(self):
        idx = torch.tensor([0, 1, 4095, 2048], dtype=torch.int64)
        packed = pack_12bit_indices(idx)
        # consensus constant — change requires SPEC_VERSION bump
        assert bytes(packed.tolist()).hex() == "001000ff0f80"
        assert torch.equal(unpack_12bit_indices(packed, 4), idx)

    def test_golden_odd(self):
        idx = torch.tensor([0, 1, 4095, 2048, 7], dtype=torch.int64)
        packed = pack_12bit_indices(idx)
        # consensus constant — change requires SPEC_VERSION bump
        assert bytes(packed.tolist()).hex() == "001000ff0f800700"
        assert torch.equal(unpack_12bit_indices(packed, 5), idx)

    @pytest.mark.parametrize("count", [1, 2, 3, 64, 127, 257])
    def test_round_trip(self, count):
        g = torch.Generator().manual_seed(count)
        idx = torch.randint(0, 4096, (count,), generator=g)
        packed = pack_12bit_indices(idx)
        assert packed.dtype == torch.uint8
        assert packed.numel() == packed_nbytes_12bit(count)
        assert torch.equal(unpack_12bit_indices(packed, count), idx)

    def test_empty(self):
        packed = pack_12bit_indices(torch.zeros(0, dtype=torch.int64))
        assert packed.numel() == 0
        assert unpack_12bit_indices(packed, 0).numel() == 0

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 4096\)"):
            pack_12bit_indices(torch.tensor([4096]))
        with pytest.raises(ValueError, match=r"\[0, 4096\)"):
            pack_12bit_indices(torch.tensor([-1]))

    def test_float_rejected(self):
        with pytest.raises(ValueError, match="integer"):
            pack_12bit_indices(torch.tensor([1.0]))

    def test_unpack_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="packed length"):
            unpack_12bit_indices(torch.zeros(3, dtype=torch.uint8), 4)
        with pytest.raises(ValueError, match="uint8"):
            unpack_12bit_indices(torch.zeros(3, dtype=torch.int64), 2)


# --------------------------------------------------------------------------- #
# 2-bit value packing
# --------------------------------------------------------------------------- #


class Test2BitPacking:
    def test_golden(self):
        codes = torch.tensor([3, 0, 1, 2, 3], dtype=torch.uint8)
        packed = pack_2bit_values(codes)
        # consensus constant — change requires SPEC_VERSION bump
        assert bytes(packed.tolist()).hex() == "9303"
        assert torch.equal(unpack_2bit_values(packed, 5), codes)

    @pytest.mark.parametrize("count", list(range(1, 10)) + [64, 255])
    def test_round_trip(self, count):
        g = torch.Generator().manual_seed(count)
        codes = torch.randint(0, 4, (count,), generator=g, dtype=torch.uint8)
        packed = pack_2bit_values(codes)
        assert packed.dtype == torch.uint8
        assert packed.numel() == packed_nbytes_2bit(count)
        assert torch.equal(unpack_2bit_values(packed, count), codes)

    def test_wire_size_halved_vs_uint8(self):
        assert packed_nbytes_2bit(4096) == 1024  # 4x smaller than raw uint8 codes

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError, match=r"\[0, 4\)"):
            pack_2bit_values(torch.tensor([4], dtype=torch.uint8))

    def test_dtype_rejected(self):
        with pytest.raises(ValueError, match="uint8"):
            pack_2bit_values(torch.tensor([1], dtype=torch.int64))

    def test_unpack_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="packed length"):
            unpack_2bit_values(torch.zeros(2, dtype=torch.uint8), 9)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


class TestChunkingTransformer:
    def test_geometry_1d(self):
        g = chunk_geometry((5000,), 64)
        assert (g.mode, g.n_chunks, g.chunk_elems) == ("flat", 2, 4096)

    def test_geometry_2d_padding(self):
        g = chunk_geometry((65, 130), 64)
        assert (g.mode, g.pad_rows, g.pad_cols, g.n_chunks) == ("grid", 128, 192, 6)

    def test_geometry_3d_collapses_leading_dims(self):
        g = chunk_geometry((3, 4, 5), 64)
        assert (g.mode, g.rows, g.cols, g.n_chunks) == ("grid", 12, 5, 1)

    def test_geometry_rejects_zero_dim(self):
        with pytest.raises(ValueError, match="zero-sized"):
            chunk_geometry((0, 4), 64)

    @pytest.mark.parametrize("shape", [(5000,), (64, 64), (65, 130), (3, 4, 5), (1,), ()])
    def test_encode_decode_identity(self, shape):
        tf = ChunkingTransformer({"p": shape}, target_chunk=64)
        g = torch.Generator().manual_seed(0)
        t = torch.randn(shape, generator=g)
        chunked = tf.encode("p", t)
        geom = tf.geometry("p")
        assert chunked.shape == (geom.n_chunks, geom.chunk_elems)
        assert torch.equal(tf.decode("p", chunked), t)

    def test_padding_is_zero(self):
        tf = ChunkingTransformer({"b": (10,)}, target_chunk=4)
        chunked = tf.encode("b", torch.ones(10))
        flat = chunked.reshape(-1)
        assert torch.equal(flat[:10], torch.ones(10))
        assert torch.equal(flat[10:], torch.zeros(6))

    def test_grid_block_layout(self):
        # 4x4 with target_chunk 2 -> 4 chunks of 2x2 blocks, row-major over the block grid
        tf = ChunkingTransformer({"w": (4, 4)}, target_chunk=2)
        t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        chunked = tf.encode("w", t)
        assert torch.equal(chunked[0], torch.tensor([0.0, 1.0, 4.0, 5.0]))
        assert torch.equal(chunked[1], torch.tensor([2.0, 3.0, 6.0, 7.0]))
        assert torch.equal(chunked[3], torch.tensor([10.0, 11.0, 14.0, 15.0]))

    def test_use_dct_guard(self):
        with pytest.raises(ValueError, match="use_dct"):
            ChunkingTransformer({"p": (8,)}, target_chunk=4, use_dct=True)
        tf = ChunkingTransformer({"p": (8,)}, target_chunk=4, use_dct=False)
        assert tf.use_dct is False

    def test_shape_mismatch_rejected(self):
        tf = ChunkingTransformer({"p": (8,)}, target_chunk=4)
        with pytest.raises(ValueError, match="registered"):
            tf.encode("p", torch.zeros(9))
        with pytest.raises(KeyError):
            tf.encode("q", torch.zeros(8))
        with pytest.raises(ValueError, match="chunked shape"):
            tf.decode("p", torch.zeros(2, 16))


# --------------------------------------------------------------------------- #
# Quantizer
# --------------------------------------------------------------------------- #


class TestQuantizer:
    def test_round_trip_statistics(self):
        q = Quantizer(bins=4, range_sigmas=6.0)
        g = torch.Generator().manual_seed(7)
        vals = torch.randn(4096, generator=g) * 0.3 + 1.5
        codes, qparams = q.quantize(vals)
        assert codes.dtype == torch.uint8
        assert int(codes.max()) < 4
        deq = q.dequantize(codes, qparams)
        # per-bin-mean lookup reconstructs the global mean exactly (up to fp32 rounding)
        assert torch.allclose(deq.mean(), vals.mean(), atol=1e-4)
        sigma = vals.std().item()
        assert (deq - vals).abs().mean().item() < 1.5 * sigma
        corr = torch.corrcoef(torch.stack([deq, vals]))[0, 1].item()
        assert corr > 0.8

    def test_degenerate_all_equal_exact(self):
        q = Quantizer(bins=4)
        vals = torch.full((64,), 2.5)  # dyadic: fp32 mean is exact, centered residual is zero
        codes, qparams = q.quantize(vals)
        assert qparams["scale"] == SCALE_FLOOR
        assert torch.allclose(q.dequantize(codes, qparams), vals, atol=1e-6)

    def test_degenerate_all_equal_nondyadic(self):
        q = Quantizer(bins=4)
        vals = torch.full((64,), 3.14)  # fp32 mean rounds; still near-exact reconstruction
        codes, qparams = q.quantize(vals)
        assert torch.allclose(q.dequantize(codes, qparams), vals, atol=1e-5)

    def test_single_element_exact(self):
        q = Quantizer(bins=4)
        codes, qparams = q.quantize(torch.tensor([42.5]))
        assert q.dequantize(codes, qparams).item() == pytest.approx(42.5, abs=1e-6)

    def test_all_zero(self):
        q = Quantizer(bins=4)
        codes, qparams = q.quantize(torch.zeros(16))
        assert torch.equal(q.dequantize(codes, qparams), torch.zeros(16))

    def test_lookup_shape_and_dtype(self):
        q = Quantizer(bins=4)
        _, qparams = q.quantize(torch.randn(100, generator=torch.Generator().manual_seed(1)))
        lookup = qparams["lookup"]
        assert isinstance(lookup, torch.Tensor)
        assert lookup.dtype == torch.float32 and lookup.shape == (4,)
        assert isinstance(qparams["shift"], float) and isinstance(qparams["scale"], float)

    def test_rejects_bad_input(self):
        q = Quantizer(bins=4)
        with pytest.raises(ValueError, match="empty"):
            q.quantize(torch.zeros(0))
        with pytest.raises(ValueError, match="non-finite"):
            q.quantize(torch.tensor([1.0, float("nan")]))
        with pytest.raises(ValueError, match="uint8"):
            q.dequantize(torch.zeros(4, dtype=torch.int64), {"shift": 0.0, "scale": 1.0, "lookup": torch.zeros(4)})

    def test_code_out_of_lookup_rejected(self):
        q = Quantizer(bins=4)
        with pytest.raises(ValueError, match="out of lookup"):
            q.dequantize(
                torch.tensor([3], dtype=torch.uint8),
                {"shift": 0.0, "scale": 1.0, "lookup": torch.zeros(2)},
            )

    def test_deterministic(self):
        q = Quantizer(bins=4)
        vals = torch.linspace(-2.0, 2.0, 1000)
        c1, p1 = q.quantize(vals)
        c2, p2 = q.quantize(vals)
        assert torch.equal(c1, c2)
        assert p1["shift"] == p2["shift"] and p1["scale"] == p2["scale"]
        assert torch.equal(p1["lookup"], p2["lookup"])


# --------------------------------------------------------------------------- #
# TopKCompressor
# --------------------------------------------------------------------------- #


def _make_compressor(shapes: dict[str, tuple[int, ...]], target_chunk: int = 4, topk: int = 4):
    tf = ChunkingTransformer(shapes, target_chunk=target_chunk)
    return TopKCompressor(tf, Quantizer(bins=4, range_sigmas=6.0), topk=topk)


class TestTopKCompressor:
    def test_exact_indices_and_sparsity(self):
        comp = _make_compressor({"w": (8, 12)}, target_chunk=4, topk=4)
        g = torch.Generator().manual_seed(3)
        t = torch.randn(8, 12, generator=g)
        ct = comp.compress("w", t)
        geom = comp.transformer.geometry("w")
        assert (ct.n_chunks, ct.chunk_elems, ct.topk) == (geom.n_chunks, geom.chunk_elems, 4)

        # recompute the expected canonical indices independently
        chunked = comp.transformer.encode("w", t)
        expected_idx, _ = torch.topk(chunked.abs(), k=4, dim=-1, sorted=False).indices.sort(dim=-1)
        got_idx = unpack_12bit_indices(ct.idxs_packed, ct.n_values).reshape(ct.n_chunks, ct.topk)
        assert torch.equal(got_idx, expected_idx)

        dense = comp.decompress("w", ct)
        assert dense.shape == t.shape
        # non-selected positions are exactly zero
        dense_chunked = comp.transformer.encode("w", dense)
        mask = torch.zeros_like(chunked, dtype=torch.bool)
        mask.scatter_(-1, expected_idx, True)
        assert torch.equal(dense_chunked[~mask], torch.zeros((~mask).sum().item()))

    def test_selected_values_within_quant_tolerance(self):
        comp = _make_compressor({"w": (16, 16)}, target_chunk=4, topk=6)
        g = torch.Generator().manual_seed(11)
        t = torch.randn(16, 16, generator=g)
        ct = comp.compress("w", t)
        dense = comp.decompress("w", ct)
        sel = dense != 0
        selected_orig = t[sel]
        selected_deq = dense[sel]
        # 4 bins over 6 sigmas of the selected values: within-bin error bounded by the range
        sigma = selected_orig.std().item()
        assert (selected_deq - selected_orig).abs().max().item() < 6.0 * sigma
        assert (selected_deq - selected_orig).abs().mean().item() < 1.5 * sigma
        corr = torch.corrcoef(torch.stack([selected_deq, selected_orig]))[0, 1].item()
        assert corr > 0.8

    def test_topk_clamped_to_chunk(self):
        comp = _make_compressor({"b": (10,)}, target_chunk=2, topk=64)  # chunk_elems = 4
        ct = comp.compress("b", torch.arange(10, dtype=torch.float32))
        assert ct.topk == 4  # fully dense per chunk

    def test_deterministic_wire_bytes(self):
        comp = _make_compressor({"w": (8, 8)}, target_chunk=4, topk=4)
        t = torch.linspace(-1.0, 1.0, 64).reshape(8, 8)
        c1 = comp.compress("w", t)
        c2 = comp.compress("w", t)
        assert torch.equal(c1.idxs_packed, c2.idxs_packed)
        assert torch.equal(c1.codes_packed, c2.codes_packed)
        assert c1.qparams["shift"] == c2.qparams["shift"]

    def test_rejects_bins_over_4(self):
        tf = ChunkingTransformer({"w": (8, 8)}, target_chunk=4)
        with pytest.raises(ValueError, match="2-bit"):
            TopKCompressor(tf, Quantizer(bins=8), topk=4)

    def test_decompress_geometry_mismatch_rejected(self):
        comp = _make_compressor({"w": (8, 8), "v": (16, 16)}, target_chunk=4, topk=4)
        ct = comp.compress("w", torch.randn(8, 8, generator=torch.Generator().manual_seed(0)))
        with pytest.raises(ValueError, match="geometry"):
            comp.decompress("v", ct)


# --------------------------------------------------------------------------- #
# ErrorFeedback
# --------------------------------------------------------------------------- #


class TestErrorFeedback:
    def test_closed_form_two_updates(self):
        beta = 0.9
        ef = ErrorFeedback(beta=beta)
        g = torch.Generator().manual_seed(5)
        d1 = torch.randn(4, generator=g)
        d2 = torch.randn(4, generator=g)
        t1 = torch.randn(4, generator=g)

        r1 = ef.update("p", d1)
        assert torch.equal(r1, d1)  # beta * 0 + d1
        ef.subtract_transmitted("p", t1)
        r2 = ef.update("p", d2)
        expected = (d1 - t1).mul(beta).add(d2)
        assert torch.equal(r2, expected)

    def test_returns_clone_not_alias(self):
        ef = ErrorFeedback(beta=0.5)
        r = ef.update("p", torch.ones(3))
        r.fill_(99.0)
        assert torch.equal(ef.buffer("p"), torch.ones(3))

    def test_dtype_and_device_coercion(self):
        ef = ErrorFeedback()
        r = ef.update("p", torch.ones(3, dtype=torch.bfloat16))
        assert r.dtype == torch.float32 and r.device.type == "cpu"

    def test_state_dict_round_trip(self):
        ef = ErrorFeedback(beta=0.95)
        g = torch.Generator().manual_seed(9)
        ef.update("a", torch.randn(6, generator=g))
        ef.update("b", torch.randn(2, 3, generator=g))
        sd = ef.state_dict()

        ef2 = ErrorFeedback(beta=0.95)
        ef2.load_state_dict(sd)
        assert ef2.names == ef.names
        for name in ef.names:
            assert torch.equal(ef2.buffer(name), ef.buffer(name))
        assert ef2.merkle_root() == ef.merkle_root()

        # state_dict is a deep copy
        sd["a"].fill_(0.0)
        assert not torch.equal(ef.buffer("a"), sd["a"])

    def test_reset_zeroes(self):
        ef = ErrorFeedback()
        ef.update("p", torch.ones(4))
        ef.reset()
        assert torch.equal(ef.buffer("p"), torch.zeros(4))

    def test_merkle_root_properties(self):
        ef = ErrorFeedback()
        empty_root = ef.merkle_root()
        assert len(empty_root) == 64 and all(c in "0123456789abcdef" for c in empty_root)
        ef.update("p", torch.ones(4))
        r1 = ef.merkle_root()
        assert r1 != empty_root
        assert ef.merkle_root() == r1  # stable across calls
        ef.update("p", torch.ones(4))
        assert ef.merkle_root() != r1  # sensitive to buffer contents

    def test_shape_mismatch_rejected(self):
        ef = ErrorFeedback()
        ef.update("p", torch.ones(4))
        with pytest.raises(ValueError, match="shape"):
            ef.update("p", torch.ones(5))
        with pytest.raises(ValueError, match="shape"):
            ef.subtract_transmitted("p", torch.ones(5))
        with pytest.raises(KeyError):
            ef.subtract_transmitted("q", torch.ones(4))

    def test_beta_validation(self):
        with pytest.raises(ValueError, match="beta"):
            ErrorFeedback(beta=1.0)
        with pytest.raises(ValueError, match="beta"):
            ErrorFeedback(beta=-0.1)


# --------------------------------------------------------------------------- #
# Full pipeline sanity: EF + compress reduces residual over repeated windows
# --------------------------------------------------------------------------- #


def test_error_feedback_pipeline_converges():
    """Repeatedly compressing a constant signal with EF: transmitted average approaches
    the signal and the residual buffer stays bounded (no divergence)."""
    comp = _make_compressor({"w": (8, 8)}, target_chunk=4, topk=8)
    ef = ErrorFeedback(beta=0.95)
    g = torch.Generator().manual_seed(21)
    signal = torch.randn(8, 8, generator=g)
    recent: list[torch.Tensor] = []
    for _ in range(60):
        buffered = ef.update("w", signal)
        ct = comp.compress("w", buffered)
        decompressed = comp.decompress("w", ct)
        ef.subtract_transmitted("w", decompressed)
        recent.append(decompressed)
    steady_avg = torch.stack(recent[-20:]).mean(0)
    rel_err = (steady_avg - signal).abs().mean().item() / signal.abs().mean().item()
    assert rel_err < 0.15
    resid_norm = ef.buffer("w").norm().item()
    assert math.isfinite(resid_norm)
    assert resid_norm < 5.0 * signal.norm().item()
