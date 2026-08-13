"""Unit tests for C/core/payload.py — wire format, hashing, validation, ownership."""

from __future__ import annotations

import json

import pytest
import torch
import zstandard

from C.core.compress import (
    ChunkingTransformer,
    Quantizer,
    TopKCompressor,
    pack_2bit_values,
    pack_12bit_indices,
)
from C.core.payload import (
    MAGIC,
    WIRE_VERSION,
    PayloadError,
    PayloadMeta,
    WindowPayload,
    assign_owned_params,
    canonical_payload_hash,
    deserialize,
    serialize,
    validate_structure,
)
from mok_core.config.canonical import canonical_bytes

TARGET_CHUNK = 4
TOPK = 3
PARAM_SHAPES: dict[str, tuple[int, ...]] = {"a.weight": (6, 10), "b.bias": (7,)}
DENSE_SHAPES: dict[str, tuple[int, ...]] = {"router.balance_bias": (5,)}
MAX_BYTES = 1 << 20

_HEX_A = "aa" * 32
_HEX_B = "bb" * 32
_HEX_C = "cc" * 32


def _make_meta() -> PayloadMeta:
    return PayloadMeta(
        sample_digest=_HEX_A,
        sample_count=131072,
        theta_end_hash=_HEX_B,
        state_root=_HEX_C,
        global_step=12500,
        spec_version=1,
    )


def _make_compressor() -> TopKCompressor:
    tf = ChunkingTransformer(PARAM_SHAPES, target_chunk=TARGET_CHUNK)
    return TopKCompressor(tf, Quantizer(bins=4, range_sigmas=6.0), topk=TOPK)


def _make_payload() -> WindowPayload:
    """Fully deterministic payload — tensor contents derived from arange, no RNG."""
    comp = _make_compressor()
    compressed = {}
    for i, (name, shape) in enumerate(sorted(PARAM_SHAPES.items())):
        n = 1
        for s in shape:
            n *= s
        t = (torch.arange(n, dtype=torch.float32).reshape(shape) - n / 2) * (0.01 * (i + 1))
        compressed[name] = comp.compress(name, t)
    dense = {
        name: torch.arange(shape[0], dtype=torch.float32) * 0.125
        for name, shape in DENSE_SHAPES.items()
    }
    return WindowPayload(uid=42, window=1337, compressed=compressed, dense=dense, metadata=_make_meta())


def _frame(body: bytes) -> bytes:
    """Wrap a raw body in the outer MOKP envelope (for crafting malicious frames)."""
    cctx = zstandard.ZstdCompressor(level=3, write_checksum=False, threads=0)
    return MAGIC + bytes([WIRE_VERSION]) + cctx.compress(body)


def _body_of(data: bytes) -> bytes:
    return zstandard.ZstdDecompressor().decompress(data[5:], max_output_size=4 * MAX_BYTES)


def _reframe_header(data: bytes, mutate) -> bytes:
    """Decode a valid frame, apply `mutate(header_dict)`, rebuild canonically."""
    body = _body_of(data)
    header_len = int.from_bytes(body[:4], "little")
    header = json.loads(body[4 : 4 + header_len].decode())
    blobs = body[4 + header_len :]
    mutate(header)
    new_header = canonical_bytes(header)
    return _frame(len(new_header).to_bytes(4, "little") + new_header + blobs)


# --------------------------------------------------------------------------- #
# Serialization determinism + golden vector
# --------------------------------------------------------------------------- #


class TestSerializeDeterminism:
    def test_identical_bytes_twice(self):
        p1, p2 = _make_payload(), _make_payload()
        assert serialize(p1) == serialize(p2)
        assert canonical_payload_hash(p1) == canonical_payload_hash(p2)

    def test_round_trip_reserializes_identically(self):
        p = _make_payload()
        data = serialize(p)
        assert serialize(deserialize(data, max_bytes=MAX_BYTES)) == data

    def test_golden_payload_hash(self):
        # consensus constant — change requires SPEC_VERSION bump
        # (binds header canonicalization, blob order, and the pinned zstd level-3 frame)
        expected = "aee9446706d2c5804c780ee0da1e3463d663b5d903c1949fa28c8ac6f9360414"
        assert canonical_payload_hash(_make_payload()) == expected

    def test_hash_sensitive_to_contents(self):
        p = _make_payload()
        base = canonical_payload_hash(p)
        p.window += 1
        assert canonical_payload_hash(p) != base

    def test_magic_and_version_prefix(self):
        data = serialize(_make_payload())
        assert data[:4] == MAGIC
        assert data[4] == WIRE_VERSION


# --------------------------------------------------------------------------- #
# Round-trip integrity
# --------------------------------------------------------------------------- #


class TestRoundTrip:
    def test_full_field_equality(self):
        p = _make_payload()
        q = deserialize(serialize(p), max_bytes=MAX_BYTES)
        assert (q.uid, q.window) == (p.uid, p.window)
        assert q.metadata == p.metadata
        assert set(q.compressed) == set(p.compressed)
        for name, ct in p.compressed.items():
            rt = q.compressed[name]
            assert torch.equal(rt.idxs_packed, ct.idxs_packed)
            assert torch.equal(rt.codes_packed, ct.codes_packed)
            assert rt.qparams["shift"] == ct.qparams["shift"]
            assert rt.qparams["scale"] == ct.qparams["scale"]
            assert torch.equal(rt.qparams["lookup"], ct.qparams["lookup"])
            assert (rt.n_chunks, rt.chunk_elems, rt.topk) == (ct.n_chunks, ct.chunk_elems, ct.topk)
            assert rt.orig_shape == tuple(ct.orig_shape)
        assert set(q.dense) == set(p.dense)
        for name, t in p.dense.items():
            assert torch.equal(q.dense[name], t)

    def test_round_trip_decompresses_identically(self):
        p = _make_payload()
        q = deserialize(serialize(p), max_bytes=MAX_BYTES)
        comp = _make_compressor()
        for name in PARAM_SHAPES:
            assert torch.equal(comp.decompress(name, p.compressed[name]), comp.decompress(name, q.compressed[name]))

    def test_empty_payload(self):
        p = WindowPayload(uid=0, window=0, compressed={}, dense={}, metadata=_make_meta())
        q = deserialize(serialize(p), max_bytes=MAX_BYTES)
        assert q.compressed == {} and q.dense == {}

    def test_validate_after_round_trip(self):
        q = deserialize(serialize(_make_payload()), max_bytes=MAX_BYTES)
        validate_structure(q, PARAM_SHAPES, DENSE_SHAPES, TOPK, target_chunk=TARGET_CHUNK)


# --------------------------------------------------------------------------- #
# Deserialize bounds checks
# --------------------------------------------------------------------------- #


class TestDeserializeRejection:
    def test_oversize_rejected(self):
        data = serialize(_make_payload())
        with pytest.raises(PayloadError, match="max_bytes"):
            deserialize(data, max_bytes=len(data) - 1)

    def test_wrong_magic(self):
        data = serialize(_make_payload())
        with pytest.raises(PayloadError, match="magic"):
            deserialize(b"XXXX" + data[4:], max_bytes=MAX_BYTES)

    def test_wrong_version(self):
        data = serialize(_make_payload())
        with pytest.raises(PayloadError, match="version"):
            deserialize(data[:4] + bytes([99]) + data[5:], max_bytes=MAX_BYTES)

    def test_too_short(self):
        with pytest.raises(PayloadError, match="too short"):
            deserialize(b"MOK", max_bytes=MAX_BYTES)

    def test_corrupt_zstd(self):
        data = serialize(_make_payload())
        with pytest.raises(PayloadError, match="zstd|content size"):
            deserialize(data[:5] + b"\x00" * 20, max_bytes=MAX_BYTES)

    def test_decompressed_budget_enforced_before_decompression(self):
        data = serialize(_make_payload())
        with pytest.raises(PayloadError, match="content size"):
            deserialize(data, max_bytes=MAX_BYTES, max_decompressed_bytes=10)

    def test_header_len_out_of_range(self):
        with pytest.raises(PayloadError, match="header length"):
            deserialize(_frame((1 << 30).to_bytes(4, "little") + b"{}"), max_bytes=MAX_BYTES)

    def test_header_not_json(self):
        raw = b"not json at all!"
        body = len(raw).to_bytes(4, "little") + raw
        with pytest.raises(PayloadError, match="JSON"):
            deserialize(_frame(body), max_bytes=MAX_BYTES)

    def test_non_canonical_header_rejected(self):
        # semantically valid JSON but with whitespace — must be rejected byte-for-byte
        data = serialize(_make_payload())
        body = _body_of(data)
        header_len = int.from_bytes(body[:4], "little")
        loose = json.dumps(json.loads(body[4 : 4 + header_len].decode()), indent=1).encode()
        tampered = _frame(len(loose).to_bytes(4, "little") + loose + body[4 + header_len :])
        with pytest.raises(PayloadError, match="canonical"):
            deserialize(tampered, max_bytes=MAX_BYTES)

    def test_missing_header_field(self):
        def strip_uid(h):
            del h["uid"]

        with pytest.raises(PayloadError, match="fields"):
            deserialize(_reframe_header(serialize(_make_payload()), strip_uid), max_bytes=MAX_BYTES)

    def test_extra_header_field(self):
        def add(h):
            h["evil"] = 1

        with pytest.raises(PayloadError, match="fields"):
            deserialize(_reframe_header(serialize(_make_payload()), add), max_bytes=MAX_BYTES)

    def test_bad_blob_count_declaration(self):
        def bump(h):
            h["compressed"][0]["idxs_bytes"] += 3

        with pytest.raises(PayloadError, match="blob sizes|body length"):
            deserialize(_reframe_header(serialize(_make_payload()), bump), max_bytes=MAX_BYTES)

    def test_huge_n_chunks_rejected_before_allocation(self):
        def inflate(h):
            e = h["compressed"][0]
            e["n_chunks"] = 1 << 23
            n = e["n_chunks"] * e["topk"]
            e["idxs_bytes"] = (n // 2) * 3 + (2 if n % 2 else 0)
            e["codes_bytes"] = (n + 3) // 4

        with pytest.raises(PayloadError, match="capacity|body length"):
            deserialize(_reframe_header(serialize(_make_payload()), inflate), max_bytes=MAX_BYTES)

    def test_trailing_bytes_rejected(self):
        data = serialize(_make_payload())
        body = _body_of(data) + b"\x00\x01\x02"
        with pytest.raises(PayloadError, match="body length"):
            deserialize(_frame(body), max_bytes=MAX_BYTES)

    def test_unsorted_names_rejected(self):
        def swap(h):
            h["compressed"].reverse()

        with pytest.raises(PayloadError, match="sorted"):
            deserialize(_reframe_header(serialize(_make_payload()), swap), max_bytes=MAX_BYTES)

    def test_bool_as_int_rejected(self):
        def falsify(h):
            h["uid"] = True

        with pytest.raises(PayloadError, match="integer"):
            deserialize(_reframe_header(serialize(_make_payload()), falsify), max_bytes=MAX_BYTES)

    def test_negative_spec_version_rejected(self):
        def zero(h):
            h["meta"]["spec_version"] = 0

        with pytest.raises(PayloadError, match="spec_version"):
            deserialize(_reframe_header(serialize(_make_payload()), zero), max_bytes=MAX_BYTES)

    def test_dense_numel_mismatch_rejected(self):
        def corrupt(h):
            h["dense"][0]["numel_bytes"] += 4

        with pytest.raises(PayloadError, match="numel_bytes|body length"):
            deserialize(_reframe_header(serialize(_make_payload()), corrupt), max_bytes=MAX_BYTES)


# --------------------------------------------------------------------------- #
# validate_structure rejection matrix
# --------------------------------------------------------------------------- #


def _validate(p: WindowPayload, **kw) -> None:
    validate_structure(p, PARAM_SHAPES, DENSE_SHAPES, TOPK, target_chunk=TARGET_CHUNK, **kw)


class TestValidateStructure:
    def test_valid_passes(self):
        assert _validate(_make_payload()) is None

    def test_missing_param(self):
        p = _make_payload()
        del p.compressed["a.weight"]
        with pytest.raises(PayloadError, match="missing"):
            _validate(p)

    def test_extra_param(self):
        p = _make_payload()
        p.compressed["z.extra"] = p.compressed["a.weight"]
        with pytest.raises(PayloadError, match="extra"):
            _validate(p)

    def test_dense_name_mismatch(self):
        p = _make_payload()
        p.dense["evil"] = torch.zeros(3)
        with pytest.raises(PayloadError, match="dense name set"):
            _validate(p)

    def test_wrong_orig_shape(self):
        p = _make_payload()
        p.compressed["a.weight"].orig_shape = (6, 11)
        with pytest.raises(PayloadError, match="orig_shape"):
            _validate(p)

    def test_wrong_geometry(self):
        p = _make_payload()
        p.compressed["a.weight"].chunk_elems = 8
        with pytest.raises(PayloadError, match="geometry"):
            _validate(p)

    def test_wrong_topk(self):
        p = _make_payload()
        with pytest.raises(PayloadError, match="topk"):
            validate_structure(p, PARAM_SHAPES, DENSE_SHAPES, TOPK + 1, target_chunk=TARGET_CHUNK)

    def test_out_of_range_index(self):
        p = _make_payload()
        ct = p.compressed["a.weight"]
        n = ct.n_chunks * ct.topk
        bad = torch.zeros(n, dtype=torch.int64)
        bad[0 : ct.topk] = torch.tensor([1, 2, 100])  # 100 >= chunk_elems (16)
        rest = torch.arange(ct.topk)
        for c in range(1, ct.n_chunks):
            bad[c * ct.topk : (c + 1) * ct.topk] = rest
        ct.idxs_packed = pack_12bit_indices(bad)
        with pytest.raises(PayloadError, match="out of bounds"):
            _validate(p)

    def test_duplicate_indices_rejected(self):
        p = _make_payload()
        ct = p.compressed["a.weight"]
        n = ct.n_chunks * ct.topk
        dup = torch.tensor([5, 5, 6], dtype=torch.int64).repeat(ct.n_chunks)[:n]
        ct.idxs_packed = pack_12bit_indices(dup)
        with pytest.raises(PayloadError, match="strictly increasing"):
            _validate(p)

    def test_bad_scale(self):
        for bad_scale in (0.0, 1e-13, 1e9, float("inf")):
            p = _make_payload()
            p.compressed["a.weight"].qparams["scale"] = bad_scale
            with pytest.raises(PayloadError, match="scale"):
                _validate(p)

    def test_nonfinite_shift(self):
        p = _make_payload()
        p.compressed["a.weight"].qparams["shift"] = float("nan")
        with pytest.raises(PayloadError, match="shift"):
            _validate(p)

    def test_bad_lookup(self):
        p = _make_payload()
        p.compressed["a.weight"].qparams["lookup"] = torch.zeros(9, dtype=torch.float32)
        with pytest.raises(PayloadError, match="lookup"):
            _validate(p)

    def test_code_exceeding_lookup(self):
        p = _make_payload()
        ct = p.compressed["a.weight"]
        ct.qparams["lookup"] = torch.zeros(2, dtype=torch.float32)  # codes reach 3 in real payloads
        n = ct.n_chunks * ct.topk
        ct.codes_packed = pack_2bit_values(torch.full((n,), 3, dtype=torch.uint8))
        with pytest.raises(PayloadError, match="lookup bins"):
            _validate(p)

    def test_wrong_codes_length(self):
        p = _make_payload()
        p.compressed["a.weight"].codes_packed = torch.zeros(1, dtype=torch.uint8)
        with pytest.raises(PayloadError, match="codes_packed length"):
            _validate(p)

    def test_nonzero_pad_bits_rejected(self):
        p = _make_payload()
        ct = p.compressed["a.weight"]
        n = ct.n_chunks * ct.topk
        if n % 4 == 0:  # force a partial final byte scenario on the bias instead
            ct = p.compressed["b.bias"]
            n = ct.n_chunks * ct.topk
        assert n % 4 != 0
        codes = ct.codes_packed.clone()
        codes[-1] |= 0xC0  # set bits above the used 2*(n%4) positions
        ct.codes_packed = codes
        with pytest.raises(PayloadError, match="pad bits"):
            _validate(p)

    def test_dense_shape_mismatch(self):
        p = _make_payload()
        p.dense["router.balance_bias"] = torch.zeros(6)
        with pytest.raises(PayloadError, match="shape"):
            _validate(p)

    def test_dense_shape_ignored_with_name_set(self):
        p = _make_payload()
        p.dense["router.balance_bias"] = torch.zeros(6)  # wrong shape but only names expected
        validate_structure(
            p, PARAM_SHAPES, set(DENSE_SHAPES), TOPK, target_chunk=TARGET_CHUNK
        )

    def test_dense_nonfinite(self):
        p = _make_payload()
        p.dense["router.balance_bias"] = torch.tensor([1.0, float("inf"), 0.0, 0.0, 0.0])
        with pytest.raises(PayloadError, match="finite"):
            _validate(p)

    def test_dense_wrong_dtype(self):
        p = _make_payload()
        p.dense["router.balance_bias"] = torch.zeros(5, dtype=torch.float64)
        with pytest.raises(PayloadError, match="fp32"):
            _validate(p)

    def test_bad_metadata_hex(self):
        p = _make_payload()
        p = WindowPayload(
            uid=p.uid,
            window=p.window,
            compressed=p.compressed,
            dense=p.dense,
            metadata=PayloadMeta(
                sample_digest="ZZ" * 32,
                sample_count=1,
                theta_end_hash=_HEX_B,
                state_root=_HEX_C,
                global_step=0,
                spec_version=1,
            ),
        )
        with pytest.raises(PayloadError, match="sample_digest"):
            _validate(p)


# --------------------------------------------------------------------------- #
# Ownership assignment
# --------------------------------------------------------------------------- #


def _is_expert(name: str) -> bool:
    return ".experts." in name


class TestAssignOwnedParams:
    NAMES = [
        "layers.0.attn.wq",
        "layers.0.moe.experts.w_in",
        "layers.0.moe.experts.w_out",
        "layers.0.moe.router.weight",
        "layers.1.attn.wq",
        "layers.1.moe.experts.w_in",
        "embed.weight",
        "lm_head.weight",
        "norm.weight",
    ]

    def test_partition_and_coverage(self):
        world = 4
        shared = {n for n in self.NAMES if not _is_expert(n)}
        owned_sets = [assign_owned_params(self.NAMES, r, world, _is_expert) for r in range(world)]
        shared_sets = [s - {n for n in s if _is_expert(n)} for s in owned_sets]
        union: set[str] = set()
        for i, si in enumerate(shared_sets):
            for j in range(i + 1, world):
                assert not si & shared_sets[j], "shared params must be disjoint across ranks"
            union |= si
        assert union == shared, "every shared param must be owned by exactly one rank"

    def test_expert_locals_always_owned(self):
        for rank in range(4):
            owned = assign_owned_params(self.NAMES, rank, 4, _is_expert)
            assert {n for n in self.NAMES if _is_expert(n)} <= owned

    def test_order_invariance(self):
        forward = assign_owned_params(self.NAMES, 2, 4, _is_expert)
        backward = assign_owned_params(list(reversed(self.NAMES)), 2, 4, _is_expert)
        assert forward == backward

    def test_round_robin_matches_sorted_index(self):
        world = 3
        shared_sorted = sorted(n for n in self.NAMES if not _is_expert(n))
        for rank in range(world):
            owned = assign_owned_params(self.NAMES, rank, world, _is_expert)
            expected = {n for i, n in enumerate(shared_sorted) if i % world == rank}
            assert {n for n in owned if not _is_expert(n)} == expected

    def test_world_size_one_owns_everything(self):
        assert assign_owned_params(self.NAMES, 0, 1, _is_expert) == set(self.NAMES)

    def test_validation(self):
        with pytest.raises(ValueError, match="world_size"):
            assign_owned_params(self.NAMES, 0, 0, _is_expert)
        with pytest.raises(ValueError, match="rank"):
            assign_owned_params(self.NAMES, 4, 4, _is_expert)
        with pytest.raises(ValueError, match="duplicate"):
            assign_owned_params(["a", "a"], 0, 2, _is_expert)
