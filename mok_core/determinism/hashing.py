"""Tensor and state hashing — the replay verdict functions.

`hash_state_dict` over the master weights IS the protocol's `state_root`:
miners commit it on-chain every window, catch-up verifies it per replayed
window, and an audit passes iff the replayed root equals the committed root
bitwise. Byte rules (golden-vector pinned, SPEC_VERSION-bound):
  - tensors hashed as: dtype tag ‖ ndim ‖ shape(le64 each) ‖ raw little-endian bytes
  - state dicts hashed in sorted-name order: name-utf8 ‖ tensor digest
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

_DIGEST_SIZE = 32

# Stable wire tags — never reuse or renumber.
_DTYPE_TAGS: dict[torch.dtype, int] = {
    torch.float32: 1,
    torch.bfloat16: 2,
    torch.float16: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.uint8: 6,
    torch.int8: 7,
    torch.float8_e4m3fn: 8,
    torch.bool: 9,
    torch.uint16: 10,
}


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a tensor, moved to CPU and made contiguous."""
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    t = t.contiguous()
    if t.numel() == 0:
        return b""
    if t.dim() == 0:
        t = t.reshape(1)
    return t.view(torch.uint8).numpy().tobytes()


def hash_tensor(t: torch.Tensor) -> bytes:
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    tag = _DTYPE_TAGS.get(t.dtype)
    if tag is None:
        raise TypeError(f"unhashable dtype {t.dtype} — extend _DTYPE_TAGS (SPEC_VERSION bump)")
    h.update(tag.to_bytes(2, "little"))
    h.update(int(t.dim()).to_bytes(2, "little"))
    for s in t.shape:
        h.update(int(s).to_bytes(8, "little"))
    h.update(tensor_bytes(t))
    return h.digest()


def hash_named_tensors(named: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hex blake2b-256 over sorted (name, tensor-digest) pairs — the state_root."""
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    for name, tensor in sorted(named, key=lambda kv: kv[0]):
        h.update(len(name).to_bytes(4, "little"))
        h.update(name.encode("utf-8"))
        h.update(hash_tensor(tensor))
    return h.hexdigest()


def hash_state_dict(sd: Mapping[str, torch.Tensor]) -> str:
    return hash_named_tensors(sd.items())


def hash_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=_DIGEST_SIZE).hexdigest()


def hash_file(path: str, chunk: int = 8 << 20) -> str:
    h = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Audit diagnostics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DivergenceRecord:
    name: str
    expected: str  # hex digest ("" = missing)
    actual: str

    def to_json(self) -> dict[str, str]:
        return {"name": self.name, "expected": self.expected, "actual": self.actual}


def first_divergence(
    expected: Mapping[str, bytes],
    actual: Mapping[str, bytes],
    limit: int = 16,
) -> list[DivergenceRecord]:
    """Per-tensor digest comparison for audit mismatch reports."""
    out: list[DivergenceRecord] = []
    for name in sorted(set(expected) | set(actual)):
        e, a = expected.get(name), actual.get(name)
        if e != a:
            out.append(
                DivergenceRecord(
                    name=name,
                    expected=e.hex() if e else "",
                    actual=a.hex() if a else "",
                )
            )
            if len(out) >= limit:
                break
    return out


def per_tensor_digests(named: Iterable[tuple[str, torch.Tensor]]) -> dict[str, bytes]:
    return {name: hash_tensor(t) for name, t in named}
