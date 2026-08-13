"""Canonical serialization + hashing of config/manifest objects.

These hashes are consensus constants: two nodes agree they are in the same run
iff their canonical hashes match. Byte stability rules:
  - JSON with sorted keys, no whitespace, ensure_ascii=False
  - floats via Python's shortest-repr (stable for IEEE-754 doubles)
  - pydantic models serialized in "json" mode (tuples -> lists, enums -> values)
Golden-vector tests pin outputs; a change is a deliberate SPEC_VERSION bump.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

_DIGEST_SIZE = 32


def canonical_bytes(obj: Any) -> bytes:
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_hash(obj: Any) -> str:
    """Hex blake2b-256 of the canonical serialization."""
    return hashlib.blake2b(canonical_bytes(obj), digest_size=_DIGEST_SIZE).hexdigest()


def config_hash(cfg: BaseModel) -> str:
    return canonical_hash(cfg)
