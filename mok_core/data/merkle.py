"""Merkle tree over blake2b-256 leaf hashes — the dataset commitment.

Dataprep commits `merkle_root` in the on-chain `DatasetManifestRef`; every node
verifies downloaded shards against it, and audit reports carry inclusion
proofs. Wire rules (golden-vector pinned, SPEC_VERSION-bound):
  - leaves are 32-byte blake2b-256 digests (of raw shard file bytes)
  - an odd-width level is padded by duplicating its last node before pairing
  - parent = blake2b-256(left ‖ right)
  - a single-leaf tree's root is the leaf itself
"""

from __future__ import annotations

import hashlib

_DIGEST_SIZE = 32

Proof = list[tuple[bytes, bool]]  # (sibling digest, sibling_is_right)


def _parent(left: bytes, right: bytes) -> bytes:
    return hashlib.blake2b(left + right, digest_size=_DIGEST_SIZE).digest()


class MerkleTree:
    """Immutable tree built once from ordered 32-byte leaf digests."""

    __slots__ = ("_levels",)

    def __init__(self, leaves: list[bytes]) -> None:
        if not leaves:
            raise ValueError("MerkleTree requires at least one leaf")
        for i, leaf in enumerate(leaves):
            if len(leaf) != _DIGEST_SIZE:
                raise ValueError(f"leaf {i} is {len(leaf)} bytes; expected {_DIGEST_SIZE}")
        # Levels are stored UNPADDED; the duplicate-last rule is applied on the fly.
        levels: list[list[bytes]] = [list(leaves)]
        while len(levels[-1]) > 1:
            level = levels[-1]
            paired = level if len(level) % 2 == 0 else [*level, level[-1]]
            levels.append([_parent(paired[j], paired[j + 1]) for j in range(0, len(paired), 2)])
        self._levels = levels

    @property
    def num_leaves(self) -> int:
        return len(self._levels[0])

    @property
    def root(self) -> bytes:
        return self._levels[-1][0]

    def proof(self, index: int) -> Proof:
        """Inclusion proof for leaf `index`: bottom-up list of (sibling, sibling_is_right)."""
        if not 0 <= index < self.num_leaves:
            raise IndexError(f"leaf index {index} out of range [0, {self.num_leaves})")
        path: Proof = []
        i = index
        for level in self._levels[:-1]:
            sibling_is_right = i % 2 == 0
            j = i + 1 if sibling_is_right else i - 1
            sibling = level[j] if j < len(level) else level[i]  # duplicate-last padding
            path.append((sibling, sibling_is_right))
            i //= 2
        return path

    @staticmethod
    def verify(root: bytes, leaf: bytes, proof: Proof) -> bool:
        """Check that `leaf` folds up to `root` through `proof`. Pure; no tree needed."""
        node = leaf
        for sibling, sibling_is_right in proof:
            node = _parent(node, sibling) if sibling_is_right else _parent(sibling, node)
        return node == root
