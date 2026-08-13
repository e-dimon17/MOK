"""MerkleTree: golden root, proof round-trips, duplicate-last padding, tampering."""

from __future__ import annotations

import hashlib

import pytest

from mok_core.data.merkle import MerkleTree


def _h(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=32).digest()


def _leaves(n: int) -> list[bytes]:
    return [_h(f"leaf-{i}".encode()) for i in range(n)]


def test_five_leaf_golden_root() -> None:
    tree = MerkleTree(_leaves(5))
    # consensus constant — change requires SPEC_VERSION bump
    assert tree.root.hex() == "28ccae79fd979a72180c1a7db6aaf548a70cc8c85856774fc02cd10e7e8a4b98"
    assert tree.num_leaves == 5


def test_three_leaf_structure_matches_manual_padding() -> None:
    a, b, c = _leaves(3)
    tree = MerkleTree([a, b, c])
    assert tree.root == _h(_h(a + b) + _h(c + c))


def test_single_leaf_root_is_leaf() -> None:
    (leaf,) = _leaves(1)
    tree = MerkleTree([leaf])
    assert tree.root == leaf
    assert tree.proof(0) == []
    assert MerkleTree.verify(tree.root, leaf, [])


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_proof_roundtrip_all_leaves(n: int) -> None:
    leaves = _leaves(n)
    tree = MerkleTree(leaves)
    for i, leaf in enumerate(leaves):
        assert MerkleTree.verify(tree.root, leaf, tree.proof(i)), (n, i)


def test_tampered_leaf_fails() -> None:
    leaves = _leaves(5)
    tree = MerkleTree(leaves)
    proof = tree.proof(2)
    bad = bytes([leaves[2][0] ^ 1]) + leaves[2][1:]
    assert not MerkleTree.verify(tree.root, bad, proof)


def test_tampered_proof_fails() -> None:
    leaves = _leaves(6)
    tree = MerkleTree(leaves)
    sibling, is_right = tree.proof(0)[0]
    bad_proof = [(bytes(32), is_right), *tree.proof(0)[1:]]
    assert not MerkleTree.verify(tree.root, leaves[0], bad_proof)
    flipped = [(sibling, not is_right), *tree.proof(0)[1:]]
    assert not MerkleTree.verify(tree.root, leaves[0], flipped)


def test_wrong_leaf_index_fails_verification() -> None:
    leaves = _leaves(5)
    tree = MerkleTree(leaves)
    assert not MerkleTree.verify(tree.root, leaves[1], tree.proof(0))


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="at least one leaf"):
        MerkleTree([])
    with pytest.raises(ValueError, match="expected 32"):
        MerkleTree([b"short"])
    tree = MerkleTree(_leaves(3))
    with pytest.raises(IndexError):
        tree.proof(3)
    with pytest.raises(IndexError):
        tree.proof(-1)


def test_leaf_order_matters() -> None:
    leaves = _leaves(4)
    assert MerkleTree(leaves).root != MerkleTree(list(reversed(leaves))).root
