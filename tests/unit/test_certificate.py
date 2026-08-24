"""Tests for subnet/core/certificate.py — ranking, reserve promotion, wire format, verification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from subnet.core.certificate import (
    WindowCertificate,
    build_certificate,
    certificate_message,
    verify_certificate,
)

# Fake ed25519-style signer: deterministic keyed hash. sign = blake2b(secret + msg);
# verify recomputes. Production injects sr25519 from the chain layer.
_SECRET = b"leader-secret"
_WRONG_SECRET = b"impostor-secret"


def _sign(msg: bytes) -> bytes:
    return hashlib.blake2b(_SECRET + msg, digest_size=64).digest()


def _verify(msg: bytes, sig: bytes) -> bool:
    return sig == hashlib.blake2b(_SECRET + msg, digest_size=64).digest()


def _verify_wrong_key(msg: bytes, sig: bytes) -> bool:
    return sig == hashlib.blake2b(_WRONG_SECRET + msg, digest_size=64).digest()


@dataclass
class Commit:
    uid: int
    payload_hash: str
    in_gate: bool = True
    valid: bool = True


def _hash_for(uid: int) -> str:
    return hashlib.blake2b(f"payload-{uid}".encode(), digest_size=32).hexdigest()


def _commits() -> dict[int, Commit]:
    return {
        1: Commit(1, _hash_for(1)),                    # score 0.9
        2: Commit(2, _hash_for(2)),                    # score 0.9 (tie with 1)
        3: Commit(3, _hash_for(3), valid=False),       # in top set but invalid -> dropped
        4: Commit(4, _hash_for(4)),                    # score 0.7
        5: Commit(5, _hash_for(5), in_gate=False),     # best score but missed the gate
        6: Commit(6, _hash_for(6)),                    # reserve, valid -> promoted
        7: Commit(7, _hash_for(7), valid=False),       # reserve, invalid -> skipped
        8: Commit(8, _hash_for(8)),                    # beyond reserve_count -> never promoted
    }


_SCORES = {1: 0.9, 2: 0.9, 3: 0.8, 4: 0.7, 5: 0.99, 6: 0.6, 7: 0.5, 8: 0.4}
_ROOT = "1f" * 32


def _build(commits=None, scores=None, gather=4, reserve=2) -> WindowCertificate:
    return build_certificate(
        window=42,
        commits=commits if commits is not None else _commits(),
        scores=scores if scores is not None else _SCORES,
        gather_count=gather,
        reserve_count=reserve,
        theta_start_root=_ROOT,
        leader_uid=250,
        sign=_sign,
    )


# --------------------------------------------------------------------------- #
# build_certificate
# --------------------------------------------------------------------------- #


def test_build_ranking_gate_and_reserve_promotion():
    cert = _build()
    # top4 by (score desc, uid asc): 1, 2, 3, 4 (5 is out of gate); 3 invalid ->
    # promote 6 from the reserve pool (7 invalid, 8 beyond reserve_count).
    assert cert.included_uids == (1, 2, 4, 6)
    assert cert.window == 42
    assert cert.leader_uid == 250
    assert cert.theta_start_root == _ROOT
    assert cert.payload_hashes == {uid: _hash_for(uid) for uid in (1, 2, 4, 6)}
    assert verify_certificate(cert, _commits(), _verify)


def test_build_tie_break_by_uid_ascending():
    commits = {uid: Commit(uid, _hash_for(uid)) for uid in (10, 3, 7, 5)}
    cert = _build(commits=commits, scores=dict.fromkeys(commits, 1.0), gather=2, reserve=0)
    assert cert.included_uids == (3, 5)


def test_build_independent_of_dict_insertion_order():
    forward = _build()
    reversed_commits = dict(reversed(list(_commits().items())))
    reversed_scores = dict(reversed(list(_SCORES.items())))
    backward = _build(commits=reversed_commits, scores=reversed_scores)
    assert forward == backward  # includes leader_sig — full bitwise agreement


def test_build_reserve_pool_is_bounded():
    commits = _commits()
    commits[1].valid = False
    commits[2].valid = False
    # top4 = 1,2,3,4 with 1,2,3 invalid; reserve pool = [6, 7] only -> just 6 promoted.
    cert = _build(commits=commits)
    assert cert.included_uids == (4, 6)


def test_build_fewer_eligible_than_gather_count():
    commits = {1: Commit(1, _hash_for(1)), 2: Commit(2, _hash_for(2), in_gate=False)}
    cert = _build(commits=commits, scores={1: 0.5}, gather=20, reserve=10)
    assert cert.included_uids == (1,)


def test_build_missing_score_defaults_to_zero():
    commits = {1: Commit(1, _hash_for(1)), 2: Commit(2, _hash_for(2))}
    cert = _build(commits=commits, scores={2: 0.5}, gather=1, reserve=0)
    assert cert.included_uids == (2,)


def test_build_rejects_bad_counts():
    with pytest.raises(ValueError):
        _build(gather=0)
    with pytest.raises(ValueError):
        _build(reserve=-1)


# --------------------------------------------------------------------------- #
# certificate_message — wire format
# --------------------------------------------------------------------------- #

_GOLDEN_CERT = WindowCertificate(
    window=42,
    included_uids=(3, 7, 11),
    payload_hashes={3: "aa" * 32, 7: "bb" * 32, 11: "cc" * 32},
    theta_start_root="1f" * 32,
    leader_uid=250,
)


def test_certificate_message_golden_vector():
    msg = certificate_message(_GOLDEN_CERT)
    assert len(msg) == 32
    # consensus constant — change requires SPEC_VERSION bump
    assert msg.hex() == "1733d1d9a7d79f3c88dbbd1d3fad9732508bb0cc7bd2bb8664f8ff095e99859e"


def test_certificate_message_excludes_signature():
    signed = _GOLDEN_CERT.model_copy(update={"leader_sig": "ab" * 64})
    assert certificate_message(signed) == certificate_message(_GOLDEN_CERT)


def test_certificate_message_binds_every_signed_field():
    base = certificate_message(_GOLDEN_CERT)
    for update in (
        {"window": 43},
        {"included_uids": (3, 7, 12)},
        {"payload_hashes": {3: "aa" * 32, 7: "bb" * 32, 11: "dd" * 32}},
        {"theta_start_root": "2f" * 32},
        {"leader_uid": 251},
    ):
        assert certificate_message(_GOLDEN_CERT.model_copy(update=update)) != base, update


# --------------------------------------------------------------------------- #
# verify_certificate
# --------------------------------------------------------------------------- #


def test_verify_accepts_good_certificate():
    assert verify_certificate(_build(), _commits(), _verify)


def test_verify_rejects_tampered_payload_hash():
    cert = _build()
    tampered_hashes = dict(cert.payload_hashes)
    tampered_hashes[1] = "00" * 32
    tampered = cert.model_copy(update={"payload_hashes": tampered_hashes})
    assert not verify_certificate(tampered, _commits(), _verify)


def test_verify_rejects_unsorted_or_duplicate_uids():
    cert = _build()
    unsorted = cert.model_copy(update={"included_uids": tuple(reversed(cert.included_uids))})
    assert not verify_certificate(unsorted, _commits(), _verify)
    dupes = cert.model_copy(
        update={"included_uids": (1, 1, 2, 4, 6), "payload_hashes": cert.payload_hashes}
    )
    assert not verify_certificate(dupes, _commits(), _verify)


def test_verify_rejects_bad_or_malformed_signature():
    cert = _build()
    sig = bytearray(bytes.fromhex(cert.leader_sig))
    sig[0] ^= 0xFF
    assert not verify_certificate(cert.model_copy(update={"leader_sig": bytes(sig).hex()}), _commits(), _verify)
    assert not verify_certificate(cert.model_copy(update={"leader_sig": "zz-not-hex"}), _commits(), _verify)
    assert not verify_certificate(cert, _commits(), _verify_wrong_key)


def test_verify_rejects_uid_missing_from_chain():
    cert = _build()
    chain = _commits()
    del chain[6]
    assert not verify_certificate(cert, chain, _verify)


def test_verify_rejects_extra_payload_hash_entries():
    cert = _build()
    padded = dict(cert.payload_hashes)
    padded[99] = _hash_for(99)
    assert not verify_certificate(cert.model_copy(update={"payload_hashes": padded}), _commits(), _verify)


def test_verify_rejects_resigned_certificate_content_swap():
    # An attacker re-signs a modified set with their own key material.
    cert = _build()
    swapped = cert.model_copy(update={"included_uids": (1, 2, 4, 8)})
    resigned = swapped.model_copy(
        update={
            "payload_hashes": {uid: _hash_for(uid) for uid in (1, 2, 4, 8)},
            "leader_sig": hashlib.blake2b(_WRONG_SECRET + certificate_message(swapped), digest_size=64)
            .digest()
            .hex(),
        }
    )
    assert not verify_certificate(resigned, _commits(), _verify)
