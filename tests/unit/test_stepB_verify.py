"""Tests for B/attestation/verify.py — verdicts + the reattestation draw."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from B.attestation.challenge import make_challenge
from B.attestation.reference_step import AttestationResponse
from B.attestation.verify import AttestationVerdict, Verifier, judge, schedule_reattestation

BLOCK_HASH = bytes([1]) * 32
ROOT = "ab" * 32
RUN_SEED = bytes(range(32))


def _challenge(**kw):
    return make_challenge(BLOCK_HASH, issued_block=7, **kw)


def _response(challenge, root: str = ROOT) -> AttestationResponse:
    return AttestationResponse(
        challenge_id=challenge.challenge_id, state_root=root, wall_time_s=55.0, fingerprint={}
    )


def test_judge_accepts_matching_in_time_response() -> None:
    ch = _challenge()
    verdict = judge(ch, _response(ch), ROOT, received_ts=1400.0, issued_ts=1000.0)
    assert verdict == AttestationVerdict(ok=True, reason="")


def test_judge_is_case_insensitive_on_roots() -> None:
    ch = _challenge()
    verdict = judge(ch, _response(ch), ROOT.upper(), received_ts=1100.0, issued_ts=1000.0)
    assert verdict.ok


def test_judge_rejects_wrong_root() -> None:
    ch = _challenge()
    verdict = judge(ch, _response(ch, "cd" * 32), ROOT, received_ts=1100.0, issued_ts=1000.0)
    assert not verdict.ok
    assert "state_root mismatch" in verdict.reason


def test_judge_rejects_deadline_miss_and_names_the_hardware_gate() -> None:
    ch = _challenge(deadline_s=420.0)
    verdict = judge(ch, _response(ch), ROOT, received_ts=1000.0 + 420.1, issued_ts=1000.0)
    assert not verdict.ok
    assert "deadline exceeded" in verdict.reason
    assert "SM103" in verdict.reason  # the deadline IS the hardware check


def test_judge_boundary_elapsed_exactly_deadline_passes() -> None:
    ch = _challenge(deadline_s=420.0)
    assert judge(ch, _response(ch), ROOT, received_ts=1420.0, issued_ts=1000.0).ok


def test_judge_rejects_out_of_order_timestamps() -> None:
    ch = _challenge()
    verdict = judge(ch, _response(ch), ROOT, received_ts=999.0, issued_ts=1000.0)
    assert not verdict.ok
    assert "out of order" in verdict.reason


def test_judge_rejects_foreign_challenge_id_and_collects_all_problems() -> None:
    ch = _challenge()
    other = make_challenge(bytes([9]) * 32, 8)
    verdict = judge(ch, _response(other, "cd" * 32), ROOT, received_ts=9999.0, issued_ts=0.0)
    assert not verdict.ok
    assert "challenge_id mismatch" in verdict.reason
    assert "deadline exceeded" in verdict.reason
    assert "state_root mismatch" in verdict.reason


# --------------------------------------------------------------------------- #
# Reattestation scheduling
# --------------------------------------------------------------------------- #


def test_schedule_reattestation_golden_vector() -> None:
    """# consensus constant — change requires SPEC_VERSION bump"""
    picked = schedule_reattestation(RUN_SEED, bytes([1]) * 32, range(20), 0.25)
    assert picked == [8, 11, 16, 17, 18]


def test_schedule_reattestation_deterministic_and_sorted() -> None:
    uids = [17, 3, 9, 3, 12, 0]
    a = schedule_reattestation(RUN_SEED, BLOCK_HASH, uids, 0.5)
    b = schedule_reattestation(RUN_SEED, BLOCK_HASH, list(reversed(uids)), 0.5)
    assert a == b == sorted(a)
    assert set(a) <= set(uids)


def test_schedule_reattestation_rate_edges() -> None:
    uids = list(range(50))
    assert schedule_reattestation(RUN_SEED, BLOCK_HASH, uids, 0.0) == []
    assert schedule_reattestation(RUN_SEED, BLOCK_HASH, uids, 1.0) == uids
    assert schedule_reattestation(RUN_SEED, BLOCK_HASH, [], 0.5) == []
    with pytest.raises(ValueError, match="rate"):
        schedule_reattestation(RUN_SEED, BLOCK_HASH, uids, 1.5)


def test_schedule_varies_with_block_hash() -> None:
    uids = list(range(100))
    a = schedule_reattestation(RUN_SEED, bytes([1]) * 32, uids, 0.3)
    b = schedule_reattestation(RUN_SEED, bytes([2]) * 32, uids, 0.3)
    assert a != b


# --------------------------------------------------------------------------- #
# Verifier
# --------------------------------------------------------------------------- #


def test_verifier_issue_uses_chain_head() -> None:
    chain = MagicMock()
    chain.current_block.return_value = 1234
    chain.block_hash.return_value = BLOCK_HASH
    v = Verifier(deadline_s=300.0, inner_steps=5)
    ch = v.issue(chain)
    chain.block_hash.assert_called_once_with(1234)
    assert ch.issued_block == 1234
    assert ch.deadline_s == 300.0
    assert ch.inner_steps == 5
    assert ch == make_challenge(BLOCK_HASH, 1234, deadline_s=300.0, inner_steps=5)


def test_verifier_methods_delegate() -> None:
    v = Verifier()
    ch = _challenge()
    assert v.judge(ch, _response(ch), ROOT, received_ts=1.0, issued_ts=0.0).ok
    assert v.schedule_reattestation(RUN_SEED, BLOCK_HASH, [1, 2], 1.0) == [1, 2]
