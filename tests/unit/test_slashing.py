"""subnet/core/slashing.py — every ladder transition, 2-of-3 quorum, naughty expiry."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from mok_core.config.schemas import AuditConfig
from subnet.core.slashing import SlashLedger


@dataclass(frozen=True)
class Report:
    auditor_uid: int
    match: bool


def _ledger(**kwargs) -> SlashLedger:
    return SlashLedger(AuditConfig(), **kwargs)


class TestMissingGradientLadder:
    def test_escalation_075_05_00(self):
        led = _ledger()
        r1 = led.missing_gradient(1, window=10)
        r2 = led.missing_gradient(1, window=11)
        r3 = led.missing_gradient(1, window=12)
        r4 = led.missing_gradient(1, window=13)
        assert (r1.multiplier, r2.multiplier, r3.multiplier, r4.multiplier) == (0.75, 0.5, 0.0, 0.0)
        assert led.apply(1, 1.0, 10) == pytest.approx(0.75)
        assert led.apply(1, 1.0, 11) == pytest.approx(0.5)
        assert led.apply(1, 1.0, 12) == 0.0
        assert led.apply(1, 1.0, 13) == 0.0

    def test_success_resets_escalation(self):
        led = _ledger()
        led.missing_gradient(1, window=10)
        led.missing_gradient(1, window=11)
        led.gradient_received(1, window=12)
        assert led.missing_gradient(1, window=13).multiplier == 0.75  # ladder restarted

    def test_independent_uids(self):
        led = _ledger()
        led.missing_gradient(1, window=10)
        assert led.missing_gradient(2, window=10).multiplier == 0.75


class TestSingleWindowEvents:
    def test_invalid_payload_zeroes_only_that_window(self):
        led = _ledger()
        led.invalid_payload(1, window=5)
        assert led.apply(1, 0.8, 5) == 0.0
        assert led.apply(1, 0.8, 6) == pytest.approx(0.8)

    def test_sync_behind(self):
        led = _ledger()
        assert led.sync_behind(1, window=5).multiplier == 0.75
        assert led.apply(1, 1.0, 5) == pytest.approx(0.75)

    def test_events_compound_within_a_window(self):
        led = _ledger()
        led.sync_behind(1, window=5)
        led.missing_gradient(1, window=5)
        assert led.apply(1, 1.0, 5) == pytest.approx(0.75 * 0.75)

    def test_nonpositive_base_passes_through(self):
        led = _ledger()
        led.sync_behind(1, window=5)
        assert led.apply(1, 0.0, 5) == 0.0
        assert led.apply(1, -0.2, 5) == -0.2  # never nudged toward zero from below

    def test_unknown_uid_untouched(self):
        assert _ledger().apply(42, 0.7, 3) == pytest.approx(0.7)


class TestInactivity:
    def test_075_per_window(self):
        led = _ledger(inactivity_reset_windows=5)
        for w in (1, 2, 3):
            assert led.inactivity(9, window=w).reason == "inactivity"
            assert led.apply(9, 1.0, w) == pytest.approx(0.75)

    def test_full_reset_after_n_consecutive(self):
        led = _ledger(inactivity_reset_windows=3)
        led.missing_gradient(9, window=0)  # some pre-existing state
        led.inactivity(9, window=1)
        led.inactivity(9, window=2)
        record = led.inactivity(9, window=3)
        assert record.reason == "inactivity_reset"
        # state wiped: no residual multipliers, escalators back to zero
        assert led.apply(9, 1.0, 1) == pytest.approx(1.0)
        assert led.missing_gradient(9, window=4).multiplier == 0.75

    def test_activity_resets_inactivity_count(self):
        led = _ledger(inactivity_reset_windows=3)
        led.inactivity(9, window=1)
        led.inactivity(9, window=2)
        led.gradient_received(9, window=3)
        assert led.inactivity(9, window=4).reason == "inactivity"  # count restarted at 1
        assert led.inactivity(9, window=5).reason == "inactivity"
        assert led.inactivity(9, window=6).reason == "inactivity_reset"


class TestOverlapEvent:
    def test_high_multiplier_no_naughty(self):
        led = _ledger()
        record = led.overlap(4, window=7, multiplier=0.5, naughty=False)
        assert record.naughty_until is None
        assert led.apply(4, 1.0, 7) == pytest.approx(0.5)
        assert not led.is_naughty(4, 8)

    def test_mega_joins_naughty_list(self):
        led = _ledger()  # AuditConfig.naughty_windows = 20
        record = led.overlap(4, window=7, multiplier=0.0, naughty=True)
        assert record.naughty_until == 27
        for w in (7, 15, 26):
            assert led.is_naughty(4, w)
            assert led.apply(4, 1.0, w) == 0.0
        assert not led.is_naughty(4, 27)
        assert led.apply(4, 1.0, 27) == pytest.approx(1.0)


class TestAuditQuorum:
    def test_two_of_three_slashes(self):
        led = _ledger()
        record = led.audit_verdicts(
            5, window=30, reports=[Report(100, False), Report(101, False), Report(102, True)]
        )
        assert record is not None
        assert record.reason == "audit"
        assert record.naughty_until == 50
        assert led.apply(5, 1.0, 30) == 0.0

    def test_duplicate_auditor_counts_once(self):
        led = _ledger()
        record = led.audit_verdicts(
            5, window=30, reports=[Report(100, False), Report(100, False), Report(102, True)]
        )
        assert record is None
        assert led.apply(5, 1.0, 30) == pytest.approx(1.0)

    def test_single_mismatch_below_quorum(self):
        led = _ledger()
        assert led.audit_verdicts(5, window=30, reports=[Report(100, False)]) is None

    def test_all_match_no_action(self):
        led = _ledger()
        assert (
            led.audit_verdicts(5, window=30, reports=[Report(100, True), Report(101, True)]) is None
        )

    def test_naughty_expiry(self):
        led = _ledger()
        led.audit_verdicts(5, window=30, reports=[Report(100, False), Report(101, False)])
        assert led.apply(5, 1.0, 49) == 0.0     # last naughty window
        assert led.apply(5, 1.0, 50) == pytest.approx(1.0)

    def test_repeat_slash_extends_naughty_span(self):
        led = _ledger()
        led.audit_verdicts(5, window=30, reports=[Report(100, False), Report(101, False)])
        led.audit_verdicts(5, window=40, reports=[Report(100, False), Report(102, False)])
        assert led.is_naughty(5, 59)
        assert not led.is_naughty(5, 60)


class TestPersistence:
    def test_state_roundtrip_via_json(self):
        led = _ledger()
        led.missing_gradient(1, window=10)
        led.missing_gradient(1, window=11)
        led.sync_behind(2, window=11)
        led.overlap(3, window=12, multiplier=0.0, naughty=True)
        led.audit_verdicts(4, window=12, reports=[Report(100, False), Report(101, False)])

        state = json.loads(json.dumps(led.state_dict()))
        restored = _ledger()
        restored.load_state_dict(state)

        for uid, window in [(1, 10), (1, 11), (2, 11), (3, 12), (3, 20), (4, 12), (4, 31)]:
            assert restored.apply(uid, 1.0, window) == led.apply(uid, 1.0, window)
        assert restored.is_naughty(3, 20) and restored.is_naughty(4, 31)
        assert [r.to_json() for r in restored.records] == [r.to_json() for r in led.records]
        # escalator state survives: next miss for uid 1 is the third rung (x0.0)
        assert restored.missing_gradient(1, window=12).multiplier == 0.0

    def test_reset_clears_uid(self):
        led = _ledger()
        led.overlap(3, window=12, multiplier=0.0, naughty=True)
        led.reset(3)
        assert not led.is_naughty(3, 13)
        assert led.apply(3, 1.0, 12) == pytest.approx(1.0)
