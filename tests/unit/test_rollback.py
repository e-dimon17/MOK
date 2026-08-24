"""subnet/core/rollback.py — spike detection, vote supermajority boundary, timeout,
activation decision + reseed-salt golden."""

from __future__ import annotations

import pytest

from mok_core.config.schemas import RollbackConfig
from subnet.core.rollback import (
    RollbackDecision,
    RollbackState,
    RollbackStateMachine,
    RollbackVote,
    SpikeDetector,
    rollback_salt,
)

RUN_SEED = bytes(32)


def _vote(voter_uid: int, stake: float, target: int = 90, cast: int = 100) -> RollbackVote:
    return RollbackVote(
        voter_uid=voter_uid, stake=stake, target_window=target, window_cast=cast, sig="00" * 64
    )


def _machine() -> RollbackStateMachine:
    # defaults: supermajority 2/3, vote_window_span 2, activation_delay 1
    return RollbackStateMachine(RollbackConfig(), RUN_SEED)


# --------------------------------------------------------------------------- #
# SpikeDetector
# --------------------------------------------------------------------------- #


class TestSpikeDetector:
    def test_no_alert_before_full_baseline(self):
        det = SpikeDetector(threshold_nats=0.15, baseline_windows=3)
        assert not det.observe(0, 2.0)
        assert not det.observe(1, 2.0)
        assert not det.observe(2, 2.0)
        assert not det.observe(3, 2.10)  # baseline full, but below threshold

    def test_alert_strictly_above_threshold(self):
        det = SpikeDetector(threshold_nats=0.15, baseline_windows=3)
        for w in range(3):
            det.observe(w, 2.0)
        assert not det.observe(3, 2.15)  # exactly at threshold: strict > required
        assert det.observe(4, 2.31)      # median now 2.0 -> delta 0.31

    def test_median_baseline(self):
        det = SpikeDetector(threshold_nats=0.15, baseline_windows=3)
        det.observe(0, 1.9)
        det.observe(1, 2.0)
        det.observe(2, 5.0)              # outlier does not drag the median
        assert det.observe(3, 2.2)       # 2.2 - median(1.9, 2.0, 5.0)=2.0 -> 0.2 > 0.15

    def test_identity_windows_are_not_recorded(self):
        # Windows with an identity outer step (applied=False) never alert and do
        # not enter the baseline — an idle streak must not collapse its spread.
        det = SpikeDetector(threshold_nats=0.15, baseline_windows=3)
        for w, loss in enumerate((1.9, 2.1, 2.0)):
            det.observe(w, loss)
        for w in range(3, 40):
            assert not det.observe(w, 2.0, applied=False)      # idle: same loss repeated
        assert not det.observe(40, 99.0, applied=False)         # even an absurd value: ignored
        assert not det.observe(41, 2.14)                        # 2.14 - median(1.9,2.1,2.0)=2.0 -> 0.14, no alert
        assert det.observe(42, 2.31)                            # median(2.1,2.0,2.14)=2.1 -> 0.21 > 0.15

    def test_reset_clears_baseline(self):
        det = SpikeDetector(threshold_nats=0.15, baseline_windows=2)
        det.observe(0, 2.0)
        det.observe(1, 2.0)
        det.reset()
        assert not det.observe(2, 99.0)  # baseline empty again

    def test_invalid_baseline(self):
        with pytest.raises(ValueError):
            SpikeDetector(threshold_nats=0.1, baseline_windows=0)


# --------------------------------------------------------------------------- #
# Reseed salt
# --------------------------------------------------------------------------- #


class TestRollbackSalt:
    def test_golden(self):
        # consensus constant — change requires SPEC_VERSION bump
        assert (
            rollback_salt(bytes(32), 42)
            == "8468c1a72c7325cc485d05af14ebdf75c1f6de57692c1f247adcfffa7f6e724d"
        )

    def test_sensitivity(self):
        assert rollback_salt(bytes(32), 42) != rollback_salt(bytes(32), 43)
        assert rollback_salt(bytes(32), 42) != rollback_salt(b"\x01" * 32, 42)


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


class TestStateMachine:
    def test_alert_opens_voting_with_target(self):
        sm = _machine()
        assert sm.state is RollbackState.NORMAL
        assert sm.on_alert(window=100, checkpoint_window=90)
        assert sm.state is RollbackState.VOTING
        assert sm.target_window == 90
        # ALERTED was passed through and recorded
        assert (100, RollbackState.ALERTED) in sm.history

    def test_alert_ignored_outside_normal(self):
        sm = _machine()
        sm.on_alert(window=100, checkpoint_window=90)
        assert not sm.on_alert(window=101, checkpoint_window=90)

    def test_alert_rejects_future_checkpoint(self):
        with pytest.raises(ValueError):
            _machine().on_alert(window=100, checkpoint_window=100)

    def test_supermajority_exact_boundary_fails(self):
        sm = _machine()
        sm.on_alert(window=100, checkpoint_window=90)
        assert sm.add_vote(_vote(1, 1.0), total_stake=3.0)
        assert sm.add_vote(_vote(2, 1.0), total_stake=3.0)
        # yes = 2/3 exactly: NOT strictly greater than the 2/3 supermajority
        assert sm.state is RollbackState.VOTING
        assert sm.add_vote(_vote(3, 0.5, cast=101), total_stake=3.0)
        assert sm.state is RollbackState.PENDING
        assert sm.activation_window == 102  # tipping cast 101 + delay 1

    def test_duplicate_voter_not_double_counted(self):
        sm = _machine()
        sm.on_alert(window=100, checkpoint_window=90)
        assert sm.add_vote(_vote(1, 2.5), total_stake=3.0)
        assert not sm.add_vote(_vote(1, 2.5), total_stake=3.0)
        assert sm.yes_stake == pytest.approx(2.5)

    def test_wrong_target_and_bad_votes_rejected(self):
        sm = _machine()
        sm.on_alert(window=100, checkpoint_window=90)
        assert not sm.add_vote(_vote(1, 1.0, target=80), total_stake=3.0)
        assert not sm.add_vote(_vote(2, 0.0), total_stake=3.0)          # zero stake
        assert not sm.add_vote(_vote(3, 1.0, cast=103), total_stake=3.0)  # past span 100+2
        assert not sm.add_vote(_vote(4, 1.0, cast=99), total_stake=3.0)   # before alert
        assert sm.yes_stake == 0.0

    def test_votes_rejected_outside_voting_state(self):
        sm = _machine()
        assert not sm.add_vote(_vote(1, 3.0), total_stake=3.0)  # NORMAL

    def test_timeout_returns_to_normal_and_clears(self):
        sm = _machine()
        sm.on_alert(window=100, checkpoint_window=90)
        sm.add_vote(_vote(1, 1.0), total_stake=3.0)
        sm.tick(102)                       # within span (100 + 2)
        assert sm.state is RollbackState.VOTING
        sm.tick(103)                       # past span
        assert sm.state is RollbackState.NORMAL
        assert sm.yes_stake == 0.0
        assert sm.target_window is None
        # a fresh alert can start over
        assert sm.on_alert(window=110, checkpoint_window=100)

    def test_activation_decision(self):
        sm = _machine()
        sm.on_alert(window=100, checkpoint_window=90)
        sm.add_vote(_vote(1, 2.9, cast=100), total_stake=3.0)
        assert sm.state is RollbackState.PENDING
        assert sm.maybe_activate(100) is None          # before activation window (101)
        decision = sm.maybe_activate(101)
        assert isinstance(decision, RollbackDecision)
        assert decision.target_window == 90
        assert decision.void.first_window == 91
        assert decision.void.last_window == 101        # through the activation window (its outer step was applied)
        # consensus constant — change requires SPEC_VERSION bump
        expected_salt = "3ca7d00160286b6e8a7e445b5ff34540ccb3be02fc98081f21bf0f88f27d9607"
        assert decision.reseed_salt_hex == expected_salt
        assert decision.void.reseed_salt_hex == expected_salt
        assert sm.state is RollbackState.NORMAL        # machine reset after activation

    def test_maybe_activate_noop_outside_pending(self):
        sm = _machine()
        assert sm.maybe_activate(500) is None
        sm.on_alert(window=100, checkpoint_window=90)
        assert sm.maybe_activate(500) is None          # VOTING, not PENDING
