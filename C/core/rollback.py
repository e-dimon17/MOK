"""Loss-spike rollback: detector, stake-weighted vote, and activation.

When a poisoned/broken window slips into the lineage the probe loss jumps.
Validators detect the spike, vote (stake-weighted) to roll back to the last
checkpoint, and on supermajority every node rewinds at the same activation
window. The voided windows enter the manifest as a VoidRange whose reseed salt
re-randomizes the reassignment of that data (the same shards must not map to
the same miners under the same seeds).

Pure state machine over window numbers — no wall clock, no chain access; the
validator app feeds it observed losses and on-chain votes.
"""

from __future__ import annotations

import hashlib
from collections import deque
from enum import StrEnum
from statistics import median

from mok_core.config.manifest import VoidRange
from mok_core.config.schemas import FrozenModel, RollbackConfig


def rollback_salt(run_seed: bytes, target_window: int) -> str:
    """Hex blake2b-256 of run_seed ‖ b"rollback" ‖ target_window(le64).

    Consensus constant — the PRF reseed salt appended to the manifest by a
    rollback; golden-vector pinned, change requires SPEC_VERSION bump.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(run_seed)
    h.update(b"rollback")
    h.update(int(target_window).to_bytes(8, "little", signed=False))
    return h.hexdigest()


class SpikeDetector:
    """Alert when probe loss exceeds the trailing-median baseline by a margin.

    `observe` returns True iff at least `baseline_windows` losses were seen
    before this one and `probe_loss - median(last baseline_windows) >
    threshold_nats` (strict). The observation is recorded either way; `reset`
    clears the baseline after a rollback (post-rewind losses form a new one).
    """

    def __init__(self, threshold_nats: float, baseline_windows: int) -> None:
        if baseline_windows < 1:
            raise ValueError(f"baseline_windows must be >= 1, got {baseline_windows}")
        self.threshold_nats = threshold_nats
        self.baseline_windows = baseline_windows
        self._losses: deque[float] = deque(maxlen=baseline_windows)

    def observe(self, window: int, probe_loss: float) -> bool:
        del window  # part of the record signature; the baseline is order-based
        alert = (
            len(self._losses) >= self.baseline_windows
            and probe_loss - median(self._losses) > self.threshold_nats
        )
        self._losses.append(float(probe_loss))
        return alert

    def reset(self) -> None:
        self._losses.clear()


class RollbackVote(FrozenModel):
    """One validator's signed yes-vote for rolling back to `target_window`."""

    voter_uid: int
    stake: float
    target_window: int
    window_cast: int
    sig: str            # hotkey signature over the canonical vote bytes (chain layer verifies)


class RollbackDecision(FrozenModel):
    """The activated rollback: rewind target, voided span, and PRF reseed salt."""

    target_window: int
    void: VoidRange
    reseed_salt_hex: str


class RollbackState(StrEnum):
    NORMAL = "normal"
    ALERTED = "alerted"
    VOTING = "voting"
    PENDING = "pending"


class RollbackStateMachine:
    """NORMAL → (alert) → VOTING → (supermajority) → PENDING → (delay) → NORMAL.

    - `on_alert` fixes the rewind target (the last checkpoint window) and opens
      voting; ALERTED is passed through and recorded in `history`.
    - `add_vote` accepts distinct-voter yes-votes for the fixed target cast
      within `vote_window_span` windows of the alert. Supermajority is STRICT:
      yes_stake / total_stake > cfg.vote_supermajority.
    - `tick` at each window boundary times an unresolved vote out back to NORMAL.
    - `maybe_activate` yields the RollbackDecision once the activation window
      (supermajority window + activation_delay_windows) is reached, then resets.
    """

    def __init__(self, cfg: RollbackConfig, run_seed: bytes) -> None:
        self.cfg = cfg
        self.run_seed = run_seed
        self.state: RollbackState = RollbackState.NORMAL
        self.history: list[tuple[int, RollbackState]] = []
        self.alert_window: int | None = None
        self.target_window: int | None = None
        self.activation_window: int | None = None
        self._votes: dict[int, float] = {}   # voter_uid -> stake

    # ------------------------------------------------------------------ #

    def _transition(self, window: int, state: RollbackState) -> None:
        self.state = state
        self.history.append((window, state))

    def _reset(self, window: int) -> None:
        self.alert_window = None
        self.target_window = None
        self.activation_window = None
        self._votes = {}
        self._transition(window, RollbackState.NORMAL)

    @property
    def yes_stake(self) -> float:
        return sum(self._votes.values())

    # ------------------------------------------------------------------ #

    def on_alert(self, window: int, checkpoint_window: int) -> bool:
        """Open voting to rewind to `checkpoint_window`; False if not in NORMAL."""
        if self.state is not RollbackState.NORMAL:
            return False
        if checkpoint_window >= window:
            raise ValueError(
                f"checkpoint_window ({checkpoint_window}) must precede the alert window ({window})"
            )
        self.alert_window = window
        self.target_window = checkpoint_window
        self._votes = {}
        self._transition(window, RollbackState.ALERTED)
        self._transition(window, RollbackState.VOTING)
        return True

    def add_vote(self, vote: RollbackVote, total_stake: float) -> bool:
        """Record one yes-vote; True iff accepted. Trips PENDING on supermajority."""
        if self.state is not RollbackState.VOTING or total_stake <= 0.0:
            return False
        assert self.alert_window is not None and self.target_window is not None
        if (
            vote.target_window != self.target_window
            or vote.voter_uid in self._votes
            or vote.stake <= 0.0
            or not self.alert_window <= vote.window_cast <= self.alert_window + self.cfg.vote_window_span
        ):
            return False
        self._votes[vote.voter_uid] = vote.stake
        if self.yes_stake / total_stake > self.cfg.vote_supermajority:
            self.activation_window = vote.window_cast + self.cfg.activation_delay_windows
            self._transition(vote.window_cast, RollbackState.PENDING)
        return True

    def tick(self, window: int) -> None:
        """Window-boundary clock: expire a vote that missed its span."""
        if (
            self.state is RollbackState.VOTING
            and self.alert_window is not None
            and window > self.alert_window + self.cfg.vote_window_span
        ):
            self._reset(window)

    def maybe_activate(self, window: int) -> RollbackDecision | None:
        """The rollback decision once its activation window arrives, else None."""
        if (
            self.state is not RollbackState.PENDING
            or self.activation_window is None
            or window < self.activation_window
        ):
            return None
        assert self.target_window is not None
        target = self.target_window
        salt = rollback_salt(self.run_seed, target)
        void = VoidRange(
            first_window=target + 1,
            last_window=max(target + 1, self.activation_window - 1),
            reseed_salt_hex=salt,
        )
        decision = RollbackDecision(target_window=target, void=void, reseed_salt_hex=salt)
        self._reset(window)
        return decision
