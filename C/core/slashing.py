"""SlashLedger — deterministic per-uid score sanctions.

Every protocol violation lands here as an event keyed by (uid, window); the
validator then calls `apply(uid, base_score, window)` when converting final
scores into weights. All state is pure bookkeeping over window numbers — no
wall clock — so every validator that sees the same events computes the same
sanctions.

Event ladder:
  missing_gradient  x0.75 / x0.5 / x0.0 by consecutive count; reset on success
  invalid_payload   x0.0 this window
  sync_behind       x0.75 this window
  inactivity        x0.75 per window; full state reset after N consecutive
  overlap           multiplier from overlap.severity; 'mega' => naughty list
  audit_verdicts    >= quorum distinct auditors reproduce a mismatch =>
                    x0.0 + naughty for AuditConfig.naughty_windows (2-of-3 rule)

Naughty uids score 0.0 for every window in [start, start + span).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from mok_core.config.schemas import AuditConfig

#: Escalating multipliers for the 1st, 2nd, 3rd+ consecutive missing gradient.
MISSING_GRADIENT_LADDER: tuple[float, ...] = (0.75, 0.5, 0.0)
SYNC_BEHIND_MULTIPLIER: float = 0.75
INACTIVITY_MULTIPLIER: float = 0.75


@runtime_checkable
class AuditReportLike(Protocol):
    """Minimal shape of an audit verdict (C/core/replay.py's AuditReport satisfies it)."""

    auditor_uid: int
    match: bool


@dataclass(frozen=True)
class SlashRecord:
    """One applied sanction — the ledger's audit trail entry."""

    uid: int
    window: int
    reason: str          # "missing_gradient" | "invalid_payload" | "sync_behind" |
    #                      "inactivity" | "inactivity_reset" | "overlap" | "audit"
    multiplier: float
    naughty_until: int | None = None   # exclusive upper bound of the naughty span
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "window": self.window,
            "reason": self.reason,
            "multiplier": self.multiplier,
            "naughty_until": self.naughty_until,
            "detail": self.detail,
        }


@dataclass
class _UidState:
    consecutive_missing: int = 0
    consecutive_inactive: int = 0
    naughty_until: int = -1                              # exclusive; -1 = never naughty
    window_multipliers: dict[int, float] = field(default_factory=dict)

    def apply_multiplier(self, window: int, multiplier: float) -> None:
        self.window_multipliers[window] = self.window_multipliers.get(window, 1.0) * multiplier


class SlashLedger:
    """Multiplicative per-uid sanction state applied on top of final scores."""

    def __init__(
        self,
        audit_cfg: AuditConfig | None = None,
        *,
        inactivity_reset_windows: int = 25,
        overlap_naughty_windows: int | None = None,
    ) -> None:
        self.audit_cfg = audit_cfg or AuditConfig()
        self.inactivity_reset_windows = inactivity_reset_windows
        self.overlap_naughty_windows = (
            overlap_naughty_windows
            if overlap_naughty_windows is not None
            else self.audit_cfg.naughty_windows
        )
        self._state: dict[int, _UidState] = {}
        self.records: list[SlashRecord] = []

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def _uid(self, uid: int) -> _UidState:
        return self._state.setdefault(uid, _UidState())

    def _record(self, record: SlashRecord) -> SlashRecord:
        self.records.append(record)
        return record

    def missing_gradient(self, uid: int, window: int) -> SlashRecord:
        """Escalating penalty per consecutive missed upload (x0.75 / x0.5 / x0.0)."""
        st = self._uid(uid)
        st.consecutive_missing += 1
        ladder_pos = min(st.consecutive_missing, len(MISSING_GRADIENT_LADDER)) - 1
        multiplier = MISSING_GRADIENT_LADDER[ladder_pos]
        st.apply_multiplier(window, multiplier)
        return self._record(
            SlashRecord(
                uid=uid,
                window=window,
                reason="missing_gradient",
                multiplier=multiplier,
                detail=f"consecutive={st.consecutive_missing}",
            )
        )

    def gradient_received(self, uid: int, window: int) -> None:
        """A valid on-time upload: resets the missing and inactivity escalators."""
        del window  # part of the event signature; escalators are count-based
        st = self._uid(uid)
        st.consecutive_missing = 0
        st.consecutive_inactive = 0

    def invalid_payload(self, uid: int, window: int) -> SlashRecord:
        """Malformed / out-of-bounds payload: score zero this window."""
        self._uid(uid).apply_multiplier(window, 0.0)
        return self._record(
            SlashRecord(uid=uid, window=window, reason="invalid_payload", multiplier=0.0)
        )

    def sync_behind(self, uid: int, window: int) -> SlashRecord:
        """Persistent desync beyond tolerance: x0.75 this window."""
        self._uid(uid).apply_multiplier(window, SYNC_BEHIND_MULTIPLIER)
        return self._record(
            SlashRecord(
                uid=uid, window=window, reason="sync_behind", multiplier=SYNC_BEHIND_MULTIPLIER
            )
        )

    def inactivity(self, uid: int, window: int) -> SlashRecord:
        """x0.75 per inactive window; after `inactivity_reset_windows` consecutive
        the uid's ledger state fully resets (reason="inactivity_reset") — the
        caller should also reset its BinaryEMA / OpenSkillBook entries."""
        st = self._uid(uid)
        st.consecutive_inactive += 1
        if st.consecutive_inactive >= self.inactivity_reset_windows:
            self.reset(uid)
            return self._record(
                SlashRecord(
                    uid=uid,
                    window=window,
                    reason="inactivity_reset",
                    multiplier=0.0,
                    detail=f"after={self.inactivity_reset_windows} windows",
                )
            )
        st.apply_multiplier(window, INACTIVITY_MULTIPLIER)
        return self._record(
            SlashRecord(
                uid=uid,
                window=window,
                reason="inactivity",
                multiplier=INACTIVITY_MULTIPLIER,
                detail=f"consecutive={st.consecutive_inactive}",
            )
        )

    def overlap(self, uid: int, window: int, multiplier: float, naughty: bool) -> SlashRecord:
        """Sanction from C/core/overlap.severity for the offender of a flagged pair."""
        st = self._uid(uid)
        st.apply_multiplier(window, multiplier)
        naughty_until: int | None = None
        if naughty:
            naughty_until = window + self.overlap_naughty_windows
            st.naughty_until = max(st.naughty_until, naughty_until)
        return self._record(
            SlashRecord(
                uid=uid,
                window=window,
                reason="overlap",
                multiplier=multiplier,
                naughty_until=naughty_until,
            )
        )

    def audit_verdicts(
        self, uid: int, window: int, reports: list[AuditReportLike]
    ) -> SlashRecord | None:
        """2-of-3 rule: >= AuditConfig.quorum DISTINCT auditors reproducing a
        bitwise mismatch => score zero + naughty for AuditConfig.naughty_windows.
        Duplicate reports from one auditor count once. Below quorum: no action."""
        mismatch_auditors = {r.auditor_uid for r in reports if not r.match}
        if len(mismatch_auditors) < self.audit_cfg.quorum:
            return None
        st = self._uid(uid)
        st.apply_multiplier(window, 0.0)
        naughty_until = window + self.audit_cfg.naughty_windows
        st.naughty_until = max(st.naughty_until, naughty_until)
        return self._record(
            SlashRecord(
                uid=uid,
                window=window,
                reason="audit",
                multiplier=0.0,
                naughty_until=naughty_until,
                detail=f"mismatch_auditors={sorted(mismatch_auditors)}",
            )
        )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #

    def is_naughty(self, uid: int, window: int) -> bool:
        st = self._state.get(uid)
        return st is not None and window < st.naughty_until

    def apply(self, uid: int, base_score: float, window: int) -> float:
        """The sanctioned score for (uid, window): 0.0 while naughty, otherwise
        base_score times the product of this window's event multipliers.
        Non-positive base scores pass through un-multiplied (a penalty must
        never move a score toward zero from below)."""
        if self.is_naughty(uid, window):
            return 0.0
        st = self._state.get(uid)
        if st is None or base_score <= 0.0:
            return base_score
        return base_score * st.window_multipliers.get(window, 1.0)

    def reset(self, uid: int) -> None:
        """Wipe uid's ledger state (deregistration or inactivity reset)."""
        self._state.pop(uid, None)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": {
                str(uid): {
                    "consecutive_missing": st.consecutive_missing,
                    "consecutive_inactive": st.consecutive_inactive,
                    "naughty_until": st.naughty_until,
                    "window_multipliers": {str(w): m for w, m in st.window_multipliers.items()},
                }
                for uid, st in self._state.items()
            },
            "records": [r.to_json() for r in self.records],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._state = {
            int(uid): _UidState(
                consecutive_missing=int(s["consecutive_missing"]),
                consecutive_inactive=int(s["consecutive_inactive"]),
                naughty_until=int(s["naughty_until"]),
                window_multipliers={int(w): float(m) for w, m in s["window_multipliers"].items()},
            )
            for uid, s in state.get("state", {}).items()
        }
        self.records = [
            SlashRecord(
                uid=int(r["uid"]),
                window=int(r["window"]),
                reason=str(r["reason"]),
                multiplier=float(r["multiplier"]),
                naughty_until=None if r.get("naughty_until") is None else int(r["naughty_until"]),
                detail=str(r.get("detail", "")),
            )
            for r in state.get("records", [])
        ]
