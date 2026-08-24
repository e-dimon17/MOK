"""Audit-report ingestion: poll auditor buckets, verify signatures, feed the
2-of-3 slash rule.

Trust model (documented in subnet/miner/bootstrap.py): auditors advertise
themselves with the `AUDITOR_COMMITMENT` chain tag; their reports live in
their own buckets under `keys.audit_report_key(window, auditor, miner)`.
Report authenticity = the hotkey signature over the canonical unsigned fields
(`subnet.core.replay.verify_report` + `chain.verify`); a report failing signature
verification is dropped, never raised. Slashing needs `AuditConfig.quorum`
DISTINCT auditors reproducing a mismatch (`SlashLedger.audit_verdicts`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mok_core.config.schemas import BucketCreds
from mok_core.telemetry import get_logger
from subnet.core.exchange import list_audit_reports
from subnet.core.replay import verify_report
from subnet.core.slashing import SlashLedger, SlashRecord
from subnet.miner.bootstrap import auditor_uids_from_chain

__all__ = ["AuditVerdict", "collect_audit_verdicts", "ingest_window_audits"]

log = get_logger("app.validator.audit")


@dataclass(frozen=True)
class AuditVerdict:
    """`slashing.AuditReportLike` — one auditor's verdict on one miner-window."""

    auditor_uid: int
    match: bool


async def collect_audit_verdicts(
    storage: Any,
    chain: Any,
    window: int,
    auditor_buckets: Mapping[int, BucketCreds],
    *,
    verify_signatures: bool = True,
) -> dict[int, list[AuditVerdict]]:
    """miner_uid -> verified verdicts for `window`, across all auditor buckets.

    Only reports keyed to (and claiming) the polled auditor's uid count, and at
    most one verdict per (auditor, miner) — duplicates keep the first, so one
    auditor can never fake a quorum.
    """
    per_miner: dict[int, dict[int, AuditVerdict]] = {}
    for auditor_uid in sorted(auditor_buckets):
        bucket = auditor_buckets[auditor_uid]
        hotkey = chain.hotkey_of(auditor_uid)
        for report in await list_audit_reports(storage, bucket, window):
            if report.get("auditor_uid") != auditor_uid:
                continue
            if verify_signatures:
                if hotkey is None:
                    log.warning("auditor has no hotkey — reports dropped", auditor=auditor_uid)
                    break
                if not verify_report(report, lambda m, s, hk=hotkey: chain.verify(hk, m, s)):
                    log.warning(
                        "audit report signature invalid",
                        auditor=auditor_uid,
                        miner=report.get("miner_uid"),
                        window=window,
                    )
                    continue
            miner = report.get("miner_uid")
            if not isinstance(miner, int):
                continue
            per_miner.setdefault(miner, {}).setdefault(
                auditor_uid, AuditVerdict(auditor_uid=auditor_uid, match=bool(report.get("match")))
            )
    return {miner: list(by_auditor.values()) for miner, by_auditor in per_miner.items()}


async def ingest_window_audits(
    storage: Any,
    chain: Any,
    window: int,
    ledger: SlashLedger,
    *,
    apply_window: int | None = None,
    verify_signatures: bool = True,
) -> list[SlashRecord]:
    """Discover auditors on-chain, ingest their reports for audited `window`,
    apply the quorum rule at `apply_window` (defaults to `window`; validators
    pass their current processing window so the zeroed score lands on live
    emissions). Returns the slash records issued (possibly empty)."""
    auditor_uids = auditor_uids_from_chain(chain)
    buckets: dict[int, BucketCreds] = {}
    for uid in auditor_uids:
        bucket = chain.get_bucket(uid)
        if bucket is not None:
            buckets[uid] = bucket
    if not buckets:
        return []
    verdicts = await collect_audit_verdicts(
        storage, chain, window, buckets, verify_signatures=verify_signatures
    )
    at_window = window if apply_window is None else apply_window
    issued: list[SlashRecord] = []
    for miner in sorted(verdicts):
        record = ledger.audit_verdicts(miner, at_window, list(verdicts[miner]))
        if record is not None:
            issued.append(record)
            log.warning(
                "audit quorum slash",
                miner=miner,
                audited_window=window,
                apply_window=at_window,
                detail=record.detail,
            )
    return issued
