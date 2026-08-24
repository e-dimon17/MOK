"""Window certificate — the leader-signed object that fixes the included-peer set.

Two-phase commit (protocol decision #2): miners commit H(payload) on-chain and
upload bytes inside the gate; the leader validator then publishes this
certificate naming EXACTLY which peers' payloads enter the outer step. Every
node applies the outer optimizer to that set, so bitwise lockstep follows.

Signing is injected (sr25519 in production via the chain layer; any
sign/verify pair in tests) so this module never imports bittensor. The signed
message is the raw 32-byte blake2b-256 canonical hash of the unsigned fields —
see `certificate_message` for the exact wire rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from mok_core.config import canonical_hash
from mok_core.config.schemas import FrozenModel

__all__ = [
    "CommitLike",
    "WindowCertificate",
    "build_certificate",
    "certificate_message",
    "verify_certificate",
]

SignFn = Callable[[bytes], bytes]
VerifyFn = Callable[[bytes, bytes], bool]  # (message, signature) -> ok

# The signed field set, in declaration order. Canonical serialization sorts keys,
# so this order is documentation; the byte layout is fixed by canonical_bytes
# (sorted-key compact JSON) and pinned by a golden-vector test.
UNSIGNED_FIELDS = ("window", "included_uids", "payload_hashes", "theta_start_root", "leader_uid")


def _commit_binds(commit: Any, full_hex64: str) -> bool:
    """True iff `commit` vouches for `full_hex64`. Real WindowCommits (wire v2)
    bind the first 128 bits of H(payload) on-chain; CommitLike doubles carrying a
    full hash compare exactly."""
    binds = getattr(commit, "binds_payload_hash", None)
    if binds is not None:
        return bool(binds(full_hex64))
    return str(commit.payload_hash) == full_hex64


@runtime_checkable
class CommitLike(Protocol):
    """A miner's on-chain window commit as seen by the leader/validators."""

    uid: int
    payload_hash: str      # hex canonical hash of the uploaded WindowPayload
    in_gate: bool          # bytes landed inside the upload gate
    valid: bool            # payload passed structural validation


class WindowCertificate(FrozenModel):
    """The consensus object every node's outer step is keyed on.

    Consensus checks (sorted uids, hash matching, signature) live in
    `verify_certificate`, NOT in model validators, so tampered instances can
    be represented and rejected rather than being unconstructable.
    """

    window: int
    included_uids: tuple[int, ...]     # sorted ascending, no duplicates
    payload_hashes: dict[int, str]     # uid -> hex payload hash, keys == included_uids
    theta_start_root: str              # state_root the window started from
    leader_uid: int
    leader_sig: str = ""               # hex signature over certificate_message(self)


def certificate_message(cert: WindowCertificate) -> bytes:
    """The exact bytes the leader signs: 32-byte blake2b-256 canonical hash.

    Wire rule (consensus constant, SPEC_VERSION-bound): dump the fields in
    UNSIGNED_FIELDS (leader_sig excluded) in pydantic json mode — tuples
    become lists, int dict keys become strings — canonicalize as sorted-key
    compact JSON, blake2b-256, return the raw digest bytes.
    """
    unsigned = cert.model_dump(mode="json", include=set(UNSIGNED_FIELDS))
    return bytes.fromhex(canonical_hash(unsigned))


def build_certificate(
    window: int,
    commits: Mapping[int, CommitLike],
    scores: Mapping[int, float],
    gather_count: int,
    reserve_count: int,
    theta_start_root: str,
    leader_uid: int,
    sign: SignFn,
) -> WindowCertificate:
    """Rank eligible commits and sign the included set.

    Eligible = committed AND in-gate. Ranking: score descending (missing score
    counts as 0.0), ties broken by uid ascending — a total order, so the
    result is independent of dict iteration order. The top `gather_count` are
    taken; structurally invalid ones are dropped and backfilled by promoting
    valid peers from the next `reserve_count` ranked candidates, in rank
    order. Fewer eligible-and-valid peers than gather_count simply yields a
    smaller set.
    """
    if gather_count <= 0:
        raise ValueError(f"gather_count must be positive, got {gather_count}")
    if reserve_count < 0:
        raise ValueError(f"reserve_count must be >= 0, got {reserve_count}")

    eligible = [uid for uid, c in commits.items() if c.in_gate]
    ranked = sorted(eligible, key=lambda uid: (-scores.get(uid, 0.0), uid))
    top = ranked[:gather_count]
    reserve_pool = ranked[gather_count : gather_count + reserve_count]

    included = [uid for uid in top if commits[uid].valid]
    for uid in reserve_pool:
        if len(included) >= gather_count:
            break
        if commits[uid].valid:
            included.append(uid)

    included_uids = tuple(sorted(included))
    unsigned = WindowCertificate(
        window=window,
        included_uids=included_uids,
        payload_hashes={uid: commits[uid].payload_hash for uid in included_uids},
        theta_start_root=theta_start_root,
        leader_uid=leader_uid,
    )
    return unsigned.model_copy(update={"leader_sig": sign(certificate_message(unsigned)).hex()})


def verify_certificate(
    cert: WindowCertificate,
    chain_commits: Mapping[int, CommitLike],
    verify_sig: VerifyFn,
) -> bool:
    """Full consensus check; returns False (never raises) on any defect.

    Rejects: uids not strictly ascending; payload_hashes keys not exactly the
    included set; any included uid missing on-chain or with a mismatched
    payload hash; malformed or invalid leader signature.
    """
    uids = cert.included_uids
    if list(uids) != sorted(set(uids)):
        return False
    if set(cert.payload_hashes) != set(uids):
        return False
    for uid in uids:
        commit = chain_commits.get(uid)
        if commit is None or not _commit_binds(commit, cert.payload_hashes[uid]):
            return False
    try:
        sig = bytes.fromhex(cert.leader_sig)
    except ValueError:
        return False
    return bool(verify_sig(certificate_message(cert), sig))
