"""Bittensor chain client: commitments, weights, blocks, metagraph, signatures.

Commitment read/write, wallet/subtensor bootstrap, weight setting, and
block queries, shaped around this subnet's typed wire objects
(mok_core.chain.schemas) and window arithmetic (windows.py).

The `bittensor` package is imported lazily inside methods only; tests inject
`wallet` / `subtensor_factory` / `keypair_factory` doubles and never touch
the SDK. Construction performs NO chain interaction.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol

from mok_core.config.schemas import BucketCreds, ChainConfig
from mok_core.telemetry.logging import get_logger

from .schemas import TAG_BUCKET, BucketCommit, ManifestCommit, VoteCommit, WindowCommit
from .windows import window_of_block

log = get_logger("chain.client")

U16_MAX = 65535


class ChainError(RuntimeError):
    """A chain interaction failed after exhausting retries."""


class WindowSchedule(Protocol):
    """Anything that pins window 0 to a block (RunManifest satisfies this)."""

    @property
    def start_block(self) -> int: ...

    @property
    def blocks_per_window(self) -> int: ...


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def normalize_weights_u16(weights: Mapping[int, float]) -> tuple[list[int], list[int]]:
    """Bittensor u16 weight normalization (matches the SDK's convert_weights_and_uids_for_emit):
    drop non-positive/non-finite entries, scale so max maps to 65535, round, drop zeros.
    Returns (uids, u16_weights) sorted by uid."""
    positive = {int(u): float(w) for u, w in weights.items() if math.isfinite(w) and w > 0.0}
    if not positive:
        return [], []
    max_w = max(positive.values())
    uids: list[int] = []
    vals: list[int] = []
    for uid in sorted(positive):
        v = round(positive[uid] / max_w * U16_MAX)
        if v > 0:
            uids.append(uid)
            vals.append(v)
    return uids, vals


def _extrinsic_outcome(result: Any) -> tuple[bool, str]:
    """(success, message) from any SDK set_weights return shape.

    bittensor < 10 returned ``(bool, str)`` (or a bare bool); >= 10 returns an
    ``ExtrinsicResponse`` dataclass with ``.success`` / ``.message``. Objects are
    NEVER judged by truthiness: ``bool(ExtrinsicResponse(success=False))`` is
    True, which silently turned every rate-limited / rejected submission into a
    logged "success" (testnet 534 finding, 2026-08-20).
    """
    if isinstance(result, tuple):
        ok = bool(result[0]) if result else False
        msg = str(result[1]) if len(result) > 1 else ""
        return ok, msg
    if isinstance(result, bool):
        return result, ""
    if result is None:
        return False, "no response"
    if hasattr(result, "success"):
        return bool(result.success), str(getattr(result, "message", "") or "")
    return bool(result), str(result)


def _as_list(x: Any) -> list[Any]:
    """Metagraph fields may be list, numpy array, or torch tensor."""
    if hasattr(x, "tolist"):
        return list(x.tolist())
    return list(x)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


class ChainClient:
    """One node's view of the chain. All handles are lazy; injection points
    (`wallet`, `subtensor_factory`, `keypair_factory`) exist for tests and
    for sharing a subtensor across components."""

    #: How far back get_bucket() searches commitment history for a peer's
    #: onboarding BucketCommit (~4 weeks at 12 s/block).
    bucket_lookback_blocks: int = 200_000
    #: Cap on commit-to-commit hops per recovery (a miner makes one per window).
    bucket_lookback_hops: int = 5_000

    def __init__(
        self,
        cfg: ChainConfig,
        *,
        wallet: Any = None,
        subtensor_factory: Callable[[], Any] | None = None,
        keypair_factory: Callable[[str], Any] | None = None,
        backoff_base_s: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self._wallet = wallet
        self._subtensor_factory = subtensor_factory
        self._keypair_factory = keypair_factory
        self._backoff_base_s = backoff_base_s
        self._subtensor: Any = None
        self._metagraph: Any = None
        self._block_cache: tuple[float, int] | None = None
        # uid -> BucketCreds recovered from commitment HISTORY (see get_bucket).
        self._bucket_cache: dict[int, BucketCreds] = {}

    # ------------------------------------------------------------------ #
    # Lazy handles (real SDK only when nothing was injected)
    # ------------------------------------------------------------------ #

    @property
    def subtensor(self) -> Any:
        if self._subtensor is None:
            if self._subtensor_factory is not None:
                self._subtensor = self._subtensor_factory()
            else:
                import bittensor as bt  # noqa: PLC0415 — heavy, lazy by design

                # SDK >=10 dropped the lowercase aliases (bt.subtensor/bt.wallet).
                subtensor_cls = getattr(bt, "subtensor", None) or bt.Subtensor
                self._subtensor = subtensor_cls(network=self.cfg.network)
        return self._subtensor

    @property
    def wallet(self) -> Any:
        if self._wallet is None:
            import bittensor as bt  # noqa: PLC0415 — heavy, lazy by design

            wallet_cls = getattr(bt, "wallet", None) or bt.Wallet
            self._wallet = wallet_cls(name=self.cfg.wallet_name, hotkey=self.cfg.wallet_hotkey)
        return self._wallet

    @property
    def metagraph(self) -> Any:
        if self._metagraph is None:
            self._metagraph = self.subtensor.metagraph(self.cfg.netuid)
        return self._metagraph

    def sync_metagraph(self) -> None:
        self.metagraph.sync(subtensor=self.subtensor)

    # ------------------------------------------------------------------ #
    # Raw commitments
    # ------------------------------------------------------------------ #

    def commit(self, data: str) -> None:
        """Commit `data` under our hotkey, retrying with exponential backoff."""
        retries = max(1, self.cfg.commit_retries)
        last: Exception | None = None
        # SDK >=10 renamed commit() to set_commitment(); both take (wallet, netuid, data).
        commit_fn = getattr(self.subtensor, "commit", None) or self.subtensor.set_commitment
        for attempt in range(retries):
            try:
                commit_fn(self.wallet, self.cfg.netuid, data)
                return
            except Exception as e:  # noqa: BLE001 — SDK raises broadly
                last = e
                log.warning("commit attempt failed", attempt=attempt, error=str(e))
                if attempt + 1 < retries:
                    time.sleep(self._backoff_base_s * (2**attempt))
        raise ChainError(f"commit failed after {retries} attempts") from last

    def get_commitment(self, uid: int) -> str | None:
        """Raw commitment string of `uid`, or None when absent/unreadable.
        (SDK >=10 prints a cosmetic ERROR line when the slot is empty — harmless.)"""
        try:
            value = self.subtensor.get_commitment(self.cfg.netuid, uid)
        except Exception as e:  # noqa: BLE001
            log.warning("get_commitment failed", uid=uid, error=str(e))
            return None
        if not value:
            return None
        return str(value)

    def get_all_commitments(self, block: int | None = None) -> dict[int, str]:
        """All raw commitments keyed by uid, via one substrate query_map."""
        return {uid: wire for uid, (wire, _blk) in self._commitments_with_blocks(block).items()}

    def _read_retry(self, what: str, fn: Callable[[], Any]) -> Any:
        """Retry an idempotent chain READ with backoff; ChainError on exhaustion.

        Public RPC endpoints are load-balanced and transiently flaky (UnknownBlock
        on a just-resolved head, dropped websockets). Reads are safe to repeat;
        one blip must never kill a node."""
        retries = max(1, self.cfg.commit_retries)
        last: Exception | None = None
        for attempt in range(retries):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — SDK raises broadly
                last = e
                log.warning("chain read failed", what=what, attempt=attempt, error=str(e))
                if attempt + 1 < retries:
                    time.sleep(self._backoff_base_s * (2**attempt))
        raise ChainError(f"{what} failed after {retries} attempts") from last

    def _commitments_with_blocks(self, block: int | None = None) -> dict[int, tuple[str, int | None]]:
        """{uid: (wire, block_the_commit_landed_in)} at `block` (None = head). The
        pallet records the commit block alongside each value; None when absent."""

        def _query() -> Any:
            substrate = self.subtensor.substrate
            block_hash = None if block is None else substrate.get_block_hash(block)
            return substrate.query_map(
                module="Commitments",
                storage_function="CommitmentOf",
                params=[self.cfg.netuid],
                block_hash=block_hash,
            )

        query_result = self._read_retry(f"commitments query_map(block={block})", _query)
        mg = self.metagraph
        hotkey_to_uid = {
            hk: int(uid)
            for hk, uid in zip(_as_list(mg.hotkeys), _as_list(mg.uids), strict=False)
        }
        out: dict[int, tuple[str, int | None]] = {}
        for key, value in query_result:
            hotkey = self._decode_account_id(key)
            commitment = self._decode_commitment_value(value)
            if hotkey is None or commitment is None:
                continue
            uid = hotkey_to_uid.get(hotkey)
            if uid is None:
                continue
            out[uid] = (commitment, self._decode_commitment_block(value))
        return out

    @staticmethod
    def _decode_commitment_block(value: Any) -> int | None:
        raw = getattr(value, "value", value)
        try:
            return int(raw["block"])
        except Exception:  # noqa: BLE001 — plain-string doubles carry no block
            return None

    @staticmethod
    def _decode_account_id(key: Any) -> str | None:
        """SS58 of a query_map key. Handles plain strings (tests / pre-decoded
        substrates) and scale-encoded account ids (real chain)."""
        raw = getattr(key, "value", key)
        if isinstance(raw, tuple | list) and len(raw) == 1:
            raw = raw[0]
        if isinstance(raw, str):
            return raw
        try:
            from bittensor.core.chain_data import decode_account_id  # noqa: PLC0415

            return decode_account_id(raw)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _decode_commitment_value(value: Any) -> str | None:
        """Commitment string out of a query_map value. Handles plain strings and
        the pallet's {'info': {'fields': [[{'Raw…': (bytes,)}]]}} shape."""
        raw = getattr(value, "value", value)
        if isinstance(raw, str):
            return raw
        try:
            fields = raw["info"]["fields"][0]
            if isinstance(fields, tuple | list):
                fields = fields[0]
            inner = fields[next(iter(fields))]
            if isinstance(inner, tuple | list) and len(inner) == 1 and not isinstance(inner[0], int):
                inner = inner[0]
            if isinstance(inner, str):  # SDK >=10 delivers hex ('0x…') instead of bytes
                return bytes.fromhex(inner.removeprefix("0x")).decode("utf-8")
            return bytes(inner).decode("utf-8")
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------ #
    # Typed commitments
    # ------------------------------------------------------------------ #

    def commit_bucket(self, creds: BucketCreds) -> None:
        self.commit(BucketCommit(creds=creds).encode())

    def _bucket_of(self, wire: str | None, uid: int) -> BucketCreds | None:
        """Decode a BucketCommit for `uid`; its bucket name derives from the hotkey."""
        if not wire or not wire.startswith(TAG_BUCKET):
            return None
        hotkey = self.hotkey_of(uid)
        if hotkey is None:
            return None
        try:
            return BucketCommit.decode(wire, hotkey_ss58=hotkey).creds
        except ValueError:
            return None

    def get_bucket(self, uid: int) -> BucketCreds | None:
        """Bucket creds for `uid` — from the live slot, or from commitment HISTORY.

        Each hotkey has ONE commitment slot, and roles overwrite it (miners with
        per-window WindowCommits, the owner with the ManifestCommit, auditors with
        their tag). Every participant commits its BucketCommit ONCE at onboarding,
        BEFORE any other commitment, so the creds remain readable at earlier
        blocks; this walks back through history until it finds them (bounded by
        BUCKET_LOOKBACK_BLOCKS) and caches the answer. Trustless: chain data only.
        """
        live = self._bucket_of(self.get_commitment(uid), uid)
        if live is not None:
            self._bucket_cache[uid] = live
            return live
        cached = self._bucket_cache.get(uid)
        if cached is not None:
            return cached
        found = self._recover_buckets_from_history({uid})
        return found.get(uid)

    def get_all_buckets(self) -> dict[int, BucketCreds]:
        """Bucket creds for every registered uid — live slots plus history-recovered
        ones (see get_bucket). Uids that never onboarded a bucket are absent."""
        out: dict[int, BucketCreds] = {}
        wires = self.get_all_commitments()
        for uid, wire in wires.items():
            creds = self._bucket_of(wire, uid)
            if creds is not None:
                out[uid] = creds
                self._bucket_cache[uid] = creds
        missing = {uid for uid in self.uids() if uid not in out}
        for uid in list(missing):
            cached = self._bucket_cache.get(uid)
            if cached is not None:
                out[uid] = cached
                missing.discard(uid)
        if missing:
            out.update(self._recover_buckets_from_history(missing))
        return out

    def _recover_buckets_from_history(self, wanted: set[int]) -> dict[int, BucketCreds]:
        """Find each wanted uid's onboarding BucketCommit in commitment history.

        The pallet stores, with every commitment, the block it landed in. So the
        state just before the current occupant is at ``commit_block - 1`` — an
        EXACT hop to the previous commitment, never skipping one. We follow that
        chain of overwrites backwards per uid until a BucketCommit appears
        (cached forever), the run of history is exhausted, or the walk exceeds
        BUCKET_LOOKBACK_BLOCKS / BUCKET_LOOKBACK_HOPS. Uids not found are absent,
        never negatively cached (they may onboard later).

        Cost: one query_map per hop, hops == commits since onboarding — but every
        node needs this at most once per peer for the peer's lifetime.
        """
        found: dict[int, BucketCreds] = {}
        remaining = set(wanted)
        if not remaining:
            return found
        try:
            head = self.current_block()
            snapshot = self._commitments_with_blocks()
        except Exception as e:  # noqa: BLE001 — history reads are best-effort
            log.warning("history commitment read failed", error=str(e))
            return found
        floor = max(1, head - self.bucket_lookback_blocks)
        # Per uid: the block at which the current occupant of its slot landed.
        cursor: dict[int, int | None] = {}
        for uid in list(remaining):
            wire, blk = snapshot.get(uid, (None, None))
            creds = self._bucket_of(wire, uid)
            if creds is not None:
                found[uid] = creds
                self._bucket_cache[uid] = creds
                remaining.discard(uid)
            elif wire is None:
                remaining.discard(uid)      # nothing ever committed: no history to walk
            else:
                cursor[uid] = blk if blk is not None else head
        hops = 0
        while remaining and hops < self.bucket_lookback_hops:
            hops += 1
            # Walk all uids that share the same "look at block X" in one read.
            targets: dict[int, list[int]] = {}
            for uid in list(remaining):
                at = cursor.get(uid)
                if at is None or at - 1 < floor:
                    remaining.discard(uid)
                    continue
                targets.setdefault(at - 1, []).append(uid)
            if not targets:
                break
            for block, uids in targets.items():
                try:
                    older = self._commitments_with_blocks(block=block)
                except Exception as e:  # noqa: BLE001
                    log.warning("history commitment read failed", block=block, error=str(e))
                    for uid in uids:
                        remaining.discard(uid)
                    continue
                for uid in uids:
                    wire, blk = older.get(uid, (None, None))
                    creds = self._bucket_of(wire, uid)
                    if creds is not None:
                        found[uid] = creds
                        self._bucket_cache[uid] = creds
                        remaining.discard(uid)
                    elif wire is None or blk is None or blk > block:
                        remaining.discard(uid)  # start of history / inconsistent record
                    else:
                        cursor[uid] = blk       # next hop reads blk - 1 (strictly older)
        return found

    def ensure_bucket_committed(self, creds: BucketCreds) -> bool:
        """Commit bucket creds unless the chain already holds them exactly.
        Returns True iff a commit was sent."""
        uid = self.my_uid()
        existing = self.get_commitment(uid) if uid is not None else None
        wire = BucketCommit(creds=creds).encode()
        if existing == wire:
            return False
        if existing and self._bucket_of(existing, uid) is None:
            # The slot already carries a role commitment (WindowCommit / Manifest /
            # auditor tag). Overwriting it would silently desync the fleet — and a
            # bucket committed AFTER other commitments is not what history-aware
            # discovery expects. Bucket onboarding must come first.
            raise ChainError(
                f"uid {uid} commitment slot holds a non-bucket commitment ({existing[:8]!r}...); "
                "commit the bucket BEFORE any other role commitment"
            )
        self.commit(wire)
        return True

    def commit_window(self, commit: WindowCommit) -> None:
        self.commit(commit.encode())

    def get_window_commits(
        self, window: int, uids: Iterable[int] | None = None
    ) -> dict[int, WindowCommit]:
        """Phase-1 window commits for `window`. Garbage/foreign/stale commitments
        are skipped; `uids=None` scans the whole metagraph in one query."""
        raw = self._raw_commitments(uids)
        out: dict[int, WindowCommit] = {}
        for uid, wire in raw.items():
            try:
                wc = WindowCommit.decode(wire)
            except ValueError:
                continue
            if wc.window == window:
                out[uid] = wc
        return out

    def commit_manifest_hash(self, manifest_hash: str) -> None:
        self.commit(ManifestCommit(manifest_hash=manifest_hash).encode())

    def get_manifest_hash(self, owner_uid: int) -> str | None:
        wire = self.get_commitment(owner_uid)
        if wire is None:
            return None
        try:
            return ManifestCommit.decode(wire).manifest_hash
        except ValueError:
            return None

    def commit_vote(self, vote: VoteCommit) -> None:
        self.commit(vote.encode())

    def get_votes(
        self,
        kind: Literal["rollback", "amendment"] | None = None,
        target: int | None = None,
        uids: Iterable[int] | None = None,
    ) -> dict[int, VoteCommit]:
        """Current vote commitments, optionally filtered by kind and/or target."""
        out: dict[int, VoteCommit] = {}
        for uid, wire in self._raw_commitments(uids).items():
            try:
                vote = VoteCommit.decode(wire)
            except ValueError:
                continue
            if (kind is None or vote.kind == kind) and (target is None or vote.target == target):
                out[uid] = vote
        return out

    def _raw_commitments(self, uids: Iterable[int] | None) -> dict[int, str]:
        if uids is None:
            return self.get_all_commitments()
        out: dict[int, str] = {}
        for uid in uids:
            wire = self.get_commitment(uid)
            if wire is not None:
                out[uid] = wire
        return out

    # ------------------------------------------------------------------ #
    # Weights
    # ------------------------------------------------------------------ #

    def set_weights(self, weights: dict[int, float], *, wait_for_inclusion: bool = True) -> bool:
        """Set on-chain weights from a uid -> score map (u16-normalized).

        Returns True only when the SDK reports the extrinsic as SUCCESSFUL; False
        for empty weights, a chain rejection, or an SDK "no attempt" (e.g. the
        weights rate limit has not reopened). The SDK's verdict is logged either
        way. ``wait_for_inclusion`` defaults to True so a True return means the
        extrinsic was actually included in a block — with it off, SDK >= 10
        reports success the moment the extrinsic is *submitted*.
        """
        uids, vals = normalize_weights_u16(weights)
        if not uids:
            log.warning("set_weights called with no positive weights")
            return False
        result = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.cfg.netuid,
            uids=uids,
            weights=vals,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=False,
        )
        success, message = _extrinsic_outcome(result)
        if success:
            log.info("set_weights accepted", uids=uids, included=wait_for_inclusion, message=message)
        else:
            log.warning("set_weights NOT accepted", uids=uids, message=message)
        return success

    # ------------------------------------------------------------------ #
    # Blocks and windows
    # ------------------------------------------------------------------ #

    def current_block(self, *, force: bool = False) -> int:
        """Chain head, cached for ~one block time to avoid hammering the RPC."""
        now = time.monotonic()
        if not force and self._block_cache is not None:
            cached_at, block = self._block_cache
            if now - cached_at < self.cfg.block_time_s:
                return block
        block = int(self._read_retry("get_current_block", self.subtensor.get_current_block))
        self._block_cache = (now, block)
        return block

    def block_hash(self, block: int) -> bytes:
        """32-byte hash of `block` (hex from the SDK, raw bytes out)."""
        h = self._read_retry(f"get_block_hash({block})", lambda: self.subtensor.get_block_hash(block))
        if isinstance(h, bytes):
            return h
        return bytes.fromhex(str(h).removeprefix("0x"))

    def block_timestamp(self, block: int) -> float:
        """Unix seconds of `block` via substrate Timestamp.Now at its hash
        (chain stores milliseconds)."""
        def _q() -> Any:
            block_hash = self.subtensor.get_block_hash(block)
            return self.subtensor.substrate.query(
                module="Timestamp", storage_function="Now", block_hash=block_hash
            )

        result = self._read_retry(f"block_timestamp({block})", _q)
        value = getattr(result, "value", result)
        return float(value) / 1000.0

    def current_window(self, schedule: WindowSchedule) -> int:
        """Window index at the chain head under `schedule` (e.g. the RunManifest)."""
        return window_of_block(self.current_block(), schedule.start_block, schedule.blocks_per_window)

    async def wait_for_window(self, window: int, schedule: WindowSchedule, poll_s: float = 12.0) -> int:
        """Poll until the chain reaches `window`; returns the window actually reached."""
        while True:
            block = self.current_block(force=True)
            if block >= schedule.start_block:
                current = window_of_block(block, schedule.start_block, schedule.blocks_per_window)
                if current >= window:
                    return current
            await asyncio.sleep(poll_s)

    # ------------------------------------------------------------------ #
    # Metagraph accessors
    # ------------------------------------------------------------------ #

    def uids(self) -> list[int]:
        return [int(u) for u in _as_list(self.metagraph.uids)]

    def hotkeys(self) -> list[str]:
        return [str(h) for h in _as_list(self.metagraph.hotkeys)]

    def stakes(self) -> dict[int, float]:
        mg = self.metagraph
        stake = getattr(mg, "S", None)
        if stake is None:
            stake = mg.stake
        return {
            int(u): float(s)
            for u, s in zip(_as_list(mg.uids), _as_list(stake), strict=True)
        }

    def hotkey_of(self, uid: int) -> str | None:
        uids = self.uids()
        if uid not in uids:
            return None
        return self.hotkeys()[uids.index(uid)]

    def uid_of_hotkey(self, hotkey: str) -> int | None:
        hotkeys = self.hotkeys()
        if hotkey not in hotkeys:
            return None
        return self.uids()[hotkeys.index(hotkey)]

    def my_uid(self) -> int | None:
        return self.uid_of_hotkey(self.wallet.hotkey.ss58_address)

    # ------------------------------------------------------------------ #
    # Signatures
    # ------------------------------------------------------------------ #

    def sign(self, data: bytes) -> bytes:
        """Sign with our hotkey (sr25519 via the wallet keypair)."""
        return bytes(self.wallet.hotkey.sign(data))

    def verify(self, hotkey_ss58: str, data: bytes, signature: bytes) -> bool:
        """Verify `signature` over `data` against `hotkey_ss58`. False on any failure."""
        try:
            keypair = (
                self._keypair_factory(hotkey_ss58)
                if self._keypair_factory is not None
                else self._default_keypair(hotkey_ss58)
            )
            return bool(keypair.verify(data, signature))
        except Exception as e:  # noqa: BLE001
            log.warning("signature verification errored", hotkey=hotkey_ss58, error=str(e))
            return False

    @staticmethod
    def _default_keypair(hotkey_ss58: str) -> Any:
        import bittensor as bt  # noqa: PLC0415 — heavy, lazy by design

        return bt.Keypair(ss58_address=hotkey_ss58)
