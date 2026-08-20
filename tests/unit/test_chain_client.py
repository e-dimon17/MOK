"""ChainClient behavior against MagicMock subtensor/wallet doubles — the real
bittensor SDK is never imported (mok_core/chain/client.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from mok_core.chain.client import U16_MAX, ChainClient, ChainError, normalize_weights_u16
from mok_core.chain.schemas import BucketCommit, ManifestCommit, VoteCommit, WindowCommit
from mok_core.config.schemas import BucketCreds, ChainConfig

NETUID = 7

HK = [
    "5FmoaEjSmmyvVqXbDqJAhDFeJpao517beWxBieMroU6VrnXu",  # uid 0
    "5EF8uRMaLNqsGMy1wGx1cceqJVcFBtqR8egmqTfDnvFuAW5Y",  # uid 1
    "5DciMXcKCLk3yC98RR3wrDWWJunJVgboZmnQXvJpu9nqEQ2E",  # uid 2
]
CREDS = BucketCreds(
    account_id="0123456789abcdef0123456789abcdef",
    bucket_name=HK[2].lower(),   # v2: derived from the committing hotkey
    access_key_id="fedcba9876543210fedcba9876543210",
    secret_access_key="00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
)


def creds_for(uid: int, **update: str) -> BucketCreds:
    """CREDS as they decode for `uid` (bucket name = its hotkey lowercased)."""
    return CREDS.model_copy(update={"bucket_name": HK[uid].lower(), **update})


def window_commit(window: int, fill: str = "ab") -> WindowCommit:
    return WindowCommit(
        window=window, payload_hash=fill * 32, state_root=fill * 32, theta_end_hash=fill * 32
    )


def make_client(**kwargs: Any) -> tuple[ChainClient, MagicMock, MagicMock, MagicMock]:
    cfg = ChainConfig(network="test", netuid=NETUID, commit_retries=3)
    subtensor = MagicMock(name="subtensor")
    subtensor.metagraph.return_value = SimpleNamespace(
        uids=[0, 1, 2],
        hotkeys=list(HK),
        S=[10.0, 5.0, 0.5],
    )
    factory = MagicMock(name="subtensor_factory", return_value=subtensor)
    wallet = MagicMock(name="wallet")
    wallet.hotkey.ss58_address = HK[0]
    wallet.hotkey.sign.return_value = b"sig-bytes"
    client = ChainClient(
        cfg, wallet=wallet, subtensor_factory=factory, backoff_base_s=0.0, **kwargs
    )
    return client, subtensor, wallet, factory


class TestLaziness:
    def test_no_chain_interaction_in_init(self) -> None:
        client, subtensor, _, factory = make_client()
        assert factory.call_count == 0
        assert subtensor.method_calls == []
        _ = client.subtensor
        assert factory.call_count == 1
        _ = client.subtensor
        assert factory.call_count == 1  # cached handle

    def test_metagraph_lazy_and_cached(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.metagraph.assert_not_called()
        _ = client.metagraph
        _ = client.metagraph
        subtensor.metagraph.assert_called_once_with(NETUID)


class TestCommit:
    def test_commit_exact_args(self) -> None:
        client, subtensor, wallet, _ = make_client()
        client.commit("payload-string")
        subtensor.commit.assert_called_once_with(wallet, NETUID, "payload-string")

    def test_retry_first_two_fail_third_succeeds(self) -> None:
        client, subtensor, wallet, _ = make_client()
        subtensor.commit.side_effect = [RuntimeError("ws drop"), TimeoutError("slow"), None]
        client.commit("retry-me")
        assert subtensor.commit.call_count == 3
        for call in subtensor.commit.call_args_list:
            assert call.args == (wallet, NETUID, "retry-me")

    def test_retries_exhausted_raises_chain_error(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.commit.side_effect = RuntimeError("always down")
        with pytest.raises(ChainError, match="3 attempts"):
            client.commit("doomed")
        assert subtensor.commit.call_count == 3

    def test_backoff_sleeps_exponentially(self) -> None:
        client, subtensor, _, _ = make_client()
        client._backoff_base_s = 1.0
        subtensor.commit.side_effect = [RuntimeError("a"), RuntimeError("b"), None]
        with patch("mok_core.chain.client.time.sleep") as sleep:
            client.commit("x")
        assert [c.args[0] for c in sleep.call_args_list] == [1.0, 2.0]


class TestRawCommitments:
    def test_get_commitment(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.get_commitment.return_value = "some-wire"
        assert client.get_commitment(3) == "some-wire"
        subtensor.get_commitment.assert_called_once_with(NETUID, 3)

    def test_get_commitment_empty_or_error_is_none(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.get_commitment.return_value = ""
        assert client.get_commitment(1) is None
        subtensor.get_commitment.side_effect = RuntimeError("rpc error")
        assert client.get_commitment(1) is None

    def test_get_all_commitments_decodes_and_skips_garbage(self) -> None:
        client, subtensor, _, _ = make_client()
        raw = "wire-for-hk1"
        pallet_value = {  # substrate commitment pallet shape: info.fields[0][0] = {"Raw…": (bytes,)}
            "info": {"fields": [[{"Raw12": (tuple(b"wire-for-hk0"),)}]]}
        }
        subtensor.substrate.query_map.return_value = [
            (HK[0], SimpleNamespace(value=pallet_value)),
            (HK[1], raw),                                     # plain-string value
            ("hk-unknown", "wire-for-stranger"),              # hotkey not in metagraph
            (HK[2], SimpleNamespace(value={"info": "mangled"})),  # undecodable value
        ]
        result = client.get_all_commitments()
        assert result == {0: "wire-for-hk0", 1: "wire-for-hk1"}
        subtensor.substrate.query_map.assert_called_once_with(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[NETUID],
            block_hash=None,
        )

    def test_get_all_commitments_at_block(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.substrate.get_block_hash.return_value = "0xfeed"
        subtensor.substrate.query_map.return_value = []
        client.get_all_commitments(block=123)
        subtensor.substrate.get_block_hash.assert_called_once_with(123)
        assert subtensor.substrate.query_map.call_args.kwargs["block_hash"] == "0xfeed"


class TestTypedCommitments:
    def test_commit_bucket_wire(self) -> None:
        client, subtensor, wallet, _ = make_client()
        client.commit_bucket(CREDS)
        subtensor.commit.assert_called_once_with(wallet, NETUID, BucketCommit(creds=CREDS).encode())

    def test_get_bucket_round_trip_and_garbage(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.substrate.query_map.return_value = []      # no history in this double
        subtensor.get_commitment.return_value = "garbage!!"
        assert client.get_bucket(2) is None                  # never seen a bucket -> None
        subtensor.get_commitment.return_value = BucketCommit(creds=CREDS).encode()
        assert client.get_bucket(2) == creds_for(2)
        # The slot moving on to a WindowCommit does NOT lose the bucket (single-slot model).
        subtensor.get_commitment.return_value = window_commit(5).encode()
        assert client.get_bucket(2) == creds_for(2)

    def test_get_all_buckets_skips_non_bucket(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.substrate.query_map.return_value = [
            (HK[0], BucketCommit(creds=CREDS).encode()),
            (HK[1], window_commit(4).encode()),
            (HK[2], "total garbage"),
        ]
        assert client.get_all_buckets() == {0: creds_for(0)}

    def test_ensure_bucket_committed(self) -> None:
        client, subtensor, _, _ = make_client()
        wire = BucketCommit(creds=CREDS).encode()
        subtensor.get_commitment.return_value = wire
        assert client.ensure_bucket_committed(CREDS) is False
        subtensor.commit.assert_not_called()
        other = BucketCommit(creds=CREDS.model_copy(update={"access_key_id": "9" * 32})).encode()
        subtensor.get_commitment.return_value = other      # rotated key -> re-commit
        assert client.ensure_bucket_committed(CREDS) is True
        subtensor.commit.assert_called_once()
        assert subtensor.commit.call_args.args[2] == wire
        subtensor.get_commitment.return_value = window_commit(3).encode()   # role commitment
        with pytest.raises(ChainError, match="non-bucket commitment"):
            client.ensure_bucket_committed(CREDS)          # never clobbered

    def test_commit_and_get_window_commits_filtered(self) -> None:
        client, subtensor, wallet, _ = make_client()
        wc = window_commit(5)
        client.commit_window(wc)
        subtensor.commit.assert_called_once_with(wallet, NETUID, wc.encode())

        subtensor.get_commitment.side_effect = [
            window_commit(5, "aa").encode(),   # uid 0: right window
            window_commit(4, "bb").encode(),   # uid 1: stale window -> filtered
            "not-a-commitment",                # uid 2: garbage -> skipped
        ]
        result = client.get_window_commits(5, uids=[0, 1, 2])
        assert set(result) == {0}
        want = window_commit(5, "aa")
        got = result[0]
        assert (got.window, got.state_root, got.theta_end_hash) == (want.window, want.state_root, want.theta_end_hash)
        assert got.binds_payload_hash(want.payload_hash)   # 128-bit prefix bound on-chain

    def test_get_window_commits_scans_all_when_uids_none(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.substrate.query_map.return_value = [
            (HK[0], window_commit(9, "aa").encode()),
            (HK[1], window_commit(9, "bb").encode()),
            (HK[2], ManifestCommit(manifest_hash="0d" * 32).encode()),
        ]
        result = client.get_window_commits(9)
        assert set(result) == {0, 1}
        assert result[1].state_root == "bb" * 32

    def test_manifest_hash_round_trip(self) -> None:
        client, subtensor, wallet, _ = make_client()
        client.commit_manifest_hash("0d" * 32)
        expected = ManifestCommit(manifest_hash="0d" * 32).encode()
        subtensor.commit.assert_called_once_with(wallet, NETUID, expected)

        subtensor.get_commitment.return_value = expected
        assert client.get_manifest_hash(0) == "0d" * 32
        subtensor.get_commitment.return_value = "junk"
        assert client.get_manifest_hash(0) is None

    def test_votes_filtering(self) -> None:
        client, subtensor, wallet, _ = make_client()
        vote = VoteCommit(kind="rollback", target=11, payload_hash="ee" * 32)
        client.commit_vote(vote)
        subtensor.commit.assert_called_once_with(wallet, NETUID, vote.encode())

        subtensor.substrate.query_map.return_value = [
            (HK[0], VoteCommit(kind="rollback", target=11, payload_hash="ee" * 32).encode()),
            (HK[1], VoteCommit(kind="amendment", target=11, payload_hash="ee" * 32).encode()),
            (HK[2], "garbage vote"),
        ]
        assert set(client.get_votes()) == {0, 1}
        assert set(client.get_votes(kind="rollback")) == {0}
        assert set(client.get_votes(kind="amendment", target=11)) == {1}
        assert client.get_votes(target=99) == {}


class TestSetWeights:
    def test_u16_normalized_exact_call(self) -> None:
        client, subtensor, wallet, _ = make_client()
        subtensor.set_weights.return_value = (True, "ok")
        assert client.set_weights({1: 0.5, 2: 1.0, 0: 0.0}) is True
        subtensor.set_weights.assert_called_once_with(
            wallet=wallet,
            netuid=NETUID,
            uids=[1, 2],
            weights=[32768, 65535],
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

    def test_extrinsic_response_object_success_is_honored(self) -> None:
        # bittensor >= 10 returns an ExtrinsicResponse dataclass; its truthiness
        # is ALWAYS True, so success must be read from the attribute (testnet 534
        # finding: rate-limited "no attempt" responses were logged as submitted).
        class _Resp:
            def __init__(self, success: bool, message: str) -> None:
                self.success = success
                self.message = message

        client, subtensor, _, _ = make_client()
        subtensor.set_weights.return_value = _Resp(False, "No attempt made. Perhaps it is too soon to set weights!")
        assert client.set_weights({1: 1.0}) is False
        subtensor.set_weights.return_value = _Resp(True, "Success")
        assert client.set_weights({1: 1.0}) is True

    def test_none_result_is_failure(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.set_weights.return_value = None
        assert client.set_weights({1: 1.0}) is False

    def test_rejection_returns_false(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.set_weights.return_value = (False, "rate limited")
        assert client.set_weights({1: 1.0}) is False

    def test_bool_result_supported(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.set_weights.return_value = True
        assert client.set_weights({1: 1.0}) is True

    def test_empty_weights_no_chain_call(self) -> None:
        client, subtensor, _, _ = make_client()
        assert client.set_weights({}) is False
        assert client.set_weights({1: 0.0, 2: -3.0}) is False
        subtensor.set_weights.assert_not_called()


class TestNormalizeWeightsU16:
    def test_vectors(self) -> None:
        assert normalize_weights_u16({1: 0.5, 2: 1.0}) == ([1, 2], [32768, 65535])
        assert normalize_weights_u16({2: 1.0, 1: 0.25}) == ([1, 2], [16384, 65535])
        assert normalize_weights_u16({5: 2.0}) == ([5], [U16_MAX])
        assert normalize_weights_u16({3: 4.0, 9: 4.0}) == ([3, 9], [U16_MAX, U16_MAX])

    def test_drops_nonpositive_and_nonfinite(self) -> None:
        assert normalize_weights_u16({}) == ([], [])
        assert normalize_weights_u16({1: 0.0, 2: -1.0}) == ([], [])
        assert normalize_weights_u16({1: float("nan"), 2: 1.0}) == ([2], [U16_MAX])
        assert normalize_weights_u16({1: float("inf"), 2: 1.0}) == ([2], [U16_MAX])

    def test_tiny_weight_rounds_to_zero_and_is_dropped(self) -> None:
        assert normalize_weights_u16({7: 1e-9, 8: 1.0}) == ([8], [U16_MAX])

    def test_scale_invariance(self) -> None:
        base = normalize_weights_u16({1: 0.2, 2: 0.6, 3: 1.0})
        scaled = normalize_weights_u16({1: 20.0, 2: 60.0, 3: 100.0})
        assert base == scaled == ([1, 2, 3], [13107, 39321, 65535])


class TestBlocks:
    def test_current_block_cached_within_block_time(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.get_current_block.side_effect = [100, 101]
        assert client.current_block() == 100
        assert client.current_block() == 100          # served from cache
        assert subtensor.get_current_block.call_count == 1
        assert client.current_block(force=True) == 101
        assert subtensor.get_current_block.call_count == 2

    def test_current_block_cache_expires(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.get_current_block.side_effect = [100, 105]
        with patch("mok_core.chain.client.time.monotonic", side_effect=[0.0, 5.0, 20.0]):
            assert client.current_block() == 100      # t=0: fetch
            assert client.current_block() == 100      # t=5 < block_time 12: cached
            assert client.current_block() == 105      # t=20: stale -> refetch
        assert subtensor.get_current_block.call_count == 2

    def test_block_hash_bytes(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.get_block_hash.return_value = "0x" + "ab" * 32
        assert client.block_hash(77) == bytes.fromhex("ab" * 32)
        subtensor.get_block_hash.assert_called_once_with(77)
        subtensor.get_block_hash.return_value = b"\x01" * 32   # bytes passthrough
        assert client.block_hash(78) == b"\x01" * 32

    def test_block_timestamp_via_substrate(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.get_block_hash.return_value = "0xdeadbeef"
        subtensor.substrate.query.return_value = SimpleNamespace(value=1_722_945_600_123)
        assert client.block_timestamp(500) == pytest.approx(1_722_945_600.123)
        subtensor.substrate.query.assert_called_once_with(
            module="Timestamp", storage_function="Now", block_hash="0xdeadbeef"
        )

    def test_current_window(self) -> None:
        client, subtensor, _, _ = make_client()
        schedule = SimpleNamespace(start_block=100, blocks_per_window=10)
        subtensor.get_current_block.return_value = 125
        assert client.current_window(schedule) == 2

    async def test_wait_for_window(self) -> None:
        client, subtensor, _, _ = make_client()
        schedule = SimpleNamespace(start_block=100, blocks_per_window=10)
        subtensor.get_current_block.side_effect = [99, 105, 121]
        reached = await client.wait_for_window(2, schedule, poll_s=0.0)
        assert reached == 2
        assert subtensor.get_current_block.call_count == 3


class TestMetagraphAccessors:
    def test_lists(self) -> None:
        client, _, _, _ = make_client()
        assert client.uids() == [0, 1, 2]
        assert client.hotkeys() == HK
        assert client.stakes() == {0: 10.0, 1: 5.0, 2: 0.5}

    def test_array_like_metagraph(self) -> None:
        import numpy as np

        client, subtensor, _, _ = make_client()
        subtensor.metagraph.return_value = SimpleNamespace(
            uids=np.array([3, 4]), hotkeys=["hk3", "hk4"], S=np.array([1.5, 2.5])
        )
        assert client.uids() == [3, 4]
        assert client.stakes() == {3: 1.5, 4: 2.5}
        assert client.hotkey_of(4) == "hk4"

    def test_hotkey_lookups(self) -> None:
        client, _, _, _ = make_client()
        assert client.hotkey_of(1) == HK[1]
        assert client.hotkey_of(99) is None
        assert client.uid_of_hotkey(HK[2]) == 2
        assert client.uid_of_hotkey("nope") is None
        assert client.my_uid() == 0   # wallet hotkey is hk0


class TestSignatures:
    def test_sign_uses_wallet_hotkey(self) -> None:
        client, _, wallet, _ = make_client()
        assert client.sign(b"message") == b"sig-bytes"
        wallet.hotkey.sign.assert_called_once_with(b"message")

    def test_verify_via_injected_keypair_factory(self) -> None:
        keypair = MagicMock()
        keypair.verify.return_value = True
        factory = MagicMock(return_value=keypair)
        client, _, _, _ = make_client(keypair_factory=factory)
        assert client.verify("hk9", b"data", b"sig") is True
        factory.assert_called_once_with("hk9")
        keypair.verify.assert_called_once_with(b"data", b"sig")

    def test_verify_false_on_bad_sig_or_error(self) -> None:
        keypair = MagicMock()
        keypair.verify.return_value = False
        client, _, _, _ = make_client(keypair_factory=lambda ss58: keypair)
        assert client.verify("hk9", b"data", b"bad") is False
        keypair.verify.side_effect = ValueError("malformed signature")
        assert client.verify("hk9", b"data", b"junk") is False


# --------------------------------------------------------------------------- #
# History-aware bucket discovery (single commitment slot per hotkey)
# --------------------------------------------------------------------------- #


class _SlotHistoryChain:
    """Subtensor double modelling the ONE-slot-per-hotkey commitment pallet with
    full history: `history[block] = {hotkey: wire}` snapshots; the live slot is
    the newest snapshot. `get_block_hash(b)` returns f"0x{b}" and query_map
    replays the snapshot at that block."""

    def __init__(self, history: dict[int, dict[str, str]], head: int) -> None:
        self.history = dict(history)
        self.head = head
        self.reads: list[int] = []
        self.substrate = SimpleNamespace(
            get_block_hash=lambda b: f"0x{b}",
            query_map=self._query_map,
        )

    def _snapshot_at(self, block: int) -> dict[str, tuple[str, int]]:
        """{hotkey: (wire, commit_block)} — per hotkey, the newest commit at or before `block`."""
        state: dict[str, tuple[str, int]] = {}
        for b in sorted(b for b in self.history if b <= block):
            for hk, wire in self.history[b].items():
                state[hk] = (wire, b)
        return state

    def _query_map(self, *, module: str, storage_function: str, params: list, block_hash: str | None):
        block = self.head if block_hash is None else int(block_hash[2:])
        self.reads.append(block)
        return [
            (hk, {"deposit": 0, "block": cb, "info": {"fields": [{"Raw": "0x" + wire.encode().hex()}]}})
            for hk, (wire, cb) in self._snapshot_at(block).items()
        ]

    # ChainClient surface used by the discovery code
    def metagraph(self, netuid: int):
        return SimpleNamespace(uids=[0, 1, 2], hotkeys=list(HK), S=[10.0, 5.0, 0.5])

    def get_commitment(self, netuid: int, uid: int):
        entry = self._snapshot_at(self.head).get(HK[uid])
        return None if entry is None else entry[0]

    def get_current_block(self) -> int:
        return self.head


def _history_client(history: dict[int, dict[str, str]], head: int) -> tuple[ChainClient, _SlotHistoryChain]:
    cfg = ChainConfig(network="test", netuid=NETUID, commit_retries=1)
    chain = _SlotHistoryChain(history, head)
    wallet = MagicMock(name="wallet")
    wallet.hotkey.ss58_address = HK[0]
    client = ChainClient(cfg, wallet=wallet, subtensor_factory=lambda: chain, backoff_base_s=0.0)
    return client, chain


class TestHistoryAwareBuckets:
    def test_live_bucket_commit_is_used_directly(self) -> None:
        client, chain = _history_client({100: {HK[1]: BucketCommit(creds=CREDS).encode()}}, head=500)
        assert client.get_bucket(1) == creds_for(1)
        assert chain.reads == []  # live slot answered; no history walk

    def test_bucket_recovered_after_window_commits_overwrite_slot(self) -> None:
        # Miner uid 1: onboarded its bucket at block 100, then WindowCommits every 50 blocks.
        history = {100: {HK[1]: BucketCommit(creds=CREDS).encode()}}
        for w, blk in enumerate(range(150, 500, 50)):
            history[blk] = {HK[1]: window_commit(w).encode()}
        client, chain = _history_client(history, head=500)
        assert client.get_commitment(1).startswith("MOKW")   # live slot is a WindowCommit
        assert client.get_bucket(1) == creds_for(1)                  # ...but the bucket is recovered
        assert chain.reads, "history was consulted"
        # Exact hops land on commit_block-1 of each occupant; the bucket is live at 149.
        assert 149 in chain.reads and 500 > min(chain.reads) >= 100
        # Cached: a second lookup does no further reads.
        n = len(chain.reads)
        assert client.get_bucket(1) == creds_for(1)
        assert len(chain.reads) == n

    def test_owner_bucket_survives_manifest_commit(self) -> None:
        history = {
            100: {HK[0]: BucketCommit(creds=CREDS).encode()},
            200: {HK[0]: ManifestCommit(manifest_hash="cd" * 32).encode()},
        }
        client, _ = _history_client(history, head=300)
        assert client.get_manifest_hash(0) == "cd" * 32   # live slot: manifest
        assert client.get_bucket(0) == creds_for(0)               # history: bucket

    def test_get_all_buckets_merges_live_and_recovered(self) -> None:
        creds1 = creds_for(1, access_key_id="1" * 32)
        creds2 = creds_for(2, access_key_id="2" * 32)
        history = {
            100: {HK[1]: BucketCommit(creds=creds1).encode(), HK[2]: BucketCommit(creds=creds2).encode()},
            300: {HK[1]: window_commit(0).encode(), HK[2]: BucketCommit(creds=creds2).encode()},
        }
        client, _ = _history_client(history, head=400)
        assert client.get_all_buckets() == {1: creds1, 2: creds2}   # uid 0 never onboarded -> absent

    def test_never_onboarded_uid_is_absent_and_not_negatively_cached(self) -> None:
        client, chain = _history_client({}, head=50)
        client.bucket_lookback_blocks = 40
        assert client.get_bucket(2) is None
        # Later onboarding becomes visible (no negative cache).
        chain.history[60] = {HK[2]: BucketCommit(creds=CREDS).encode()}
        chain.head = 61
        assert client.get_bucket(2) == creds_for(2)

    def test_lookback_bound_is_respected(self) -> None:
        # Bucket at 10, then a WindowCommit EVERY block up to head: reaching the
        # bucket needs ~head hops, but the walk must stop at head-lookback.
        history = {10: {HK[1]: BucketCommit(creds=CREDS).encode()}}
        for blk in range(11, 5000):
            history[blk] = {HK[1]: window_commit(blk % 7).encode()}
        client, chain = _history_client(history, head=5000)
        client.bucket_lookback_blocks = 1000
        client.bucket_lookback_hops = 10_000
        assert client.get_bucket(1) is None
        assert min(chain.reads) >= 5000 - 1000 - 1     # never read below the floor
        # Raising the lookback past block 10 finds it (no negative cache).
        client.bucket_lookback_blocks = 6000
        assert client.get_bucket(1) == creds_for(1)

    def test_walk_is_exact_hops_not_strides(self) -> None:
        # Bucket lived for ONE block before the first WindowCommit: a stride-based
        # walk would skip it; exact commit-block hops must find it.
        history = {100: {HK[1]: BucketCommit(creds=CREDS).encode()}}
        for w, blk in enumerate(range(101, 2000, 7)):
            history[blk] = {HK[1]: window_commit(w).encode()}
        client, chain = _history_client(history, head=2000)
        assert client.get_bucket(1) == creds_for(1)
        assert 99 in chain.reads or 100 in chain.reads

    def test_ensure_bucket_committed_refuses_to_clobber_role_commitment(self) -> None:
        # uid 0 (our wallet) already holds a ManifestCommit: committing a bucket now would
        # break history-aware discovery for every peer. Must raise, not overwrite.
        history = {100: {HK[0]: ManifestCommit(manifest_hash="cd" * 32).encode()}}
        client, chain = _history_client(history, head=200)
        chain.set_commitment = MagicMock(name="set_commitment")
        with pytest.raises(ChainError, match="non-bucket commitment"):
            client.ensure_bucket_committed(CREDS)
        chain.set_commitment.assert_not_called()

    def test_ensure_bucket_committed_on_empty_slot_commits(self) -> None:
        client, chain = _history_client({}, head=200)
        chain.set_commitment = MagicMock(name="set_commitment")
        assert client.ensure_bucket_committed(CREDS) is True
        chain.set_commitment.assert_called_once()
        assert chain.set_commitment.call_args.args[2] == BucketCommit(creds=CREDS).encode()


class TestReadRetry:
    def test_transient_query_map_failure_is_retried(self) -> None:
        client, subtensor, _, _ = make_client()
        calls = {"n": 0}

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("UnknownBlock: Header was not found in the database")
            return []

        subtensor.substrate.query_map.side_effect = flaky
        assert client.get_all_commitments() == {}     # survives the blip
        assert calls["n"] == 2

    def test_persistent_read_failure_raises_chain_error(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.substrate.query_map.side_effect = RuntimeError("down")
        with pytest.raises(ChainError, match="failed after"):
            client.get_all_commitments()
