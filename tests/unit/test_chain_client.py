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

CREDS = BucketCreds(
    account_id="0123456789abcdef0123456789abcdef",
    bucket_name="mok-miner-7",
    access_key_id="fedcba9876543210fedcba9876543210",
    secret_access_key="s3cr3t-key-for-golden-vector-test-0000000000000000000000000000",
)


def window_commit(window: int, fill: str = "ab") -> WindowCommit:
    return WindowCommit(
        window=window, payload_hash=fill * 32, state_root=fill * 32, theta_end_hash=fill * 32
    )


def make_client(**kwargs: Any) -> tuple[ChainClient, MagicMock, MagicMock, MagicMock]:
    cfg = ChainConfig(network="test", netuid=NETUID, commit_retries=3)
    subtensor = MagicMock(name="subtensor")
    subtensor.metagraph.return_value = SimpleNamespace(
        uids=[0, 1, 2],
        hotkeys=["hk0", "hk1", "hk2"],
        S=[10.0, 5.0, 0.5],
    )
    factory = MagicMock(name="subtensor_factory", return_value=subtensor)
    wallet = MagicMock(name="wallet")
    wallet.hotkey.ss58_address = "hk0"
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
            ("hk0", SimpleNamespace(value=pallet_value)),
            ("hk1", raw),                                     # plain-string value
            ("hk-unknown", "wire-for-stranger"),              # hotkey not in metagraph
            ("hk2", SimpleNamespace(value={"info": "mangled"})),  # undecodable value
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
        subtensor.get_commitment.return_value = BucketCommit(creds=CREDS).encode()
        assert client.get_bucket(2) == CREDS
        subtensor.get_commitment.return_value = "garbage!!"
        assert client.get_bucket(2) is None
        subtensor.get_commitment.return_value = window_commit(5).encode()  # wrong kind
        assert client.get_bucket(2) is None

    def test_get_all_buckets_skips_non_bucket(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.substrate.query_map.return_value = [
            ("hk0", BucketCommit(creds=CREDS).encode()),
            ("hk1", window_commit(4).encode()),
            ("hk2", "total garbage"),
        ]
        assert client.get_all_buckets() == {0: CREDS}

    def test_ensure_bucket_committed(self) -> None:
        client, subtensor, _, _ = make_client()
        wire = BucketCommit(creds=CREDS).encode()
        subtensor.get_commitment.return_value = wire
        assert client.ensure_bucket_committed(CREDS) is False
        subtensor.commit.assert_not_called()
        subtensor.get_commitment.return_value = "something-else"
        assert client.ensure_bucket_committed(CREDS) is True
        subtensor.commit.assert_called_once()
        assert subtensor.commit.call_args.args[2] == wire

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
        assert result[0] == window_commit(5, "aa")

    def test_get_window_commits_scans_all_when_uids_none(self) -> None:
        client, subtensor, _, _ = make_client()
        subtensor.substrate.query_map.return_value = [
            ("hk0", window_commit(9, "aa").encode()),
            ("hk1", window_commit(9, "bb").encode()),
            ("hk2", ManifestCommit(manifest_hash="0d" * 32).encode()),
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
            ("hk0", VoteCommit(kind="rollback", target=11, payload_hash="ee" * 32).encode()),
            ("hk1", VoteCommit(kind="amendment", target=11, payload_hash="ee" * 32).encode()),
            ("hk2", "garbage vote"),
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
            wait_for_inclusion=False,
            wait_for_finalization=False,
        )

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
        assert client.hotkeys() == ["hk0", "hk1", "hk2"]
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
        assert client.hotkey_of(1) == "hk1"
        assert client.hotkey_of(99) is None
        assert client.uid_of_hotkey("hk2") == 2
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
