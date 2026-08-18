"""Tests for B/onboarding/wallet_setup.py — fakes only, no bittensor import."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from B.onboarding.wallet_setup import (
    OnboardingError,
    WalletError,
    bucket_creds_from_env,
    commit_bucket_credentials,
    ensure_wallet,
    register,
    resolve_bucket_name,
    write_creds_from_env,
)
from mok_core.config.schemas import BucketCreds, ChainConfig

CFG = ChainConfig(network="test", netuid=11, wallet_name="w", wallet_hotkey="h")

HOTKEY = "5DciMXcKCLk3yC98RR3wrDWWJunJVgboZmnQXvJpu9nqEQ2E"
ENV = {
    "R2_ACCOUNT_ID": "acct",
    # R2_BUCKET_NAME is optional under wire v2 (derived from the hotkey); tests set
    # it to the derived value to exercise the match check.
    "R2_BUCKET_NAME": HOTKEY.lower(),
    "R2_READ_ACCESS_KEY_ID": "read-key",
    "R2_READ_SECRET_ACCESS_KEY": "read-secret",
    "R2_WRITE_ACCESS_KEY_ID": "write-key",
    "R2_WRITE_SECRET_ACCESS_KEY": "write-secret",
}


class FakeKeyFile:
    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists_on_device(self) -> bool:
        return self._exists


class FakeWallet:
    def __init__(self, cold: bool, hot: bool) -> None:
        self.coldkeypub_file = FakeKeyFile(cold)
        self.hotkey_file = FakeKeyFile(hot)
        self.created = False

    def create_if_non_existent(self, **_kw: Any) -> None:
        self.created = True
        self.coldkeypub_file = FakeKeyFile(True)
        self.hotkey_file = FakeKeyFile(True)


def test_ensure_wallet_returns_existing() -> None:
    wallet = FakeWallet(True, True)
    got = ensure_wallet(CFG, wallet_factory=lambda: wallet)
    assert got is wallet
    assert not wallet.created


def test_ensure_wallet_non_interactive_missing_raises_with_instructions() -> None:
    wallet = FakeWallet(True, False)
    with pytest.raises(WalletError, match="hotkey.*non-interactive|non-interactive"):
        ensure_wallet(CFG, wallet_factory=lambda: wallet)
    assert not wallet.created  # never silently creates


def test_ensure_wallet_interactive_creates() -> None:
    wallet = FakeWallet(False, False)
    got = ensure_wallet(CFG, interactive=True, wallet_factory=lambda: wallet)
    assert got is wallet
    assert wallet.created


def test_ensure_wallet_interactive_creation_failure_raises() -> None:
    wallet = FakeWallet(False, False)
    wallet.create_if_non_existent = lambda **_kw: None  # creation silently fails
    with pytest.raises(WalletError, match="did not produce"):
        ensure_wallet(CFG, interactive=True, wallet_factory=lambda: wallet)


def test_ensure_wallet_never_imports_bittensor_with_factory() -> None:
    before = "bittensor" in sys.modules
    ensure_wallet(CFG, wallet_factory=lambda: FakeWallet(True, True))
    assert ("bittensor" in sys.modules) == before


# --------------------------------------------------------------------------- #
# register
# --------------------------------------------------------------------------- #


def _chain(uid_sequence: list[int | None], register_ok: bool = True) -> MagicMock:
    chain = MagicMock()
    chain.cfg = CFG
    chain.my_uid.side_effect = uid_sequence
    chain.subtensor.burned_register.return_value = register_ok
    return chain


def test_register_short_circuits_when_already_registered() -> None:
    chain = _chain([7])
    assert register(chain) == 7
    chain.subtensor.burned_register.assert_not_called()


def test_register_submits_and_returns_new_uid() -> None:
    chain = _chain([None, 42])
    assert register(chain) == 42
    kwargs = chain.subtensor.burned_register.call_args.kwargs
    assert kwargs["netuid"] == 11
    assert kwargs["wallet"] is chain.wallet
    assert chain.sync_metagraph.call_count == 2


def test_register_failure_raises() -> None:
    with pytest.raises(OnboardingError, match="burned_register failed"):
        register(_chain([None], register_ok=False))


def test_register_success_without_uid_raises() -> None:
    with pytest.raises(OnboardingError, match="no UID"):
        register(_chain([None, None]))


# --------------------------------------------------------------------------- #
# credentials
# --------------------------------------------------------------------------- #


def test_bucket_creds_from_env_reads_the_read_pair() -> None:
    creds = bucket_creds_from_env(HOTKEY, ENV)
    assert creds == BucketCreds(
        account_id="acct",
        bucket_name=HOTKEY.lower(),          # wire v2: derived from the hotkey
        access_key_id="read-key",
        secret_access_key="read-secret",
    )


def test_write_creds_from_env_reads_the_write_pair() -> None:
    creds = write_creds_from_env(HOTKEY, ENV)
    assert creds.access_key_id == "write-key"
    assert creds.secret_access_key == "write-secret"
    assert creds.bucket_name == HOTKEY.lower()


def test_bucket_name_optional_and_verified() -> None:
    env = {k: v for k, v in ENV.items() if k != "R2_BUCKET_NAME"}
    assert resolve_bucket_name(HOTKEY, env) == HOTKEY.lower()          # unset -> derived
    assert bucket_creds_from_env(HOTKEY, env).bucket_name == HOTKEY.lower()
    with pytest.raises(OnboardingError, match="R2_BUCKET_NAME"):
        resolve_bucket_name(HOTKEY, {**env, "R2_BUCKET_NAME": "mok-miner"})   # mismatch -> refuse


@pytest.mark.parametrize("missing", ["R2_ACCOUNT_ID", "R2_READ_SECRET_ACCESS_KEY"])
def test_bucket_creds_missing_env_raises_naming_the_var(missing: str) -> None:
    env = {k: v for k, v in ENV.items() if k != missing}
    with pytest.raises(OnboardingError, match=missing):
        bucket_creds_from_env(HOTKEY, env)


def test_commit_bucket_credentials_delegates() -> None:
    chain = MagicMock()
    chain.ensure_bucket_committed.return_value = True
    creds = bucket_creds_from_env(HOTKEY, ENV)
    assert commit_bucket_credentials(chain, creds) is True
    chain.ensure_bucket_committed.assert_called_once_with(creds)
