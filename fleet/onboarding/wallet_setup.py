"""Wallet + registration + bucket-credential onboarding.

Everything chain-flavored goes through ``mok_core.chain.ChainClient`` (the
real Bittensor SDK is imported lazily and only when nothing was injected);
everything interactive is gated behind an explicit ``interactive`` flag so a
containerized non-interactive launch NEVER blocks on a hidden prompt — it
fails with instructions instead.

The credentials committed on-chain are the READ pair (``R2_READ_*``): peers
need to fetch this node's payloads, never to write to its bucket. The write
pair stays local to the node (used by ``StorageClient`` for uploads).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from mok_core.config.schemas import BucketCreds, ChainConfig

__all__ = [
    "R2_ENV_VARS",
    "R2_WRITE_ENV_VARS",
    "OnboardingError",
    "WalletError",
    "bucket_creds_from_env",
    "commit_bucket_credentials",
    "ensure_wallet",
    "register",
    "resolve_bucket_name",
    "write_creds_from_env",
]

#: Environment variables holding the on-chain (read-only) bucket credentials.
#: R2_BUCKET_NAME is OPTIONAL: wire v2 derives every participant's bucket name
#: from its hotkey (`bucket_name_for_hotkey`); when set it must match.
R2_ENV_VARS = (
    "R2_ACCOUNT_ID",
    "R2_READ_ACCESS_KEY_ID",
    "R2_READ_SECRET_ACCESS_KEY",
)

#: Environment variables holding this node's private write credentials.
R2_WRITE_ENV_VARS = (
    "R2_ACCOUNT_ID",
    "R2_WRITE_ACCESS_KEY_ID",
    "R2_WRITE_SECRET_ACCESS_KEY",
)


def resolve_bucket_name(hotkey_ss58: str, env: Mapping[str, str] | None = None) -> str:
    """The node's bucket name: derived from its hotkey (wire v2 convention).
    An explicit R2_BUCKET_NAME is accepted only if it matches — a mismatch would
    publish credentials for a bucket peers can never locate."""
    from mok_core.chain.schemas import bucket_name_for_hotkey  # noqa: PLC0415

    env = env if env is not None else os.environ
    derived = bucket_name_for_hotkey(hotkey_ss58)
    given = env.get("R2_BUCKET_NAME", "")
    if given and given != derived:
        raise OnboardingError(
            f"R2_BUCKET_NAME={given!r} but wire v2 requires this hotkey's bucket to be "
            f"named {derived!r} (hotkey lowercased) — rename the bucket or unset R2_BUCKET_NAME"
        )
    return derived


class OnboardingError(RuntimeError):
    pass


class WalletError(OnboardingError):
    pass


def _key_exists(key_file: Any) -> bool:
    """True iff a bittensor keyfile object reports an on-device key."""
    if key_file is None:
        return False
    exists = getattr(key_file, "exists_on_device", None)
    return bool(exists()) if callable(exists) else False


def ensure_wallet(
    cfg: ChainConfig,
    *,
    interactive: bool = False,
    wallet_factory: Callable[[], Any] | None = None,
) -> Any:
    """Return a wallet whose coldkeypub + hotkey exist on disk.

    Missing keys are created only when ``interactive=True`` (bittensor's
    creation flow prompts for a password/mnemonic confirmation); a
    non-interactive call with missing keys raises ``WalletError`` telling the
    operator exactly what to run. ``wallet_factory`` injects a fake in tests.
    """
    if wallet_factory is None:

        def wallet_factory() -> Any:
            import bittensor as bt  # noqa: PLC0415 — heavy, lazy by design

            wallet_cls = getattr(bt, "wallet", None) or bt.Wallet  # SDK >=10 casing
            return wallet_cls(name=cfg.wallet_name, hotkey=cfg.wallet_hotkey)

    wallet = wallet_factory()
    missing = [
        label
        for label, key_file in (
            ("coldkeypub", getattr(wallet, "coldkeypub_file", None)),
            ("hotkey", getattr(wallet, "hotkey_file", None)),
        )
        if not _key_exists(key_file)
    ]
    if not missing:
        return wallet
    if not interactive:
        raise WalletError(
            f"wallet {cfg.wallet_name}/{cfg.wallet_hotkey} is missing {missing} and this is a "
            "non-interactive run. Create it first (`btcli wallet create`) or pass --interactive."
        )
    wallet.create_if_non_existent(coldkey_use_password=True, hotkey_use_password=False)
    still_missing = [
        label
        for label, key_file in (
            ("coldkeypub", getattr(wallet, "coldkeypub_file", None)),
            ("hotkey", getattr(wallet, "hotkey_file", None)),
        )
        if not _key_exists(key_file)
    ]
    if still_missing:
        raise WalletError(f"wallet creation did not produce {still_missing}")
    return wallet


def register(chain: Any) -> int:
    """Ensure this hotkey holds a UID on the subnet; returns the UID.

    Already-registered hotkeys are a no-op (idempotent re-runs). Otherwise
    ``burned_register`` is submitted through the (injectable) subtensor and
    the metagraph re-synced to learn the assigned UID.
    """
    chain.sync_metagraph()
    uid = chain.my_uid()
    if uid is not None:
        return int(uid)
    ok = chain.subtensor.burned_register(
        wallet=chain.wallet,
        netuid=chain.cfg.netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    if not ok:
        raise OnboardingError(
            f"burned_register failed on netuid {chain.cfg.netuid} — check TAO balance and retry"
        )
    chain.sync_metagraph()
    uid = chain.my_uid()
    if uid is None:
        raise OnboardingError("burned_register reported success but the hotkey has no UID")
    return int(uid)


def bucket_creds_from_env(hotkey_ss58: str, env: Mapping[str, str] | None = None) -> BucketCreds:
    """The on-chain READ credential pair from ``R2_*`` (see module docstring);
    the bucket name is derived from `hotkey_ss58` (`resolve_bucket_name`)."""
    env = env if env is not None else os.environ
    missing = [var for var in R2_ENV_VARS if not env.get(var)]
    if missing:
        raise OnboardingError(f"missing R2 credential env vars: {missing} (see .env.example)")
    return BucketCreds(
        account_id=env["R2_ACCOUNT_ID"],
        bucket_name=resolve_bucket_name(hotkey_ss58, env),
        access_key_id=env["R2_READ_ACCESS_KEY_ID"],
        secret_access_key=env["R2_READ_SECRET_ACCESS_KEY"],
    )


def write_creds_from_env(hotkey_ss58: str, env: Mapping[str, str] | None = None) -> BucketCreds:
    """This node's private WRITE pair (uploads via ``StorageClient``) — never
    committed on-chain. Bucket name derived from `hotkey_ss58`."""
    env = env if env is not None else os.environ
    missing = [var for var in R2_WRITE_ENV_VARS if not env.get(var)]
    if missing:
        raise OnboardingError(f"missing R2 credential env vars: {missing} (see .env.example)")
    return BucketCreds(
        account_id=env["R2_ACCOUNT_ID"],
        bucket_name=resolve_bucket_name(hotkey_ss58, env),
        access_key_id=env["R2_WRITE_ACCESS_KEY_ID"],
        secret_access_key=env["R2_WRITE_SECRET_ACCESS_KEY"],
    )


def commit_bucket_credentials(chain: Any, creds: BucketCreds) -> bool:
    """Commit the read pair on-chain iff it is not already committed verbatim.

    Returns True when a commit transaction was submitted, False when the chain
    already held these exact credentials (``ChainClient.ensure_bucket_committed``).
    """
    return bool(chain.ensure_bucket_committed(creds))
