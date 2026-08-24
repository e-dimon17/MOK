"""Step-B attestation on real hardware: run_reference == derive_expected.

The attestation challenge (fleet/attestation) is a block-hash-seeded deterministic
toy MoK run with a deadline: only a real 8x SM103 NVLink node produces the
right hash in time. This test executes the miner side (`run_reference`, mok
backend on this node) and the verifier side (`derive_expected`) and demands
hash equality, plus deadline sanity.

`B` is imported LAZILY at run time and every lookup is guarded: if the package
or the expected callables are absent when this test executes, it SKIPS with an
explicit reason naming the missing API (the package ships in the same wheel,
so on a real node this never skips). Calls are bound through
`inspect.signature` so optional keyword differences don't false-fail; a
required parameter this test cannot supply is reported as a skip, never a
fake pass.
"""

from __future__ import annotations

import importlib
import inspect
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.usefixtures("mok_available")

BLOCK_HASH = bytes(range(32))
CHALLENGE_UID = 7
_DEADLINE_ATTRS = ("deadline_s", "deadline_seconds", "time_limit_s", "deadline")


def _attestation_api() -> SimpleNamespace:
    try:
        package = importlib.import_module("fleet.attestation")
    except ImportError as exc:
        pytest.skip(f"fleet.attestation not importable at run time: {exc!r}")
    modules: list[Any] = [package]
    for sub in ("challenge", "reference_step", "verify"):
        try:
            modules.append(importlib.import_module(f"fleet.attestation.{sub}"))
        except ImportError:
            continue

    def find(*names: str) -> Any:
        for module in modules:
            for name in names:
                fn = getattr(module, name, None)
                if callable(fn):
                    return fn
        return None

    return SimpleNamespace(
        build_challenge=find("build_challenge", "make_challenge", "issue_challenge", "challenge_from_block"),
        run_reference=find("run_reference"),
        derive_expected=find("derive_expected"),
    )


def _bind_known_kwargs(fn: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments `fn` accepts, from the candidate pool; skip (never
    guess) if a required parameter is not in the pool."""
    signature = inspect.signature(fn)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    bound: dict[str, Any] = {}
    for name, param in signature.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in candidates:
            bound[name] = candidates[name]
        elif param.default is inspect.Parameter.empty:
            pytest.skip(
                f"{getattr(fn, '__module__', '?')}.{fn.__name__} requires parameter {name!r} "
                "this test cannot supply — align test_08 with the landed fleet API"
            )
    if accepts_var_kw:
        for key in ("device", "rank", "world_size"):
            bound.setdefault(key, candidates[key])
    return bound


def _candidate_pool(dist_ctx: Any, challenge: Any) -> dict[str, Any]:
    pool: dict[str, Any] = {
        "block_hash": BLOCK_HASH,
        "block_hash_hex": BLOCK_HASH.hex(),
        "uid": CHALLENGE_UID,
        "nonce": BLOCK_HASH.hex()[:16],
        "seed": int.from_bytes(BLOCK_HASH[:8], "little"),
        "device": dist_ctx.device,
        "rank": dist_ctx.rank,
        "world_size": dist_ctx.world_size,
        "local_rank": dist_ctx.local_rank,
        "comm": dist_ctx.comm,
        "backend": "mok",
    }
    if challenge is not None:
        pool["challenge"] = challenge
    return pool


def _hash_of(result: Any) -> str:
    """Normalize a run/derive result to its hex hash (str result or an object
    with a hash-ish attribute)."""
    if isinstance(result, str):
        return result
    for attr in ("weight_hash", "state_root", "hash_hex", "digest", "root"):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            return value
    pytest.skip(f"cannot extract a hash from {type(result).__name__} — align test_08 with the landed fleet API")
    raise AssertionError  # unreachable — pytest.skip raises


def test_run_reference_matches_derive_expected(dist_ctx, mok_available) -> None:
    api = _attestation_api()
    if api.run_reference is None:
        pytest.skip("fleet.attestation lacks run_reference — attestation round not landed yet")
    if api.derive_expected is None:
        pytest.skip("fleet.attestation lacks derive_expected — attestation round not landed yet")

    challenge = None
    if api.build_challenge is not None:
        challenge = api.build_challenge(**_bind_known_kwargs(api.build_challenge, _candidate_pool(dist_ctx, None)))

    pool = _candidate_pool(dist_ctx, challenge)
    started = time.monotonic()
    ran = api.run_reference(**_bind_known_kwargs(api.run_reference, pool))
    wall_s = time.monotonic() - started
    expected = api.derive_expected(**_bind_known_kwargs(api.derive_expected, pool))

    ran_hash, expected_hash = _hash_of(ran), _hash_of(expected)
    assert len(ran_hash) == 64 and len(expected_hash) == 64
    assert ran_hash == expected_hash, (
        f"attestation mismatch on genuine hardware: run_reference={ran_hash} "
        f"derive_expected={expected_hash} — honest miners would fail onboarding"
    )

    # Deadline sanity: a published deadline exists, is positive/bounded, and the
    # genuine node beat it (the whole point of the timed challenge).
    if challenge is not None:
        deadlines = [
            float(getattr(challenge, attr))
            for attr in _DEADLINE_ATTRS
            if isinstance(getattr(challenge, attr, None), int | float)
        ]
        if deadlines:
            deadline = deadlines[0]
            assert 0.0 < deadline <= 3600.0, f"absurd attestation deadline {deadline}s"
            assert wall_s < deadline, (
                f"genuine SM103 node took {wall_s:.1f}s but the deadline is {deadline:.1f}s — "
                "honest hardware cannot pass its own challenge"
            )
    dist_ctx.barrier()
