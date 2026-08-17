"""Step-B console entrypoints (wired in pyproject):

  mok-attest        attest_main()        challenge | respond | verify
  mok-onboard       onboard_main()       preflight → wallet → register → bucket
                                         → fetch init → self-attest (--skip-*)
  mok-init-publish  init_publish_main()  owner-side seed-42 init publication
  mok-calibrate     calibrate_main()     rehearse | sweep | adam-ab

Every heavy dependency (chain, storage, GPUs) sits behind a flag or an
injected object so the parsers and the flow orchestration are CPU-testable;
tests monkeypatch the module-level names this file calls (``run_preflight``,
``ensure_wallet``, ``run_reference``, ...).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from B.attestation.challenge import (
    DEFAULT_DEADLINE_S,
    DEFAULT_INNER_STEPS,
    AttestationChallenge,
    make_challenge,
)
from B.attestation.reference_step import AttestationResponse, run_reference
from B.attestation.verify import Verifier, judge
from B.calibration.adam_ab import DEFAULT_K, DEFAULT_THRESHOLD_NATS, run_adam_ab
from B.calibration.local_harness import local_manifest
from B.calibration.rehearsal import run_calibration_windows
from B.calibration.sweep import (
    DEFAULT_TUNED_PATH,
    SweepPoint,
    emit_tuned_yaml,
    run_sweep,
    select_best,
)
from B.onboarding.init_publish import (
    DEFAULT_INIT_SEED,
    build_and_publish_init,
    fetch_and_verify_init,
)
from B.onboarding.preflight import run_preflight
from B.onboarding.wallet_setup import (
    bucket_creds_from_env,
    commit_bucket_credentials,
    ensure_wallet,
    register,
    write_creds_from_env,
)
from mok_core.config import RunConfig, load_run_config
from mok_core.data import DatasetShardIndex
from mok_core.determinism import enforce_determinism
from mok_core.data.shards import shard_filename
from mok_core.model import MoKTransformer, init_model, reference_config

__all__ = ["attest_main", "calibrate_main", "init_publish_main", "onboard_main"]


def _emit(obj: Any, out: str = "-") -> None:
    payload = json.dumps(obj, sort_keys=True, default=str)
    if out == "-":
        print(payload)
    else:
        Path(out).write_text(payload + "\n", encoding="utf-8")


def _load_config(config: str, overlays: Sequence[str]) -> RunConfig:
    return load_run_config(config, *overlays)


def _add_config_args(p: argparse.ArgumentParser, *, required: bool = True) -> None:
    p.add_argument("--config", required=required, help="base RunConfig YAML (C/configs/base.yaml)")
    p.add_argument(
        "--overlay", action="append", default=[], help="overlay YAML(s), applied in order"
    )


def _build_model(cfg: RunConfig, *, backend: str, seed: int, device: str) -> MoKTransformer:
    model_cfg = reference_config(cfg.model) if backend == "reference" else cfg.model
    return init_model(model_cfg, seed, device=device, backend=backend)


def _make_chain(cfg: RunConfig, wallet: Any = None) -> Any:
    from mok_core.chain import ChainClient  # noqa: PLC0415 — chain layer stays optional for tests

    return ChainClient(cfg.chain, wallet=wallet)


# ===========================================================================
# mok-attest
# ===========================================================================


def build_attest_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mok-attest", description="Hardware attestation tooling.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("challenge", help="derive a challenge from a block hash (or the chain head)")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--block-hash", help="64-hex block hash to derive from")
    src.add_argument("--from-chain", action="store_true", help="use the current chain head")
    c.add_argument("--block", type=int, default=0, help="issued block number (with --block-hash)")
    c.add_argument("--deadline-s", type=float, default=DEFAULT_DEADLINE_S)
    c.add_argument("--inner-steps", type=int, default=DEFAULT_INNER_STEPS)
    c.add_argument("--out", default="-")
    _add_config_args(c, required=False)  # config only needed for --from-chain

    r = sub.add_parser("respond", help="run a challenge locally and emit the response JSON")
    r.add_argument("--challenge", required=True, help="challenge JSON file, '-' for stdin")
    r.add_argument("--backend", choices=("mok", "reference"), default="mok")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--out", default="-")

    v = sub.add_parser("verify", help="judge a response against the expected root + deadline")
    v.add_argument("--challenge", required=True)
    v.add_argument("--response", required=True)
    v.add_argument("--expected-root", required=True)
    v.add_argument("--issued-ts", type=float, required=True)
    v.add_argument("--received-ts", type=float, required=True)
    v.add_argument("--out", default="-")
    return p


def _read_json(path: str) -> Any:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    return json.loads(raw)


def attest_main(argv: list[str] | None = None) -> int:
    args = build_attest_parser().parse_args(argv)
    if args.cmd == "challenge":
        if args.block_hash:
            challenge = make_challenge(
                bytes.fromhex(args.block_hash),
                args.block,
                deadline_s=args.deadline_s,
                inner_steps=args.inner_steps,
            )
        else:
            if args.config is None:
                raise SystemExit("--from-chain requires --config")
            cfg = _load_config(args.config, args.overlay)
            chain = _make_chain(cfg)
            challenge = Verifier(
                deadline_s=args.deadline_s, inner_steps=args.inner_steps
            ).issue(chain)
        _emit(challenge.model_dump(), args.out)
        return 0
    if args.cmd == "respond":
        challenge = AttestationChallenge.model_validate(_read_json(args.challenge))
        enforce_determinism()  # same pins as the torchrun entry (reference_step.main)
        response = run_reference(challenge, backend=args.backend, device=args.device)
        _emit(response.model_dump(), args.out)
        return 0
    # verify
    challenge = AttestationChallenge.model_validate(_read_json(args.challenge))
    response = AttestationResponse.model_validate(_read_json(args.response))
    verdict = judge(
        challenge,
        response,
        args.expected_root,
        received_ts=args.received_ts,
        issued_ts=args.issued_ts,
    )
    _emit(verdict.model_dump(), args.out)
    return 0 if verdict.ok else 1


# ===========================================================================
# mok-onboard
# ===========================================================================


def build_onboard_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mok-onboard",
        description="Full miner onboarding: preflight → wallet → register → bucket → init → self-attest.",
    )
    _add_config_args(p)
    p.add_argument("--interactive", action="store_true", help="allow wallet-creation prompts")
    p.add_argument("--cache-dir", default=None, help="shard-cache dir for the preflight disk check")
    p.add_argument("--local-dir", default="checkpoints", help="local checkpoint dir for the init")
    p.add_argument("--owner-uid", type=int, default=None, help="UID publishing the init checkpoint")
    p.add_argument("--expected-init-root", default=None, help="override the on-chain init root")
    p.add_argument("--backend", choices=("mok", "reference"), default="mok")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for step in ("preflight", "wallet", "register", "bucket", "init", "attest"):
        p.add_argument(f"--skip-{step}", action="store_true", help=f"skip the {step} step")
    return p


def onboard_main(argv: list[str] | None = None) -> int:
    args = build_onboard_parser().parse_args(argv)
    cfg = _load_config(args.config, args.overlay)
    enforce_determinism()  # process entry, before any torch work (DCP init-fetch creates a CUDA context)

    def step(name: str, **fields: Any) -> None:
        _emit({"step": name, **fields})

    if args.skip_preflight:
        step("preflight", skipped=True)
    else:
        report = run_preflight(cache_dir=args.cache_dir)
        step("preflight", ok=report.ok, checks=[c.model_dump() for c in report.checks])
        report.strict()

    wallet = None
    if args.skip_wallet:
        step("wallet", skipped=True)
    else:
        wallet = ensure_wallet(cfg.chain, interactive=args.interactive)
        step("wallet", ok=True)

    chain = None
    if args.skip_register:
        step("register", skipped=True)
    else:
        chain = _make_chain(cfg, wallet)
        uid = register(chain)
        step("register", ok=True, uid=uid)

    if args.skip_bucket:
        step("bucket", skipped=True)
    else:
        chain = chain if chain is not None else _make_chain(cfg, wallet)
        committed = commit_bucket_credentials(chain, bucket_creds_from_env())
        step("bucket", ok=True, committed=committed)

    if args.skip_init:
        step("init", skipped=True)
    else:
        chain = chain if chain is not None else _make_chain(cfg, wallet)
        if args.owner_uid is None:
            raise SystemExit("--owner-uid is required for the init step (or pass --skip-init)")
        expected = args.expected_init_root or chain.get_manifest_hash(args.owner_uid)
        if not expected:
            raise SystemExit(
                f"uid {args.owner_uid} has no init commitment on-chain; pass --expected-init-root"
            )
        owner_bucket = chain.get_bucket(args.owner_uid)
        if owner_bucket is None:
            raise SystemExit(f"uid {args.owner_uid} has no bucket committed on-chain")

        async def _fetch() -> Any:
            from mok_core.storage import StorageClient  # noqa: PLC0415 — aioboto3 stays lazy

            async with StorageClient(write_creds_from_env(), cfg.storage) as storage:
                return await fetch_and_verify_init(
                    storage,
                    chain,
                    expected,
                    local_dir=args.local_dir,
                    bucket=owner_bucket,
                    owner_uid=args.owner_uid,
                )

        _state, _outer, meta = asyncio.run(_fetch())
        step("init", ok=True, state_root=meta.state_root, window=meta.window)

    if args.skip_attest:
        step("attest", skipped=True)
    else:
        chain = chain if chain is not None else _make_chain(cfg, wallet)
        challenge = Verifier().issue(chain)
        first = run_reference(challenge, backend=args.backend, device=args.device)
        second = run_reference(challenge, backend=args.backend, device=args.device)
        deterministic = first.state_root == second.state_root
        in_time = first.wall_time_s <= challenge.deadline_s
        step(
            "attest",
            ok=deterministic and in_time,
            deterministic=deterministic,
            wall_time_s=first.wall_time_s,
            deadline_s=challenge.deadline_s,
            state_root=first.state_root,
        )
        if not (deterministic and in_time):
            return 1

    step("done", ok=True)
    return 0


# ===========================================================================
# mok-init-publish
# ===========================================================================


def build_init_publish_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mok-init-publish",
        description="Owner-side: build, checkpoint, upload and commit the seed-42 initialization.",
    )
    _add_config_args(p)
    p.add_argument("--local-dir", required=True, help="local checkpoint directory")
    p.add_argument("--seed", type=int, default=DEFAULT_INIT_SEED)
    p.add_argument("--backend", choices=("mok", "reference"), default="reference")
    p.add_argument("--device", default="cpu")
    p.add_argument("--spec-version", type=int, default=1)
    p.add_argument(
        "--local-only",
        action="store_true",
        help="skip R2 upload and the on-chain commit (offline/dry run)",
    )
    p.add_argument("--out", default="-")
    return p


def init_publish_main(argv: list[str] | None = None) -> int:
    args = build_init_publish_parser().parse_args(argv)
    cfg = _load_config(args.config, args.overlay)

    async def _go() -> str:
        common: dict[str, Any] = {
            "local_dir": args.local_dir,
            "seed": args.seed,
            "device": args.device,
            "backend": args.backend,
            "spec_version": args.spec_version,
        }
        if args.local_only:
            return await build_and_publish_init(cfg, None, None, **common)
        from mok_core.storage import StorageClient  # noqa: PLC0415 — aioboto3 stays lazy

        chain = _make_chain(cfg)
        async with StorageClient(write_creds_from_env(), cfg.storage) as storage:
            return await build_and_publish_init(cfg, storage, chain, **common)

    root = asyncio.run(_go())
    _emit({"init_state_root": root, "seed": args.seed, "local_only": args.local_only}, args.out)
    return 0


# ===========================================================================
# mok-calibrate
# ===========================================================================


def build_calibrate_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mok-calibrate", description="Step-B calibration suite.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        _add_config_args(sp)
        sp.add_argument("--data-dir", required=True, help="dir with shard_index.json + shard files")
        sp.add_argument("--work-dir", required=True, help="scratch dir (storage/cache/checkpoints)")
        sp.add_argument("--run-seed", default="00" * 32, help="64-hex PRF run seed")
        sp.add_argument("--seed", type=int, default=DEFAULT_INIT_SEED, help="model init seed")
        sp.add_argument("--uid", type=int, default=0)
        sp.add_argument("--start-window", type=int, default=0)
        sp.add_argument("--backend", choices=("mok", "reference"), default="reference")
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--out", default="-")

    r = sub.add_parser("rehearse", help="full-protocol loopback windows + determinism check")
    common(r)
    r.add_argument("--windows", type=int, default=3)

    s = sub.add_parser("sweep", help="MoKConfig grid sweep -> C/configs/mok_tuned.yaml")
    common(s)
    s.add_argument("--windows-per-point", type=int, default=2)
    s.add_argument("--sms", default="24,36,48", help="comma list of comm-SM counts (fwd == bwd)")
    s.add_argument("--minibatch", default="2048,4096,8192", help="comma list of minibatch sizes")
    s.add_argument("--tuned-out", default=str(DEFAULT_TUNED_PATH))

    a = sub.add_parser("adam-ab", help="Adam-reset A/B (reset every window vs every K)")
    common(a)
    a.add_argument("--windows", type=int, default=5)
    a.add_argument("--k", type=int, default=DEFAULT_K)
    a.add_argument("--threshold-nats", type=float, default=DEFAULT_THRESHOLD_NATS)
    return p


def _load_dataset(data_dir: str | Path) -> tuple[DatasetShardIndex, Callable[[int], Path]]:
    """The step-A shard tree: ``shard_index.json`` + content-addressed shard files."""
    root = Path(data_dir)
    index = DatasetShardIndex.model_validate(
        json.loads((root / "shard_index.json").read_text(encoding="utf-8"))
    )

    def shard_path(i: int) -> Path:
        return root / shard_filename(index.leaf(i))

    return index, shard_path


def calibrate_main(argv: list[str] | None = None) -> int:
    args = build_calibrate_parser().parse_args(argv)
    cfg = _load_config(args.config, args.overlay)
    index, shard_path = _load_dataset(args.data_dir)
    run_seed = bytes.fromhex(args.run_seed)
    manifest = local_manifest(cfg, index, shard_path=shard_path, run_seed=run_seed)

    if args.cmd == "rehearse":
        model = _build_model(cfg, backend=args.backend, seed=args.seed, device=args.device)
        report = run_calibration_windows(
            args.windows,
            cfg,
            manifest,
            model=model,
            index=index,
            shard_path=shard_path,
            work_dir=args.work_dir,
            uid=args.uid,
            start_window=args.start_window,
            device=args.device,
        )
        _emit(report.model_dump(), args.out)
        return 0 if report.ok else 1

    if args.cmd == "sweep":
        points = [
            SweepPoint(fwd_num_comm_sms=sms, bwd_num_comm_sms=sms, minibatch_size=mb)
            for sms in (int(v) for v in args.sms.split(","))
            for mb in (int(v) for v in args.minibatch.split(","))
        ]
        results = run_sweep(
            cfg,
            manifest,
            model_factory=lambda c: _build_model(
                c, backend=args.backend, seed=args.seed, device=args.device
            ),
            shard_path=shard_path,
            points=points,
            windows_per_point=args.windows_per_point,
            start_window=args.start_window,
            uid=args.uid,
            device=args.device,
        )
        best = select_best(results)
        tuned = emit_tuned_yaml(
            best.mok,
            args.tuned_out,
            provenance=(
                f"grid sms={args.sms} minibatch={args.minibatch} "
                f"windows_per_point={args.windows_per_point} backend={args.backend}"
            ),
        )
        _emit(
            {
                "results": [
                    {
                        "point": r.point.model_dump(),
                        "mean_window_s": r.mean_window_s,
                        "final_loss": r.final_loss,
                    }
                    for r in results
                ],
                "best": best.point.model_dump(),
                "tuned_yaml": str(tuned),
            },
            args.out,
        )
        return 0

    # adam-ab
    model = _build_model(cfg, backend=args.backend, seed=args.seed, device=args.device)
    ab = run_adam_ab(
        args.windows,
        cfg,
        manifest,
        model=model,
        shard_path=shard_path,
        k=args.k,
        threshold_nats=args.threshold_nats,
        uid=args.uid,
        start_window=args.start_window,
        device=args.device,
    )
    _emit(ab.model_dump(), args.out)
    return 0
