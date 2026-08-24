"""Replay-any-window-yourself CLI — the headline artifact of the release bundle.

Given the run manifest (from a provenance bundle or a bare file), a window
number, a miner uid and that window's θ_start checkpoint, this tool re-derives
the miner's entire window bitwise — data assignment, inner loop, compression,
certified outer step — via subnet.core.replay.WindowReplayer and writes the
AuditReport JSON. Exit code 0 iff the replayed θ_end root matches the
committed one.

Usage::

    python -m release.replay_window --bundle ./bundle --window 1234 --miner-uid 17 \
        --theta-start ./checkpoints/w00001234 --config ./configs/bulk.yaml \
        [--out report.json]

θ_start follows the checkpoint layout contract: a `checkpoints/w{window:08d}/`
directory whose `model/` subdirectory is a torch.distributed.checkpoint (DCP)
save of the master state dict (names from MoKTransformer.iter_master_params()).
Passing the `model/` directory itself also works.

All heavy imports (torch DCP, mok_core.model, subnet.core.replay) happen inside the
replay path, so `--help`/argument validation stay instant and unit-testable on
any machine. Verification of the surrounding bundle is release/verify_bundle.py's
job; this tool assumes its inputs and burns GPUs (or, with
--backend reference, CPU cycles) to check the compute itself.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from release.provenance import MANIFEST_FILENAME, audit_report_problems

DEFAULT_INIT_SEED = 42  # the published seed-42 initialization; values are overwritten by θ_start


class ReplayCLIError(RuntimeError):
    """User-facing failure of the replay CLI (bad inputs, mismatched config...)."""


# --------------------------------------------------------------------------- #
# Argument surface (pure — unit-tested)
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-replay-window",
        description="Bitwise-replay one miner's window and emit the AuditReport.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bundle", type=Path, help="provenance bundle directory (uses its manifest.json)")
    src.add_argument("--manifest", type=Path, help="path to a canonical RunManifest JSON file")
    parser.add_argument("--window", type=int, required=True, help="window number to replay")
    parser.add_argument("--miner-uid", type=int, required=True, help="uid whose window to replay")
    parser.add_argument(
        "--theta-start",
        type=Path,
        required=True,
        help="checkpoint dir for the window start (contract layout: contains model/ DCP dir)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="RunConfig YAML; its canonical config_hash must equal manifest.config_hash",
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "mok"),
        default="mok",
        help="model backend: 'mok' (8xB300 megakernel) or 'reference' (pure PyTorch)",
    )
    parser.add_argument("--device", default="cuda", help="device for the replay (cuda|cpu)")
    parser.add_argument("--out", type=Path, default=None, help="write the AuditReport JSON here")
    return parser


def load_manifest_arg(bundle: Path | None, manifest_path: Path | None) -> Any:
    """Resolve the RunManifest from --bundle or --manifest (lazy pydantic import)."""
    from mok_core.config import RunManifest  # noqa: PLC0415

    path = (Path(bundle) / MANIFEST_FILENAME) if bundle is not None else Path(manifest_path)  # type: ignore[arg-type]
    if not path.is_file():
        raise ReplayCLIError(f"manifest not found: {path}")
    try:
        return RunManifest.model_validate(json.loads(path.read_bytes()))
    except (json.JSONDecodeError, ValueError) as e:
        raise ReplayCLIError(f"invalid manifest at {path}: {e}") from e


def report_to_dict(report: Any) -> dict[str, Any]:
    """Normalize an AuditReport (dataclass / pydantic / plain dict) to the wire dict."""
    if isinstance(report, Mapping):
        out = dict(report)
    elif dataclasses.is_dataclass(report) and not isinstance(report, type):
        out = dataclasses.asdict(report)
    elif hasattr(report, "model_dump"):
        out = report.model_dump(mode="json")
    else:
        raise ReplayCLIError(f"unsupported AuditReport type: {type(report).__name__}")
    problems = audit_report_problems(out, where="AuditReport")
    if problems:
        raise ReplayCLIError("replayer returned a malformed AuditReport:\n" + "\n".join(problems))
    return out


# --------------------------------------------------------------------------- #
# Heavy path (behind main(); monkeypatched in unit tests, exercised on GPU)
# --------------------------------------------------------------------------- #


def _load_theta_start(model: Any, ckpt_dir: Path) -> None:
    """Load the master state from a contract-layout checkpoint into `model`.

    Prefers subnet/core/checkpoint.py's loader when it exposes one; otherwise reads
    the DCP directory directly per the checkpoint layout contract (single-rank
    CPU load via FileSystemReader is guaranteed to work).
    """
    try:
        from subnet.core import checkpoint as ckpt_mod  # noqa: PLC0415
    except ImportError:
        ckpt_mod = None
    loader = getattr(ckpt_mod, "load_master_state", None) if ckpt_mod is not None else None
    if loader is not None:
        loader(model, ckpt_dir)
        return

    import torch.distributed.checkpoint as dcp  # noqa: PLC0415

    model_dir = ckpt_dir / "model" if (ckpt_dir / "model").is_dir() else ckpt_dir
    if not model_dir.is_dir():
        raise ReplayCLIError(f"theta-start checkpoint dir not found: {ckpt_dir}")
    state = dict(model.iter_master_params())
    dcp.load(state_dict=state, storage_reader=dcp.FileSystemReader(str(model_dir)))


def _replay(
    manifest: Any,
    *,
    window: int,
    miner_uid: int,
    theta_start: Path,
    config_path: Path | None,
    backend: str,
    device: str,
) -> dict[str, Any]:
    """Build the model per the manifest's config hash, load θ_start, run the replay."""
    if config_path is None:
        raise ReplayCLIError(
            "--config is required for replay: the manifest pins only the canonical "
            "config_hash; supply the RunConfig YAML that hashes to it"
        )
    from mok_core.config import config_hash, load_run_config  # noqa: PLC0415

    cfg = load_run_config(config_path)
    got = config_hash(cfg)
    if got != manifest.config_hash:
        raise ReplayCLIError(
            f"config hash mismatch: {config_path} -> {got}, manifest pins {manifest.config_hash}"
        )

    from mok_core.model import init_model  # noqa: PLC0415

    model = init_model(cfg.model, seed=DEFAULT_INIT_SEED, device=device, backend=backend)
    _load_theta_start(model, Path(theta_start))

    from subnet.core.replay import WindowReplayer  # noqa: PLC0415

    replayer = WindowReplayer(manifest=manifest, cfg=cfg, model=model)
    report = replayer.replay(uid=miner_uid, window=window)
    return report_to_dict(report)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest_arg(args.bundle, args.manifest)
        report = _replay(
            manifest,
            window=args.window,
            miner_uid=args.miner_uid,
            theta_start=args.theta_start,
            config_path=args.config,
            backend=args.backend,
            device=args.device,
        )
    except ReplayCLIError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    blob = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n", encoding="utf-8")
    else:
        print(blob)

    match = bool(report["match"])
    verdict = "MATCH" if match else "MISMATCH"
    print(f"replay w{args.window} uid{args.miner_uid}: {verdict}", file=sys.stderr)
    return 0 if match else 1


if __name__ == "__main__":
    raise SystemExit(main())
