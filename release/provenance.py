"""Release provenance bundle — the artifact that makes the whole run auditable offline.

The bundle packages everything a third party needs to check the training run
without trusting us: the on-chain run manifest, every window's certified
state_root / payload hashes, the signed audit log, the release weights (or
hashed references to them), the benchmark results, and a copy of the replay
CLI so "replay any window yourself" is a one-liner.

Layout (all paths relative to the bundle root)::

    index.json              canonical JSON of BundleManifest:
                              {spec_version, manifest_hash, files, root_hash, built_at_block}
    manifest.json           canonical JSON of the RunManifest (file hash == manifest_hash)
    windows.jsonl           one canonical-JSON WindowRecord per line, strictly increasing window
    audits.jsonl            one canonical-JSON audit report per line, sorted
                            by (window, miner_uid, auditor_uid)
    evals.json              canonical JSON {"extra": ..., "results": {task: {metric: value}}}
    weights/<name>          release weight files, copied verbatim
    weights/<name>.ref.json hashed reference when copy_weights=False
    replay/replay_window.py verbatim copy of release/replay_window.py

Byte rules (consensus surface of the release — golden-vector pinned in
tests/unit/test_release_provenance.py, any change is a bundle spec bump):
  - every JSON artifact is written via mok_core.config.canonical_bytes
    (sorted keys, no whitespace, shortest-repr floats, utf-8)
  - file hashes are hex blake2b-256 over raw file bytes
  - root_hash = blake2b-256 over sorted (relpath, digest) pairs, each pair
    encoded as len(relpath) le32 ‖ relpath utf-8 ‖ raw 32 digest bytes
  - the build is a pure function of its inputs: no timestamps anywhere;
    `built_at_block` (the chain height the bundle was cut at) is an input.

The root_hash is what the owner commits on-chain at release time.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from mok_core.config import RunManifest, canonical_bytes, canonical_hash
from mok_core.config.schemas import FrozenModel

BUNDLE_SPEC_VERSION = 1

INDEX_FILENAME = "index.json"
MANIFEST_FILENAME = "manifest.json"
WINDOWS_FILENAME = "windows.jsonl"
AUDITS_FILENAME = "audits.jsonl"
EVALS_FILENAME = "evals.json"
WEIGHTS_DIRNAME = "weights"
REPLAY_DIRNAME = "replay"
REPLAY_SCRIPT_NAME = "replay_window.py"
WEIGHTS_REF_SUFFIX = ".ref.json"

#: Required files of every bundle (index.json is implicit — it lists the rest).
REQUIRED_FILES = (MANIFEST_FILENAME, WINDOWS_FILENAME, AUDITS_FILENAME, EVALS_FILENAME)

#: The AuditReport wire shape produced by subnet/core/replay.py (plan contract).
AUDIT_REPORT_FIELDS = (
    "miner_uid",
    "window",
    "theta_start_root",
    "committed_theta_end",
    "replayed_theta_end",
    "match",
    "divergences",
    "wall_time_s",
    "auditor_uid",
    "signature",
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})+$")


class BundleError(ValueError):
    """A provenance bundle could not be built from the given inputs."""


def is_hex64(value: Any) -> bool:
    """True iff `value` is a 64-char lowercase hex string (a blake2b-256 digest)."""
    return isinstance(value, str) and _HEX64_RE.fullmatch(value) is not None


def blake2b_hex(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def blake2b_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex blake2b-256 of a file's raw bytes (streaming; no torch dependency)."""
    h = hashlib.blake2b(digest_size=32)
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def bundle_root_hash(files: Mapping[str, str]) -> str:
    """The release root: blake2b-256 over sorted (relpath, digest) pairs.

    Wire rule (consensus constant): for each relpath in ascending string order,
    absorb len(relpath) le32 ‖ relpath utf-8 ‖ the raw 32 digest bytes.
    """
    h = hashlib.blake2b(digest_size=32)
    for relpath in sorted(files):
        digest = files[relpath]
        if not is_hex64(digest):
            raise BundleError(f"file digest for {relpath!r} is not lowercase hex64: {digest!r}")
        h.update(len(relpath).to_bytes(4, "little"))
        h.update(relpath.encode("utf-8"))
        h.update(bytes.fromhex(digest))
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


class WindowRecord(FrozenModel):
    """One certified window of the run, as archived in windows.jsonl."""

    window: int
    state_root: str                                   # θ_start root certified for this window
    certificate: dict[str, Any] | None = None         # WindowCertificate dump (leader-signed)
    payload_hashes: dict[int, str] = Field(default_factory=dict)   # uid -> payload hash
    telemetry_hash: str | None = None

    @model_validator(mode="after")
    def _check(self) -> WindowRecord:
        if self.window < 0:
            raise ValueError(f"window must be >= 0, got {self.window}")
        if not is_hex64(self.state_root):
            raise ValueError(f"state_root must be 64 lowercase hex chars, got {self.state_root!r}")
        for uid, ph in self.payload_hashes.items():
            if uid < 0:
                raise ValueError(f"payload_hashes uid must be >= 0, got {uid}")
            if not is_hex64(ph):
                raise ValueError(f"payload hash for uid {uid} must be 64 lowercase hex chars")
        if self.telemetry_hash is not None and not is_hex64(self.telemetry_hash):
            raise ValueError("telemetry_hash must be 64 lowercase hex chars or None")
        return self


class BundleManifest(FrozenModel):
    """The bundle's index.json — canonical JSON of exactly these five fields."""

    spec_version: int
    manifest_hash: str
    files: dict[str, str]                             # relpath -> hex blake2b-256
    root_hash: str
    built_at_block: int

    @model_validator(mode="after")
    def _check(self) -> BundleManifest:
        if not is_hex64(self.manifest_hash):
            raise ValueError("manifest_hash must be 64 lowercase hex chars")
        if self.built_at_block < 0:
            raise ValueError("built_at_block must be >= 0")
        if self.root_hash != bundle_root_hash(self.files):
            raise ValueError("root_hash does not match the files mapping")
        return self


# --------------------------------------------------------------------------- #
# Audit report validation (the plan's AuditReport dict shape)
# --------------------------------------------------------------------------- #


def audit_report_problems(report: Mapping[str, Any], *, where: str = "audit report") -> list[str]:
    """Well-formedness problems of one AuditReport dict; empty list == valid."""
    problems: list[str] = []
    if not isinstance(report, Mapping):
        return [f"{where}: not a JSON object"]
    for key in AUDIT_REPORT_FIELDS:
        if key not in report:
            problems.append(f"{where}: missing field {key!r}")
    if problems:
        return problems

    for key in ("miner_uid", "window", "auditor_uid"):
        v = report[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            problems.append(f"{where}: {key} must be a non-negative integer, got {v!r}")
    for key in ("theta_start_root", "committed_theta_end", "replayed_theta_end"):
        if not is_hex64(report[key]):
            problems.append(f"{where}: {key} must be 64 lowercase hex chars, got {report[key]!r}")
    if not isinstance(report["match"], bool):
        problems.append(f"{where}: match must be a bool, got {report['match']!r}")
    div = report["divergences"]
    if not isinstance(div, list) or not all(isinstance(d, Mapping) for d in div):
        problems.append(f"{where}: divergences must be a list of objects")
    wall = report["wall_time_s"]
    if isinstance(wall, bool) or not isinstance(wall, int | float) or not math.isfinite(wall) or wall < 0:
        problems.append(f"{where}: wall_time_s must be a finite number >= 0, got {wall!r}")
    sig = report["signature"]
    if not isinstance(sig, str) or (sig != "" and _HEX_RE.fullmatch(sig) is None):
        problems.append(f"{where}: signature must be '' or an even-length lowercase hex string")

    # Internal consistency: `match` is *defined* as bitwise equality of the
    # replayed and committed θ_end roots, and a matching replay has no divergences.
    if not problems:
        equal = report["committed_theta_end"] == report["replayed_theta_end"]
        if bool(report["match"]) != equal:
            problems.append(f"{where}: match={report['match']} inconsistent with theta_end roots")
        if report["match"] and div:
            problems.append(f"{where}: match=true but divergences is non-empty")
    return problems


def audit_report_message(report: Mapping[str, Any]) -> bytes:
    """The 32 bytes an auditor signs: blake2b-256 of the canonical JSON of the
    report with the 'signature' field removed. Consensus constant."""
    body = {k: v for k, v in report.items() if k != "signature"}
    return hashlib.blake2b(canonical_bytes(body), digest_size=32).digest()


def audit_sort_key(report: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(report["window"]), int(report["miner_uid"]), int(report["auditor_uid"]))


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def _write(out_dir: Path, relpath: str, data: bytes, files: dict[str, str]) -> None:
    path = out_dir / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    files[relpath] = blake2b_hex(data)


def build_bundle(
    out_dir: str | Path,
    *,
    manifest: RunManifest,
    window_records: Sequence[WindowRecord],
    audit_reports: Sequence[Mapping[str, Any]],
    weights_files: Sequence[str | Path],
    eval_results: Mapping[str, Any],
    extra: Mapping[str, Any],
    built_at_block: int = 0,
    copy_weights: bool = True,
    include_replay_script: bool = True,
) -> BundleManifest:
    """Build the release provenance bundle under `out_dir` (must be empty/absent).

    Fully deterministic given its inputs: identical arguments produce a
    byte-identical bundle and the same root_hash on any machine. No wall-clock
    enters any hash — `built_at_block` is the chain height of the release cut
    and is supplied by the caller.

    With copy_weights=False each weights file is represented by a
    `weights/<name>.ref.json` hashed reference instead of a copy (for release
    channels where the multi-GB safetensors live on HF and only their digests
    belong in the bundle).
    """
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise BundleError(f"refusing to build into non-empty directory {out}")
    out.mkdir(parents=True, exist_ok=True)

    # -- validate inputs before writing anything -------------------------- #
    records = sorted(window_records, key=lambda r: r.window)
    for prev, cur in zip(records, records[1:], strict=False):
        if cur.window <= prev.window:
            raise BundleError(f"window_records must have distinct windows (duplicate {cur.window})")
    known_windows = {r.window for r in records}

    problems: list[str] = []
    for i, report in enumerate(audit_reports):
        rep_problems = audit_report_problems(report, where=f"audit_reports[{i}]")
        if not rep_problems and report["window"] not in known_windows:
            rep_problems.append(f"audit_reports[{i}]: window {report['window']} has no WindowRecord")
        problems.extend(rep_problems)
    if problems:
        raise BundleError("invalid audit reports:\n" + "\n".join(problems))

    weight_paths = [Path(p) for p in weights_files]
    for p in weight_paths:
        if not p.is_file():
            raise BundleError(f"weights file does not exist: {p}")
    names = [p.name for p in weight_paths]
    if len(set(names)) != len(names):
        raise BundleError(f"weights file basenames must be unique, got {names}")

    # -- write ------------------------------------------------------------- #
    files: dict[str, str] = {}

    _write(out, MANIFEST_FILENAME, canonical_bytes(manifest), files)

    windows_blob = b"".join(canonical_bytes(r) + b"\n" for r in records)
    _write(out, WINDOWS_FILENAME, windows_blob, files)

    audits_sorted = sorted(audit_reports, key=audit_sort_key)
    audits_blob = b"".join(canonical_bytes(dict(r)) + b"\n" for r in audits_sorted)
    _write(out, AUDITS_FILENAME, audits_blob, files)

    _write(out, EVALS_FILENAME, canonical_bytes({"extra": dict(extra), "results": dict(eval_results)}), files)

    weights_dir = out / WEIGHTS_DIRNAME
    weights_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(weight_paths, key=lambda q: q.name):
        if copy_weights:
            relpath = f"{WEIGHTS_DIRNAME}/{p.name}"
            shutil.copyfile(p, weights_dir / p.name)
            files[relpath] = blake2b_file(weights_dir / p.name)
        else:
            ref = {"blake2b": blake2b_file(p), "bytes": p.stat().st_size, "filename": p.name}
            _write(out, f"{WEIGHTS_DIRNAME}/{p.name}{WEIGHTS_REF_SUFFIX}", canonical_bytes(ref), files)

    if include_replay_script:
        script_src = Path(__file__).with_name(REPLAY_SCRIPT_NAME)
        _write(out, f"{REPLAY_DIRNAME}/{REPLAY_SCRIPT_NAME}", script_src.read_bytes(), files)

    index = BundleManifest(
        spec_version=BUNDLE_SPEC_VERSION,
        manifest_hash=canonical_hash(manifest),
        files=files,
        root_hash=bundle_root_hash(files),
        built_at_block=built_at_block,
    )
    (out / INDEX_FILENAME).write_bytes(canonical_bytes(index))
    return index
