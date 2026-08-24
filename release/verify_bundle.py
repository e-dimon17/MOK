"""Offline verifier for the release provenance bundle. CPU-only, zero GPU deps.

`verify(bundle_dir)` re-derives everything index.json claims:

  1. index.json parses and has the exact BundleManifest shape;
  2. every listed file exists and its blake2b-256 matches; no unlisted files
     hide in the bundle; the root_hash recomputes from the (relpath, digest)
     pairs;
  3. manifest.json parses as a RunManifest and its canonical manifest_hash
     equals the index's claim;
  4. windows.jsonl is strictly window-monotonic with hex64 state_roots;
  5. every audit report is well-formed per the AuditReport contract, and —
     when a report carries a signature AND `verify_audit_signature` is
     importable from mok_core.chain — the signature checks out (optional-pass:
     a chain layer without that hook skips the check, it never fails it);
  6. evals.json parses; weights `.ref.json` references are well-formed
     (copied weight files are covered by the hash check in step 2).

Never raises on bad bundles: every defect becomes one line in
VerifyReport.problems. `python -m release.verify_bundle <dir>` exits 0 iff ok.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from release.provenance import (
    AUDITS_FILENAME,
    BUNDLE_SPEC_VERSION,
    EVALS_FILENAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    REQUIRED_FILES,
    WEIGHTS_DIRNAME,
    WEIGHTS_REF_SUFFIX,
    WINDOWS_FILENAME,
    WindowRecord,
    audit_report_message,
    audit_report_problems,
    blake2b_file,
    bundle_root_hash,
    is_hex64,
)

#: Signature of the optional chain hook: (message32, signature, auditor_uid) -> bool
SignatureVerifier = Callable[[bytes, bytes, int], bool]


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    files_checked: int = 0
    windows: int = 0
    audits: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "files_checked": self.files_checked,
            "windows": self.windows,
            "audits": self.audits,
        }


def _chain_signature_verifier() -> SignatureVerifier | None:
    """The optional `verify_audit_signature` hook from mok_core.chain (lazy).

    Returns None when the chain layer (or the hook) is unavailable — signature
    checks then optional-pass, keeping this verifier runnable fully offline.
    """
    try:
        from mok_core import chain as chain_mod  # noqa: PLC0415
    except Exception:
        return None
    fn = getattr(chain_mod, "verify_audit_signature", None)
    return fn if callable(fn) else None


def _load_json(path: Path, problems: list[str], label: str) -> Any | None:
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        problems.append(f"{label}: missing")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        problems.append(f"{label}: invalid JSON ({e})")
    return None


def _check_index_shape(index: Any, problems: list[str]) -> dict[str, str] | None:
    """Validate the raw index.json object field-by-field; return its files map."""
    if not isinstance(index, Mapping):
        problems.append(f"{INDEX_FILENAME}: not a JSON object")
        return None
    expected_keys = {"spec_version", "manifest_hash", "files", "root_hash", "built_at_block"}
    keys = set(index)
    for k in sorted(expected_keys - keys):
        problems.append(f"{INDEX_FILENAME}: missing key {k!r}")
    for k in sorted(keys - expected_keys):
        problems.append(f"{INDEX_FILENAME}: unexpected key {k!r}")
    if expected_keys - keys:
        return None

    sv = index["spec_version"]
    if not isinstance(sv, int) or isinstance(sv, bool):
        problems.append(f"{INDEX_FILENAME}: spec_version must be an integer")
    elif sv != BUNDLE_SPEC_VERSION:
        problems.append(f"{INDEX_FILENAME}: unsupported spec_version {sv} (expected {BUNDLE_SPEC_VERSION})")
    if not is_hex64(index["manifest_hash"]):
        problems.append(f"{INDEX_FILENAME}: manifest_hash is not lowercase hex64")
    if not is_hex64(index["root_hash"]):
        problems.append(f"{INDEX_FILENAME}: root_hash is not lowercase hex64")
    bab = index["built_at_block"]
    if not isinstance(bab, int) or isinstance(bab, bool) or bab < 0:
        problems.append(f"{INDEX_FILENAME}: built_at_block must be a non-negative integer")

    files = index["files"]
    if not isinstance(files, Mapping):
        problems.append(f"{INDEX_FILENAME}: files must be an object")
        return None
    out: dict[str, str] = {}
    for rel, digest in files.items():
        if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in rel.split("/"):
            problems.append(f"{INDEX_FILENAME}: illegal file path {rel!r}")
            continue
        if not is_hex64(digest):
            problems.append(f"{INDEX_FILENAME}: digest for {rel!r} is not lowercase hex64")
            continue
        out[rel] = digest
    return out


def _check_files(bundle: Path, files: dict[str, str], problems: list[str]) -> int:
    checked = 0
    for rel in sorted(files):
        path = bundle / rel
        if not path.is_file():
            problems.append(f"missing file: {rel}")
            continue
        checked += 1
        if blake2b_file(path) != files[rel]:
            problems.append(f"hash mismatch: {rel}")
    on_disk = {
        p.relative_to(bundle).as_posix()
        for p in bundle.rglob("*")
        if p.is_file() and p.relative_to(bundle).as_posix() != INDEX_FILENAME
    }
    for rel in sorted(on_disk - set(files)):
        problems.append(f"unlisted file: {rel}")
    for rel in REQUIRED_FILES:
        if rel not in files:
            problems.append(f"{INDEX_FILENAME}: required file {rel!r} not listed")
    return checked


def _check_manifest(bundle: Path, claimed_hash: str, problems: list[str]) -> None:
    raw = _load_json(bundle / MANIFEST_FILENAME, problems, MANIFEST_FILENAME)
    if raw is None:
        return
    from mok_core.config import RunManifest  # noqa: PLC0415  (pydantic-only, no torch)

    try:
        manifest = RunManifest.model_validate(raw)
    except Exception as e:
        problems.append(f"{MANIFEST_FILENAME}: not a valid RunManifest ({type(e).__name__})")
        return
    if manifest.manifest_hash() != claimed_hash:
        problems.append(f"{MANIFEST_FILENAME}: canonical hash does not match index manifest_hash")


def _check_windows(bundle: Path, problems: list[str]) -> tuple[int, set[int]]:
    path = bundle / WINDOWS_FILENAME
    if not path.is_file():
        return 0, set()
    windows: set[int] = set()
    count = 0
    prev = -1
    for i, line in enumerate(path.read_bytes().splitlines(), start=1):
        if not line.strip():
            problems.append(f"{WINDOWS_FILENAME} line {i}: empty line")
            continue
        try:
            record = WindowRecord.model_validate(json.loads(line))
        except Exception as e:
            problems.append(f"{WINDOWS_FILENAME} line {i}: invalid WindowRecord ({type(e).__name__})")
            continue
        count += 1
        if record.window <= prev:
            problems.append(
                f"{WINDOWS_FILENAME} line {i}: window {record.window} not strictly increasing"
            )
        prev = max(prev, record.window)
        windows.add(record.window)
    return count, windows


def _check_audits(bundle: Path, known_windows: set[int], problems: list[str]) -> int:
    path = bundle / AUDITS_FILENAME
    if not path.is_file():
        return 0
    verifier = _chain_signature_verifier()
    count = 0
    for i, line in enumerate(path.read_bytes().splitlines(), start=1):
        where = f"{AUDITS_FILENAME} line {i}"
        if not line.strip():
            problems.append(f"{where}: empty line")
            continue
        try:
            report = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"{where}: invalid JSON")
            continue
        count += 1
        rep_problems = audit_report_problems(report, where=where)
        problems.extend(rep_problems)
        if rep_problems:
            continue
        if known_windows and report["window"] not in known_windows:
            problems.append(f"{where}: window {report['window']} has no WindowRecord")
        sig = report["signature"]
        if sig and verifier is not None:
            try:
                ok = verifier(audit_report_message(report), bytes.fromhex(sig), report["auditor_uid"])
            except Exception as e:
                ok = False
                problems.append(f"{where}: signature verification errored ({type(e).__name__})")
            else:
                if not ok:
                    problems.append(f"{where}: signature verification failed")
    return count


def _check_evals(bundle: Path, problems: list[str]) -> None:
    raw = _load_json(bundle / EVALS_FILENAME, problems, EVALS_FILENAME)
    if raw is None:
        return
    if not isinstance(raw, Mapping) or "results" not in raw or "extra" not in raw:
        problems.append(f"{EVALS_FILENAME}: must be an object with 'results' and 'extra'")
    elif not isinstance(raw["results"], Mapping):
        problems.append(f"{EVALS_FILENAME}: 'results' must be an object")


def _check_weight_refs(bundle: Path, files: dict[str, str], problems: list[str]) -> None:
    for rel in sorted(files):
        if not (rel.startswith(f"{WEIGHTS_DIRNAME}/") and rel.endswith(WEIGHTS_REF_SUFFIX)):
            continue
        ref = _load_json(bundle / rel, problems, rel)
        if ref is None:
            continue
        if (
            not isinstance(ref, Mapping)
            or set(ref) != {"blake2b", "bytes", "filename"}
            or not is_hex64(ref.get("blake2b"))
            or not isinstance(ref.get("filename"), str)
            or isinstance(ref.get("bytes"), bool)
            or not isinstance(ref.get("bytes"), int)
            or ref["bytes"] < 0
        ):
            problems.append(f"{rel}: malformed weights reference")


def verify(bundle_dir: str | Path) -> VerifyReport:
    """Check a provenance bundle end to end. Collects problems, never raises."""
    bundle = Path(bundle_dir)
    problems: list[str] = []
    if not bundle.is_dir():
        return VerifyReport(ok=False, problems=[f"bundle directory does not exist: {bundle}"])

    index_raw = _load_json(bundle / INDEX_FILENAME, problems, INDEX_FILENAME)
    if index_raw is None:
        return VerifyReport(ok=False, problems=problems)
    files = _check_index_shape(index_raw, problems)
    if files is None:
        return VerifyReport(ok=False, problems=problems)

    files_checked = _check_files(bundle, files, problems)

    if is_hex64(index_raw.get("root_hash")) and bundle_root_hash(files) != index_raw["root_hash"]:
        problems.append(f"{INDEX_FILENAME}: root_hash does not recompute from files")

    if is_hex64(index_raw.get("manifest_hash")):
        _check_manifest(bundle, index_raw["manifest_hash"], problems)
    n_windows, known_windows = _check_windows(bundle, problems)
    n_audits = _check_audits(bundle, known_windows, problems)
    _check_evals(bundle, problems)
    _check_weight_refs(bundle, files, problems)

    return VerifyReport(
        ok=not problems,
        problems=problems,
        files_checked=files_checked,
        windows=n_windows,
        audits=n_audits,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mok-verify-bundle",
        description="Verify a MoK release provenance bundle offline (no GPU, no network).",
    )
    parser.add_argument("bundle", type=Path, help="path to the bundle directory")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify(args.bundle)
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    elif report.ok:
        print(
            f"OK: {report.files_checked} files, {report.windows} windows, "
            f"{report.audits} audit reports verified"
        )
    else:
        for p in report.problems:
            print(f"PROBLEM: {p}", file=sys.stderr)
        print(f"FAILED: {len(report.problems)} problem(s)", file=sys.stderr)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
