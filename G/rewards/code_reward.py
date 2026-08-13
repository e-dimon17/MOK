"""Execution-verified code rewards for RLVR (step G), sandboxed.

`verify_code(completion, tests)` extracts the completion's Python code block
and runs `code + "\\n\\n" + test` in an isolated subprocess once per test
assertion; the reward is the fraction of tests that exit 0.

Sandbox (never trust model-written code):
  - preferred: `bwrap` (bubblewrap) when installed — read-only binds of the
    interpreter/system dirs, tmpfs /tmp, private namespaces (incl. network),
    fresh working dir, killed on wall timeout;
  - fallback (CI path): a restricted plain subprocess — `python -I` (isolated
    mode), resource rlimits (CPU, address space, nproc, fsize, nofile), a
    minimal environment, a throwaway cwd, its own session (process-group
    SIGKILL on timeout), and `unshare -n` network isolation when the host
    allows it.

`run_code_batch` is the asyncio batch runner GRPO uses: verification is
subprocess-bound, so items run concurrently via `asyncio.to_thread` under a
semaphore.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import math
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DEFAULT_TIMEOUT_S = 6.0
DEFAULT_MEM_BYTES = 1 << 30  # 1 GiB address space
DEFAULT_NPROC = 128
MAX_OUTPUT_CHARS = 4096
_FSIZE_BYTES = 10 * 1024 * 1024
_NOFILE = 256

SandboxMode = Literal["auto", "bwrap", "rlimit"]

_PY_FENCE_RE = re.compile(r"```(?:python|py)[ \t]*\r?\n(.*?)```", re.DOTALL)
_ANY_FENCE_RE = re.compile(r"```[^\n`]*\r?\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------- #
# Code extraction
# --------------------------------------------------------------------------- #


def extract_code(completion: str) -> str | None:
    """The LAST ```python fenced block (models often revise), else the last
    generic fenced block, else the raw completion. None if effectively empty."""
    blocks = _PY_FENCE_RE.findall(completion) or _ANY_FENCE_RE.findall(completion)
    text = blocks[-1] if blocks else completion
    text = text.strip("\n").rstrip()
    return text if text.strip() else None


# --------------------------------------------------------------------------- #
# Sandboxed execution
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ExecResult:
    """Outcome of one sandboxed program run."""

    passed: bool
    returncode: int | None
    timed_out: bool
    stdout: str
    stderr: str


def bwrap_available() -> bool:
    return shutil.which("bwrap") is not None


@functools.lru_cache(maxsize=1)
def _unshare_net_usable() -> bool:
    """Whether `unshare -n` works here (needs root or user namespaces)."""
    if shutil.which("unshare") is None:
        return False
    try:
        probe = subprocess.run(  # noqa: S603 — fixed argv
            ["unshare", "-n", "true"], capture_output=True, timeout=5, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def resolve_sandbox(sandbox: SandboxMode) -> Literal["bwrap", "rlimit"]:
    if sandbox == "auto":
        return "bwrap" if bwrap_available() else "rlimit"
    if sandbox in ("bwrap", "rlimit"):
        return sandbox
    raise ValueError(f"unknown sandbox mode {sandbox!r}; expected auto|bwrap|rlimit")


def _bwrap_argv(script: Path, workdir: Path) -> list[str]:
    """bubblewrap invocation: RO system dirs + interpreter prefix, RW workdir,
    tmpfs /tmp, all namespaces unshared (no network), die with parent."""
    argv = [
        "bwrap",
        "--die-with-parent",
        "--unshare-all",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",  # noqa: S108 — path inside the sandbox mount ns
    ]
    ro_dirs = {"/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/alternatives"}
    ro_dirs.add(str(Path(sys.base_prefix).resolve()))
    ro_dirs.add(str(Path(sys.prefix).resolve()))
    for directory in sorted(ro_dirs):
        if os.path.exists(directory):
            argv += ["--ro-bind", directory, directory]
    argv += ["--bind", str(workdir), str(workdir), "--chdir", str(workdir)]
    argv += ["--", sys.executable, "-I", str(script)]
    return argv


def _rlimit_preexec(cpu_s: int, mem_bytes: int, nproc: int):
    """preexec_fn applying rlimits in the child (after fork, before exec)."""

    def apply_limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (_FSIZE_BYTES, _FSIZE_BYTES))
        for limit, value in (
            (resource.RLIMIT_AS, mem_bytes),
            (resource.RLIMIT_NPROC, nproc),
            (resource.RLIMIT_NOFILE, _NOFILE),
        ):
            with contextlib.suppress(ValueError, OSError):  # host cap below ours — keep it
                resource.setrlimit(limit, (value, value))

    return apply_limits


def run_snippet(
    program: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sandbox: SandboxMode = "auto",
    mem_bytes: int = DEFAULT_MEM_BYTES,
    nproc: int = DEFAULT_NPROC,
) -> ExecResult:
    """Run one Python program in the sandbox; passed == exited 0 in time."""
    mode = resolve_sandbox(sandbox)
    workdir = Path(tempfile.mkdtemp(prefix="mok-code-"))
    try:
        script = workdir / "prog.py"
        script.write_text(program, encoding="utf-8")
        env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "LC_ALL": "C.UTF-8"}
        if mode == "bwrap":
            argv = _bwrap_argv(script, workdir)
            preexec = None
        else:
            argv = [sys.executable, "-I", str(script)]
            if _unshare_net_usable():
                argv = ["unshare", "-n", "--", *argv]
            preexec = _rlimit_preexec(int(math.ceil(timeout_s)), mem_bytes, nproc)
        proc = subprocess.Popen(  # noqa: S603 — argv built above, sandboxed on purpose
            argv,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec,  # noqa: PLW1509 — rlimits must apply pre-exec
            start_new_session=True,  # own process group -> killable as a group
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(proc.pid)
            stdout, stderr = proc.communicate()
        if proc.returncode == -signal.SIGXCPU:
            # RLIMIT_CPU (== ceil(timeout_s)) fired before the wall clock:
            # death by CPU limit is a timeout, whichever kill lands first.
            timed_out = True
        return ExecResult(
            passed=(not timed_out and proc.returncode == 0),
            returncode=proc.returncode,
            timed_out=timed_out,
            stdout=stdout.decode("utf-8", errors="replace")[-MAX_OUTPUT_CHARS:],
            stderr=stderr.decode("utf-8", errors="replace")[-MAX_OUTPUT_CHARS:],
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _kill_process_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(pid), signal.SIGKILL)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def verify_code(
    completion: str,
    tests: list[str],
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sandbox: SandboxMode = "auto",
    mem_bytes: int = DEFAULT_MEM_BYTES,
    nproc: int = DEFAULT_NPROC,
) -> float:
    """Fraction of `tests` that pass against the completion's code block.

    Each test runs in its OWN subprocess (`code + "\\n\\n" + test`), so a crash
    or hang in one assertion cannot mask the others. No extractable code -> 0.
    """
    if not tests:
        raise ValueError("verify_code needs at least one test")
    code = extract_code(completion)
    if code is None:
        return 0.0
    passed = 0
    for test in tests:
        program = f"{code}\n\n{test}\n"
        result = run_snippet(
            program, timeout_s=timeout_s, sandbox=sandbox, mem_bytes=mem_bytes, nproc=nproc
        )
        passed += int(result.passed)
    return passed / len(tests)


async def run_code_batch(
    items: Sequence[Mapping[str, Any]],
    *,
    concurrency: int = 8,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sandbox: SandboxMode = "auto",
) -> list[float]:
    """Score a batch concurrently. Each item: {"completion": str, "tests":
    list[str], optional "timeout_s": float}. Returns scores in input order."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    semaphore = asyncio.Semaphore(concurrency)

    async def score(item: Mapping[str, Any]) -> float:
        async with semaphore:
            return await asyncio.to_thread(
                verify_code,
                str(item["completion"]),
                list(item["tests"]),
                timeout_s=float(item.get("timeout_s", timeout_s)),
                sandbox=sandbox,
            )

    return list(await asyncio.gather(*(score(item) for item in items)))


__all__ = [
    "DEFAULT_MEM_BYTES",
    "DEFAULT_NPROC",
    "DEFAULT_TIMEOUT_S",
    "ExecResult",
    "SandboxMode",
    "bwrap_available",
    "extract_code",
    "resolve_sandbox",
    "run_code_batch",
    "run_snippet",
    "verify_code",
]
