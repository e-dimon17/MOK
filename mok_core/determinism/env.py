"""Determinism environment enforcement + fingerprinting.

Called at the top of every miner/validator/auditor process, BEFORE any CUDA
context is created. Raises rather than warns: a nondeterministic node cannot
pass audits, so failing fast is the kind option.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field

# Pinned NCCL selection: algorithm/protocol autotuning is timing-dependent and
# may diverge across otherwise-identical nodes.
_REQUIRED_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    # Inductor autotuning is benchmarking-based (timing-dependent). The blessed
    # container ships a pre-baked FX-graph cache; runtime autotune stays off.
    "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
    "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
}

CONTAINER_DIGEST_ENV = "MOK_CONTAINER_DIGEST"


class DeterminismError(RuntimeError):
    pass


def enforce_determinism(*, allow_uninitialized_cuda_check: bool = True) -> None:
    import torch  # noqa: PLC0415

    if allow_uninitialized_cuda_check and torch.cuda.is_initialized():
        raise DeterminismError(
            "enforce_determinism() must run before any CUDA context is created "
            "(CUBLAS_WORKSPACE_CONFIG would be ignored)"
        )

    for key, value in _REQUIRED_ENV.items():
        current = os.environ.get(key)
        if current is None:
            os.environ[key] = value
        elif current != value:
            raise DeterminismError(f"{key}={current!r} conflicts with required {value!r}")

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Pinned matmul policy: bf16 autocast covers the fast paths; TF32 stays off
    # so any stray fp32 matmul is bit-identical across the fleet.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def assert_container_digest(expected: str) -> None:
    got = os.environ.get(CONTAINER_DIGEST_ENV, "")
    if got != expected:
        raise DeterminismError(
            f"container digest mismatch: manifest pins {expected!r}, environment reports {got!r}. "
            "Run inside the blessed image."
        )


@dataclass(frozen=True)
class EnvironmentFingerprint:
    torch_version: str
    cuda_version: str
    cudnn_version: str
    driver_visible_devices: str
    mok_version: str
    python_version: str
    platform: str
    env_pins: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return self.__dict__ | {"env_pins": dict(self.env_pins)}


def environment_fingerprint() -> EnvironmentFingerprint:
    import torch  # noqa: PLC0415

    try:
        import mok  # noqa: PLC0415

        mok_version = getattr(mok, "__version__", "unknown")
    except ImportError:
        mok_version = "absent"

    return EnvironmentFingerprint(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "cpu",
        cudnn_version=str(torch.backends.cudnn.version() or "n/a"),
        driver_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        mok_version=mok_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
        env_pins={k: os.environ.get(k, "") for k in _REQUIRED_ENV},
    )
