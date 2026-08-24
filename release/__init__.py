"""Evaluation + release: benchmark runner, provenance bundle
builder/verifier, replay CLI, and the HuggingFace release uploader.

Everything here is CPU-importable; heavy deps (lm-eval, bigcode harness,
huggingface_hub, torch DCP, subnet.core.replay) load lazily inside the functions
that need them.
"""

from .hf_upload import MODEL_CARD_TEMPLATE, PlannedOp, render_model_card, upload_release
from .provenance import (
    BUNDLE_SPEC_VERSION,
    BundleError,
    BundleManifest,
    WindowRecord,
    build_bundle,
    bundle_root_hash,
)
from .run_evals import DEFAULT_TASKS, extract_results, humaneval_cmd, results_to_markdown
from .verify_bundle import VerifyReport, verify

__all__ = [
    "BUNDLE_SPEC_VERSION",
    "DEFAULT_TASKS",
    "MODEL_CARD_TEMPLATE",
    "BundleError",
    "BundleManifest",
    "PlannedOp",
    "VerifyReport",
    "WindowRecord",
    "build_bundle",
    "bundle_root_hash",
    "extract_results",
    "humaneval_cmd",
    "render_model_card",
    "results_to_markdown",
    "upload_release",
    "verify",
]
