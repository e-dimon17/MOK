"""Calibration: loopback dress rehearsal, MoKConfig sweep,
and the Adam-reset A/B — plus the production loopback harness they share."""

from .adam_ab import DEFAULT_K, DEFAULT_THRESHOLD_NATS, ABReport, run_adam_ab, run_arm
from .local_harness import (
    LocalLoopbackHarness,
    LoopbackClock,
    MemoryStorage,
    ScriptedChain,
    local_manifest,
    make_compressor,
    make_outer_step,
)
from .rehearsal import CalibrationError, CalibrationReport, run_calibration_windows
from .sweep import (
    DEFAULT_TUNED_PATH,
    SweepPoint,
    SweepResult,
    apply_point,
    default_grid,
    emit_tuned_yaml,
    run_sweep,
    select_best,
)

__all__ = [
    "DEFAULT_K",
    "DEFAULT_THRESHOLD_NATS",
    "DEFAULT_TUNED_PATH",
    "ABReport",
    "CalibrationError",
    "CalibrationReport",
    "LocalLoopbackHarness",
    "LoopbackClock",
    "MemoryStorage",
    "ScriptedChain",
    "SweepPoint",
    "SweepResult",
    "apply_point",
    "default_grid",
    "emit_tuned_yaml",
    "local_manifest",
    "make_compressor",
    "make_outer_step",
    "run_adam_ab",
    "run_arm",
    "run_calibration_windows",
    "run_sweep",
    "select_best",
]
