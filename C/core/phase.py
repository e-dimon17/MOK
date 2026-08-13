"""Phase resolution — every time-varying knob as a pure function of (manifest, window).

Steps D (quality anneal) and E (context extension) are nothing but phase
entries in the on-chain manifest; capacity anneals arrive as phase amendments.
`resolve_phase` folds ALL applicable entries cumulatively so a later amendment
(e.g. `capacity_0.5` at window 150) layers on top of the phase that introduced
the other knobs (e.g. `anneal` at window 100) instead of resetting them.
LR and grad-accum are closed-form functions of progress — no stateful
schedulers, so replay of any window needs no scheduler state.
"""

from __future__ import annotations

from dataclasses import dataclass

from mok_core.config import LRSpec, RunConfig, RunManifest, WindowConfig

__all__ = ["PhaseConfig", "accum_at", "lr_at", "resolve_phase"]


@dataclass(frozen=True)
class PhaseConfig:
    """The fully-resolved knob set for one window (RunConfig defaults + phase overrides)."""

    name: str
    data: str                          # DatasetManifestRef.name to draw windows from
    lr: LRSpec
    seq_len: int
    tokens_per_rank_microbatch: int
    rope_theta: float
    grad_accum: int
    capacity_multiplier: float
    inner_steps: int
    requires_restart: bool


def resolve_phase(manifest: RunManifest, cfg: RunConfig, window: int) -> PhaseConfig:
    """Resolve the active PhaseConfig for `window`.

    Starts from RunConfig defaults (data defaults to the manifest's first
    dataset), then walks every phase entry with start_window <= window in
    table order, applying each entry's non-None overrides — later amendments
    override earlier ones cumulatively. `name` and `requires_restart` come
    from the LAST applicable entry (manifest.phase_at): requires_restart marks
    the transition into that entry, not the whole lineage since window 0.
    """
    if window < 0:
        raise ValueError(f"window must be >= 0, got {window}")
    if not manifest.datasets:
        raise ValueError("manifest has no datasets — cannot resolve a data source")

    active = manifest.phase_at(window)
    data = manifest.datasets[0].name
    lr = cfg.inner.lr
    seq_len = cfg.model.seq_len
    tokens_per_rank_microbatch = cfg.window.tokens_per_rank_microbatch
    rope_theta = cfg.model.rope_theta
    grad_accum = cfg.window.grad_accum
    capacity_multiplier = cfg.mok.schedule_capacity_multiplier
    inner_steps = cfg.window.inner_steps

    for entry in manifest.phase_table:  # validated strictly increasing start_window
        if entry.start_window > window:
            break
        o = entry.overrides
        if o.data is not None:
            data = o.data
        if o.lr is not None:
            lr = o.lr
        if o.seq_len is not None:
            seq_len = o.seq_len
        if o.tokens_per_rank_microbatch is not None:
            tokens_per_rank_microbatch = o.tokens_per_rank_microbatch
        if o.rope_theta is not None:
            rope_theta = o.rope_theta
        if o.grad_accum is not None:
            grad_accum = o.grad_accum
        if o.capacity_multiplier is not None:
            capacity_multiplier = o.capacity_multiplier
        if o.inner_steps is not None:
            inner_steps = o.inner_steps

    return PhaseConfig(
        name=active.name,
        data=data,
        lr=lr,
        seq_len=seq_len,
        tokens_per_rank_microbatch=tokens_per_rank_microbatch,
        rope_theta=rope_theta,
        grad_accum=grad_accum,
        capacity_multiplier=capacity_multiplier,
        inner_steps=inner_steps,
        requires_restart=active.overrides.requires_restart,
    )


def lr_at(spec: LRSpec, global_inner_step: int, tokens_per_inner_step: int) -> float:
    """Closed-form LR at a global inner step (consensus value — identical on every node).

    - warmup (wsd_* kinds): linear 0 -> peak over warmup_steps; step s in
      [0, warmup) gets peak * (s + 1) / warmup_steps, hitting peak exactly at
      s = warmup_steps - 1 (no dead zero-LR step).
    - wsd_flat: peak thereafter (step C stable phase).
    - wsd_linear_decay: after warmup, linear peak -> 0 across
      decay_total_tokens, measured as (s - warmup) * tokens_per_inner_step;
      clamped at 0.0 once the token budget is consumed (step D anneal).
    - const: const_lr at every step, no warmup (step E long-context phase).

    Pure IEEE-754 double arithmetic in a fixed operation order — bitwise
    reproducible across nodes.
    """
    if global_inner_step < 0:
        raise ValueError(f"global_inner_step must be >= 0, got {global_inner_step}")
    if spec.kind == "const":
        return float(spec.const_lr)  # validated non-None by LRSpec

    s = global_inner_step
    warmup = spec.warmup_steps
    if warmup > 0 and s < warmup:
        return spec.peak_lr * (s + 1) / warmup
    if spec.kind == "wsd_flat":
        return spec.peak_lr

    # wsd_linear_decay
    if tokens_per_inner_step <= 0:
        raise ValueError(f"tokens_per_inner_step must be positive, got {tokens_per_inner_step}")
    decay_total = spec.decay_total_tokens  # validated non-None by LRSpec
    tokens_into_decay = (s - warmup) * tokens_per_inner_step
    frac = 1.0 - tokens_into_decay / decay_total
    return spec.peak_lr * max(0.0, frac)


def accum_at(window_cfg: WindowConfig, tokens_consumed: int) -> int:
    """Gradient-accumulation factor after `tokens_consumed` run tokens.

    Ramps accum_ramp_start -> grad_accum linearly over accum_ramp_tokens using
    exact integer floor arithmetic (monotonic non-decreasing, consensus value);
    grad_accum from the ramp end onward.
    """
    if tokens_consumed < 0:
        raise ValueError(f"tokens_consumed must be >= 0, got {tokens_consumed}")
    start = window_cfg.accum_ramp_start
    full = window_cfg.grad_accum
    ramp = window_cfg.accum_ramp_tokens
    if ramp <= 0 or tokens_consumed >= ramp:
        return full
    return start + (full - start) * tokens_consumed // ramp
