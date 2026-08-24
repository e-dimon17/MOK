"""Typed, frozen configuration schemas for the whole subnet.

Every hard constraint of the MoK kernel and of the window protocol is encoded
here as a pydantic validator, so an invalid run is rejected at config-load time
on every node — never at kernel-launch time on a $500k machine.

MoK API ground truth: mixture-of-kittens/mok/functional.py
  - get_workspace: num_local_tokens >= 512 and % 256; hidden % 256; topk in [1,255]
  - EP size in {4, 8, 16, 32, 64}; SM103 only
  - minibatch % 256; macrobatch % minibatch; comm SMs positive and even
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

EP_SIZES = (4, 8, 16, 32, 64)


class FrozenModel(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}


# --------------------------------------------------------------------------- #
# Model architecture
# --------------------------------------------------------------------------- #


class ModelConfig(FrozenModel):
    """The MoK-54B architecture (playbook §model spec). All MoK shape rules enforced."""

    num_layers: int = 32
    hidden_size: int = 4096
    num_q_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 65536
    seq_len: int = 4096
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5

    num_experts: int = 128
    top_k: int = 8
    intermediate_size: int = 1024
    ep_size: int = 8
    # First N layers use a dense SwiGLU FFN instead of MoE (DeepSeek-V3 keeps 3:
    # first-layer load balance converges slowest — arXiv:2401.06066 §4).
    num_dense_layers: int = 3
    dense_intermediate_size: int = 9216   # match activated MoE width (9 x 1024); % 256


    # Loss shaping (fixed reduction order in model/losses.py)
    aux_loss_coef: float = 1e-4          # backup aux loss under aux-free balancing
    router_z_coef: float = 1e-3
    output_z_coef: float = 1e-4
    bias_update_rate: float = 1e-3       # aux-free per-expert bias nudge (sign rule)

    routed_precision: Literal["mxfp8", "bf16"] = "mxfp8"

    @model_validator(mode="after")
    def _mok_constraints(self) -> ModelConfig:
        if self.hidden_size % 256 != 0:
            raise ValueError(f"hidden_size must be divisible by 256 (MoK), got {self.hidden_size}")
        if self.intermediate_size % 256 != 0:
            raise ValueError(
                f"intermediate_size must be divisible by 256 (MoK), got {self.intermediate_size}"
            )
        if not 1 <= self.top_k <= 255:
            raise ValueError(f"top_k must be in [1, 255] (MoK), got {self.top_k}")
        if self.ep_size not in EP_SIZES:
            raise ValueError(f"ep_size must be one of {EP_SIZES} (MoK), got {self.ep_size}")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({self.num_experts}) must divide evenly over ep_size ({self.ep_size})"
            )
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be a multiple of num_kv_heads (GQA)")
        if self.num_q_heads * self.head_dim != self.hidden_size:
            raise ValueError("num_q_heads * head_dim must equal hidden_size")
        if not 0 <= self.num_dense_layers < self.num_layers:
            raise ValueError("num_dense_layers must be in [0, num_layers)")
        if self.dense_intermediate_size % 256 != 0:
            raise ValueError("dense_intermediate_size must be divisible by 256")
        if self.seq_len % 256 != 0:
            raise ValueError("seq_len must be divisible by 256 (token-count alignment)")
        return self

    @property
    def num_local_experts(self) -> int:
        return self.num_experts // self.ep_size


class MoKRuntimeConfig(FrozenModel):
    """Mirror of mok.functional.MoKConfig plus our capacity-anneal policy knobs.

    The anneal itself is NEVER decided locally: it arrives as a manifest phase
    amendment (>= 2 windows ahead). These fields only describe the policy.
    """

    fwd_num_comm_sms: int = 36
    bwd_num_comm_sms: int = 36
    minibatch_size: int = 4096
    macrobatch_size: int = 131072
    schedule_capacity_multiplier: float = 1.0
    all_gather_top_experts_chunk_bytes: int = 2048

    capacity_anneal_to: float = 0.5
    capacity_anneal_util_threshold: float = 0.4

    @model_validator(mode="after")
    def _mok_constraints(self) -> MoKRuntimeConfig:
        for name in ("fwd_num_comm_sms", "bwd_num_comm_sms"):
            v = getattr(self, name)
            if v < 2 or v % 2 != 0:
                raise ValueError(f"{name} must be a positive even integer >= 2 (MoK), got {v}")
        if self.minibatch_size % 256 != 0:
            raise ValueError("minibatch_size must be divisible by 256 (MoK)")
        if self.macrobatch_size % self.minibatch_size != 0:
            raise ValueError("macrobatch_size must be a multiple of minibatch_size (MoK)")
        if not self.schedule_capacity_multiplier > 0.0:
            raise ValueError("schedule_capacity_multiplier must be positive")
        return self

    def to_mok(self):  # -> mok.functional.MoKConfig  (lazy: SM103-only wheel)
        from mok import functional  # noqa: PLC0415

        return functional.MoKConfig(
            fwd_num_comm_sms=self.fwd_num_comm_sms,
            bwd_num_comm_sms=self.bwd_num_comm_sms,
            minibatch_size=self.minibatch_size,
            macrobatch_size=self.macrobatch_size,
            schedule_capacity_multiplier=self.schedule_capacity_multiplier,
            all_gather_top_experts_chunk_bytes=self.all_gather_top_experts_chunk_bytes,
        )


# --------------------------------------------------------------------------- #
# Window protocol
# --------------------------------------------------------------------------- #


class WindowConfig(FrozenModel):
    """One outer-loop window (H=500 inner steps ~= 45 min)."""

    inner_steps: int = 500
    blocks_per_window: int = 225                 # ~45 min at 12 s/block
    upload_grace_s: int = 90                     # two-phase-commit gate after boundary
    tokens_per_rank_microbatch: int = 8192       # == MoK num_local_tokens; fixed per workspace
    grad_accum: int = 8
    accum_ramp_tokens: int = 50_000_000_000      # ramp 2 -> grad_accum over first 50B tokens
    accum_ramp_start: int = 2
    gather_peer_count: int = 20
    reserve_peer_count: int = 10
    checkpoint_every_windows: int = 10
    warmup_null_windows: int = 2                 # new miners train but do not submit

    @model_validator(mode="after")
    def _constraints(self) -> WindowConfig:
        t = self.tokens_per_rank_microbatch
        if t < 512 or t % 256 != 0:
            raise ValueError(f"tokens_per_rank_microbatch must be >=512 and %256 (MoK), got {t}")
        if self.inner_steps <= 0 or self.grad_accum <= 0:
            raise ValueError("inner_steps and grad_accum must be positive")
        if self.accum_ramp_start > self.grad_accum:
            raise ValueError("accum_ramp_start cannot exceed grad_accum")
        return self


class CompressionConfig(FrozenModel):
    """SparseLoCo (production-proven at 72B scale): chunked top-k + 2-bit quant + error feedback."""

    target_chunk: int = 64          # 2-D params chunk to 64x64 = 4096 elements
    topk: int = 64                  # per 4096-element chunk (~1.56% density)
    quant_bins: int = 4             # 2 bits of information per value
    quant_range_sigmas: float = 6.0
    ef_beta: float = 0.95
    use_dct: bool = False           # accepted for config parity; must stay off

    @model_validator(mode="after")
    def _constraints(self) -> CompressionConfig:
        chunk_elems = self.target_chunk * self.target_chunk
        if self.topk > chunk_elems:
            raise ValueError("topk cannot exceed elements per chunk")
        if self.quant_bins not in (2, 4, 8, 16, 256):
            raise ValueError("quant_bins must be a supported power of two")
        if chunk_elems > 4096:
            raise ValueError("chunk must be <= 4096 elements (12-bit index packing bound)")
        return self


class OuterOptConfig(FrozenModel):
    kind: Literal["nesterov", "sgd"] = "nesterov"   # fleet calibration pins the final choice
    lr: float = 0.7
    momentum: float = 0.9
    clip: Literal["median_norm", "none"] = "median_norm"


class LRSpec(FrozenModel):
    """Closed-form WSD segment — a pure function of the global inner step (no stateful scheduler)."""

    kind: Literal["wsd_flat", "wsd_linear_decay", "const"] = "wsd_flat"
    peak_lr: float = 3e-4
    warmup_steps: int = 2000
    # wsd_linear_decay: decay from peak to 0 across this many tokens (quality anneal)
    decay_total_tokens: int | None = None
    # const: fixed value (long-context phase)
    const_lr: float | None = None

    @model_validator(mode="after")
    def _constraints(self) -> LRSpec:
        if self.kind == "wsd_linear_decay" and not self.decay_total_tokens:
            raise ValueError("wsd_linear_decay requires decay_total_tokens")
        if self.kind == "const" and self.const_lr is None:
            raise ValueError("const requires const_lr")
        return self


class InnerOptConfig(FrozenModel):
    """Fresh AdamW per window (protocol decision #1 — makes a window a pure function)."""

    lr: LRSpec = LRSpec()
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    adam_reset_every_windows: int = 1   # 1 = reset each window; calibration may pin K=5 fallback


# --------------------------------------------------------------------------- #
# Trust layer
# --------------------------------------------------------------------------- #


class ScoringConfig(FrozenModel):
    eval_sequences: int = 8
    eval_lr_factor: float = 0.2
    binary_ema_alpha: float = 0.05
    binary_ema_threshold: float = 0.10
    binary_warmup_windows: int = 10
    openskill_beta: float = 7.0
    openskill_tau: float = 0.1
    sync_max_steps_behind: float = 2.0
    windows_per_weights: int = 3
    gather_share: float = 0.75          # emission share of top gather peers
    reserve_share: float = 0.25
    top_ratio: float = 2.0              # linear ramp top:bottom inside gather set
    reserve_decay: float = 0.5          # geometric decay inside reserve set
    overlap_threshold: float = 0.4      # pairwise top-k index overlap => copy suspicion


class AuditConfig(FrozenModel):
    probability: float = 0.075          # rho: per miner-window sample rate
    quorum: int = 2                     # mismatches from >= quorum distinct auditors => slash
    auditors: int = 3
    naughty_windows: int = 20           # exclusion span after a confirmed audit slash
    report_deadline_windows: int = 2


class RollbackConfig(FrozenModel):
    spike_threshold_nats: float = 0.15
    spike_baseline_windows: int = 5
    vote_supermajority: float = 2 / 3
    vote_window_span: int = 2
    activation_delay_windows: int = 1


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #


class BucketCreds(FrozenModel):
    """Read credentials for one participant's R2 bucket (committed on-chain)."""

    account_id: str
    bucket_name: str
    access_key_id: str
    secret_access_key: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class ChainConfig(FrozenModel):
    network: str = "finney"             # or "test" / a custom websocket endpoint
    netuid: int = 0                     # assigned at subnet registration
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    commit_retries: int = 3
    block_time_s: float = 12.0


class StorageConfig(FrozenModel):
    max_payload_bytes: int = 4 * 1024**3
    gather_timeout_s: float = 300.0
    download_chunk_bytes: int = 64 * 1024**2
    multipart_threshold_bytes: int = 256 * 1024**2


class DataConfig(FrozenModel):
    dataset_manifest_path: str = "manifest.json"   # local cache path of the dataset manifest
    prefetch_windows: int = 2
    shard_cache_dir: str = "~/.cache/mok-subnet/shards"
    shard_cache_max_bytes: int = 2 * 1024**4       # 2 TB NVMe budget


class TelemetryConfig(FrozenModel):
    wandb_project: str | None = None
    wandb_entity: str | None = None
    dashboard_bucket_key_prefix: str = "telemetry"
    log_level: str = "INFO"


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


class RunConfig(FrozenModel):
    """Everything a node needs to participate in the run. `config_hash` of the
    canonical serialization of this object goes into the on-chain manifest."""

    model: ModelConfig = ModelConfig()
    mok: MoKRuntimeConfig = MoKRuntimeConfig()
    window: WindowConfig = WindowConfig()
    compression: CompressionConfig = CompressionConfig()
    outer: OuterOptConfig = OuterOptConfig()
    inner: InnerOptConfig = InnerOptConfig()
    scoring: ScoringConfig = ScoringConfig()
    audit: AuditConfig = AuditConfig()
    rollback: RollbackConfig = RollbackConfig()
    chain: ChainConfig = ChainConfig()
    storage: StorageConfig = StorageConfig()
    data: DataConfig = DataConfig()
    telemetry: TelemetryConfig = TelemetryConfig()

    @property
    def tokens_per_inner_step(self) -> int:
        return self.window.tokens_per_rank_microbatch * self.model.ep_size * self.window.grad_accum

    @property
    def tokens_per_window_per_miner(self) -> int:
        return self.tokens_per_inner_step * self.window.inner_steps
