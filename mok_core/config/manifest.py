"""The on-chain run manifest: the single consensus object every node derives
its behavior from.

A window is a pure function of (θ_start, uid, window, manifest). Everything
time-varying — LR segment, data tree, sequence length, capacity multiplier,
grad-accum ramp — is resolved through the manifest's PHASE TABLE, and changes
arrive as signed AMENDMENTS effective >= 2 windows in the future. The anneal and context stages
are nothing but phase entries. Rollbacks append VOID RANGES (windows whose
gradients left the lineage) plus a reseed salt for the reassigned data.
"""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .canonical import canonical_hash
from .schemas import FrozenModel, LRSpec, RunConfig


class DatasetManifestRef(FrozenModel):
    """Pointer to the frozen dataset tree produced by dataprep."""

    name: str                            # e.g. "bulk", "anneal", "longdoc"
    merkle_root: str                     # hex blake2b-256 over sorted shard hashes
    num_shards: int
    shard_bytes: int                     # exact bytes per full shard (512 MiB)
    seq_len: int                         # tokens per packed sequence in this tree
    tokens_total: int
    tokenizer_hash: str


class PRFSpec(FrozenModel):
    """Identifies the assignment PRF so validators can reproduce any miner's data."""

    kind: Literal["blake2b_philox_v1"] = "blake2b_philox_v1"
    run_seed_hex: str                    # 32-byte hex; committed before first window
    reseed_salt_hex: str = ""            # appended by rollbacks; changes assignments of voided data

    @model_validator(mode="after")
    def _check(self) -> PRFSpec:
        if len(bytes.fromhex(self.run_seed_hex)) != 32:
            raise ValueError("run_seed must be exactly 32 bytes")
        return self


class PhaseOverrides(FrozenModel):
    """Knobs a phase entry may change. Anything not set inherits the RunConfig value."""

    data: str | None = None              # DatasetManifestRef.name to draw windows from
    lr: LRSpec | None = None
    seq_len: int | None = None
    tokens_per_rank_microbatch: int | None = None
    rope_theta: float | None = None
    grad_accum: int | None = None
    capacity_multiplier: float | None = None
    inner_steps: int | None = None
    requires_restart: bool = False       # workspace/dataloader shape change => clean relaunch


class PhaseEntry(FrozenModel):
    start_window: int
    name: str                            # "bulk" | "anneal" | "context16k" | ...
    overrides: PhaseOverrides = PhaseOverrides()


class VoidRange(FrozenModel):
    """Windows excluded from the lineage by a rollback vote (inclusive bounds)."""

    first_window: int
    last_window: int
    reseed_salt_hex: str

    @model_validator(mode="after")
    def _check(self) -> VoidRange:
        if self.last_window < self.first_window:
            raise ValueError("void range bounds out of order")
        return self


class Amendment(FrozenModel):
    """A manifest change committed on-chain, effective in the future only."""

    seq: int
    effective_window: int
    kind: Literal["phase", "void", "capacity", "prf_salt"]
    payload_hash: str                    # canonical_hash of the appended object
    committed_block: int


class RunManifest(FrozenModel):
    """The frozen description of the entire Stage-2 run."""

    spec_version: int
    run_id: str
    netuid: int
    network: str

    config_hash: str                     # canonical hash of RunConfig
    container_digest: str                # the one blessed image (sha256:...)
    mok_commit: str                      # pinned mixture-of-kittens commit
    tk_commit: str                       # pinned ThunderKittens submodule commit
    attention_backend: Literal["cudnn_det", "flash_det"]

    start_block: int                     # window 0 boundary block
    blocks_per_window: int

    prf: PRFSpec
    datasets: tuple[DatasetManifestRef, ...]
    init_checkpoint_hash: str            # state_root of the seed-42 initialization

    phase_table: tuple[PhaseEntry, ...] = (PhaseEntry(start_window=0, name="bulk"),)
    void_ranges: tuple[VoidRange, ...] = ()
    amendments: tuple[Amendment, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> RunManifest:
        starts = [p.start_window for p in self.phase_table]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("phase_table start_windows must be strictly increasing")
        if not self.phase_table or self.phase_table[0].start_window != 0:
            raise ValueError("phase_table must begin at window 0")
        names = {d.name for d in self.datasets}
        for p in self.phase_table:
            if p.overrides.data is not None and p.overrides.data not in names:
                raise ValueError(f"phase {p.name!r} references unknown dataset {p.overrides.data!r}")
        seqs = [a.seq for a in self.amendments]
        if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
            raise ValueError("amendment seq numbers must be strictly increasing")
        return self

    # ------------------------------------------------------------------ #

    def manifest_hash(self) -> str:
        return canonical_hash(self)

    def dataset(self, name: str) -> DatasetManifestRef:
        for d in self.datasets:
            if d.name == name:
                return d
        raise KeyError(f"dataset {name!r} not in manifest")

    def phase_at(self, window: int) -> PhaseEntry:
        """The phase entry active at `window` (subnet/core/phase.py builds the full
        PhaseConfig by merging these overrides into the RunConfig)."""
        active = self.phase_table[0]
        for entry in self.phase_table:
            if entry.start_window <= window:
                active = entry
            else:
                break
        return active

    def is_void(self, window: int) -> bool:
        return any(v.first_window <= window <= v.last_window for v in self.void_ranges)

    def with_amendment(
        self,
        *,
        kind: Literal["phase", "void", "capacity", "prf_salt"],
        effective_window: int,
        committed_block: int,
        phase: PhaseEntry | None = None,
        void: VoidRange | None = None,
        capacity_multiplier: float | None = None,
        reseed_salt_hex: str | None = None,
    ) -> RunManifest:
        """Pure append — returns the amended manifest. Callers must have checked
        the amendment is signed by the owner key and effective_window is at
        least 2 windows ahead of the chain head (enforced in chain layer)."""
        data = self.model_dump()
        obj: FrozenModel
        if kind == "phase":
            if phase is None:
                raise ValueError("phase amendment requires a PhaseEntry")
            data["phase_table"] = [*self.phase_table, phase]
            obj = phase
        elif kind == "void":
            if void is None:
                raise ValueError("void amendment requires a VoidRange")
            data["void_ranges"] = [*self.void_ranges, void]
            obj = void
        elif kind == "capacity":
            if capacity_multiplier is None:
                raise ValueError("capacity amendment requires capacity_multiplier")
            entry = PhaseEntry(
                start_window=effective_window,
                name=f"capacity_{capacity_multiplier}",
                overrides=PhaseOverrides(capacity_multiplier=capacity_multiplier),
            )
            data["phase_table"] = [*self.phase_table, entry]
            obj = entry
        elif kind == "prf_salt":
            if reseed_salt_hex is None:
                raise ValueError("prf_salt amendment requires reseed_salt_hex")
            new_prf = self.prf.model_copy(update={"reseed_salt_hex": reseed_salt_hex})
            data["prf"] = new_prf.model_dump()
            obj = new_prf
        else:  # pragma: no cover
            raise ValueError(kind)

        amendment = Amendment(
            seq=len(self.amendments),
            effective_window=effective_window,
            kind=kind,
            payload_hash=canonical_hash(obj),
            committed_block=committed_block,
        )
        data["amendments"] = [*self.amendments, amendment]
        return RunManifest.model_validate(data)


def build_manifest(
    cfg: RunConfig,
    *,
    run_id: str,
    start_block: int,
    container_digest: str,
    mok_commit: str,
    tk_commit: str,
    attention_backend: Literal["cudnn_det", "flash_det"],
    prf: PRFSpec,
    datasets: tuple[DatasetManifestRef, ...],
    init_checkpoint_hash: str,
    spec_version: int,
) -> RunManifest:
    from .canonical import config_hash  # noqa: PLC0415

    return RunManifest(
        spec_version=spec_version,
        run_id=run_id,
        netuid=cfg.chain.netuid,
        network=cfg.chain.network,
        config_hash=config_hash(cfg),
        container_digest=container_digest,
        mok_commit=mok_commit,
        tk_commit=tk_commit,
        attention_backend=attention_backend,
        start_block=start_block,
        blocks_per_window=cfg.window.blocks_per_window,
        prf=prf,
        datasets=datasets,
        init_checkpoint_hash=init_checkpoint_hash,
    )
