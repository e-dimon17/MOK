# Internal API contracts (generated from round-A implementation reports)
Source of truth is the code; this file orients round-B implementers.

## mok_core.data

```
mok_core.data.merkle:
  Proof = list[tuple[bytes, bool]]  # (sibling, sibling_is_right)
  class MerkleTree(leaves: list[bytes])  # 32-byte blake2b-256 leaves; duplicate-last padding; parent=blake2b(l||r); 1-leaf root=leaf
    .root -> bytes; .num_leaves -> int; .proof(index: int) -> Proof
    @staticmethod verify(root: bytes, leaf: bytes, proof: Proof) -> bool

mok_core.data.assignment:
  effective_run_seed(run_seed: bytes, reseed_salt: bytes) -> bytes  # blake2b(run_seed||"reseed.v1"||salt); identity if salt empty
  tokens_per_shard(dataset: DatasetManifestRef) -> int
  shard_ids(dataset: DatasetManifestRef, run_seed: bytes, uid: int, window: int, tokens_needed: int, *, reseed_salt: bytes = b"") -> list[int]  # philox(window_seed) permutation prefix, ceil cover
  sample_order(run_seed: bytes, uid: int, window: int, n: int, num_sequences_available: int, *, reseed_salt: bytes = b"") -> np.ndarray[int64]  # without replacement, "order.v1" domain
  sequences_per_window(*, tokens_per_rank_microbatch: int, grad_accum: int, inner_steps: int, ranks: int, seq_len: int) -> int

mok_core.data.shards:
  shard_leaf_hash(path) -> bytes  # blake2b-256 of file bytes
  shard_filename(leaf_hash: bytes) -> str  # "shard-<first16hex>.bin"
  class ShardReader(path, seq_len: int)  # numpy memmap '<u2'; context manager
    .num_sequences: int; .sequence(i: int) -> np.ndarray[uint16] (owned copy); .verify(expected_leaf_hash: bytes) -> bool; .close()
  class DatasetShardIndex(FrozenModel)  # name: str; seq_len: int; shard_hashes: list[str] (64-hex, shard-index order)
    .num_shards: int; .leaf(i: int) -> bytes; .merkle() -> MerkleTree
  verify_index_matches_ref(index: DatasetShardIndex, ref: DatasetManifestRef) -> None  # raises ValueError on any mismatch

mok_core.data.window_dataset:
  @dataclass(frozen=True, eq=False) class WindowBatchPlan
    fields: dataset_name, uid, window, rank, world_size, seq_len, tokens_per_rank_microbatch, grad_accum, inner_steps, shard_ids: tuple[int, ...], global_pairs: np.ndarray[N,2 int64], schedule: np.ndarray[steps,accum,seqs_per_mb,2] (both read-only)
    @classmethod build(manifest: RunManifest, *, run_seed: bytes, uid: int, window: int, rank: int, world_size: int, tokens_per_rank_microbatch: int, grad_accum: int, inner_steps: int, seq_len: int, dataset: str) -> WindowBatchPlan
    .microbatch_tokens(step: int, accum_idx: int, shard_lookup: Callable[[int], ShardReader]) -> torch.LongTensor  # [tokens_per_rank_microbatch]
    .microbatch_pairs(step, accum_idx) -> np.ndarray  # [seqs_per_mb, 2]
    .sample_digest() -> str  # rank-invariant receipt
    .total_sequences: int (all ranks); .sequences_per_rank: int; .seqs_per_microbatch: int

mok_core.data.download:
  FetchFn = Callable[[int], Awaitable[bytes]]
  class ShardCacheError(RuntimeError); class ShardVerificationError(ShardCacheError)
  class ShardCache(cache_dir, max_bytes: int, index: DatasetShardIndex)  # files under cache_dir/<index.name>/
    @classmethod from_config(cfg: DataConfig, index) -> ShardCache
    async .prefetch(shard_indices: Iterable[int], fetch_fn: FetchFn, *, concurrency: int = 4) -> dict[int, Path]
    .path_for(shard_idx: int) -> Path (touches LRU; FileNotFoundError if absent); .has(shard_idx) -> bool; .resident_bytes: int

All names re-exported from mok_core.data (__init__.py, __all__ of 17 names).
```
Notes:
- Consensus constants golden-pinned (SPEC_VERSION-bound): 5-leaf merkle root 28ccae79...8a4b98; shard_ids(uid=7,w=42)=[915,629,732,976,217,633] and salted variant; sample_order=[923,383,744,385,670,797,778,68]; effective_run_seed(b'salt')=42cd7d5e...; WindowBatchPlan sample_digest 449915d4...14d2.
- New consensus rules introduced here (documented in docstrings): reseed salt folds as blake2b(run_seed||'reseed.v1'||salt); sample_order seeds from an 'order.v1' blake2b domain (same byte layout as seeding.window_seed) so the shard draw and sequence-order draw are independent Philox streams; sample_digest wire = 'samples.v1' || n(le64) || sorted-(shard,seq) pairs as le64 pairs.
- Two pre-existing ruff findings in other agents' files (tests/unit/test_compress.py I001, tests/unit/test_window_state.py C416) were left untouched per ownership rules.

## dataprep

```
dataprep/pipeline/tokenizer_train.py:
  PAD_TOKEN/PAD_ID=0, BOS_TOKEN/BOS_ID=1, EOS_TOKEN/EOS_ID=2 (consensus constants)
  class TokenizerConfig(FrozenModel): vocab_size=65536, min_frequency=2, sample_chars_total=400e9, pad/bos/eos tokens
  load_tokenizer_config(path) -> TokenizerConfig
  tokenizer_file_hash(path) -> str  # hex blake2b-256, the manifest tokenizer_hash
  @dataclass TrainedTokenizer(path: Path, tokenizer_hash: str)
  train_tokenizer(texts: Iterable[str], out_path, cfg: TokenizerConfig | None = None) -> TrainedTokenizer

dataprep/pipeline/download.py:
  class SourceSpec(FrozenModel): name, hf_path, hf_name?, split, text_column, weight, max_tokens, score_column?/min_score?
  class CorpusConfig(FrozenModel): name, seq_len=4096, chars_per_token=4.0, dedup_order (permutation), sources; .source(name), .dedup_sequence(), .char_budget(spec)
  load_corpus_config(path) -> CorpusConfig
  class SpoolState(FrozenModel): source, docs, chars, parts, complete
  spool_documents(docs, spool_root, source, *, char_budget=None, part_docs=100_000) -> SpoolState  # resumable, atomic parts
  load_spool_state(spool_root, source) -> SpoolState
  iter_source_documents(spool_root, source) -> Iterator[str]
  download_source(cfg, spec, spool_root, *, doc_iter=None, part_docs=100_000) -> SpoolState  # doc_iter injects; default lazy HF stream
  download_corpus(cfg, spool_root, *, only=None, part_docs=100_000) -> dict[str, SpoolState]

dataprep/pipeline/dedup.py:
  SHINGLE_K=13, NUM_PERM=128, THRESHOLD=0.8
  shingles(text, k=13) -> set[bytes]; minhash_of(text, *, num_perm, shingle_k) -> MinHash
  @dataclass DedupStats(kept/dropped/empty per-source dicts, total_kept, total_dropped)
  dedup_documents(sources: Iterable[tuple[str, Iterable[str]]], *, threshold, num_perm, shingle_k, stats=None) -> Iterator[tuple[str, str]]

dataprep/pipeline/tokenize_pack.py:
  MAX_TOKEN_ID=65535
  pack_documents(token_iters, seq_len=4096, eos_id=2) -> Iterator[np.ndarray uint16]  # EOS-join, split long docs, drop partial tail
  chunk_token_stream(tokens, seq_len) / chunk_token_arrays(arrays, seq_len) -> Iterator[np.ndarray]
  encode_documents(texts, tokenizer_path, *, batch_size=256) -> Iterator[list[int]]  # lazy tokenizers
  write_token_stream(token_iters, out_path, *, eos_id=2, flush_tokens) -> int  # flat <u2 file (tokenize->shard interchange)
  iter_token_file_arrays(paths, *, block_tokens) -> Iterator[np.ndarray]

dataprep/pipeline/shard_writer.py:
  FULL_SHARD_SEQUENCES=65536; SHARD_METAS_FILENAME="shards.json"
  @dataclass ShardMeta(path: Path, hash_hex: str, num_sequences: int)
  write_shards(seq_iter, out_dir, *, shard_sequences=65536, seq_len=4096) -> list[ShardMeta]  # incremental hash, rename to shard-<hash16>.bin
  save_shard_metas(metas, path) / load_shard_metas(path)

dataprep/pipeline/build_manifest.py:
  SHARD_INDEX_FILENAME="shard_index.json", MANIFEST_FILENAME="manifest.json", TOKENIZER_FILENAME="tokenizer.json"
  build_dataset_manifest(metas, *, name, seq_len, tokenizer_hash, out_dir, shard_sequences=65536) -> tuple[DatasetShardIndex, DatasetManifestRef]
  load_shard_index(path) / load_manifest_ref(path)

dataprep/pipeline/upload.py:
  @dataclass UploadReport(uploaded, skipped, bytes_sent)
  resolve_endpoint(endpoint_url=None) -> str  # falls back to https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com
  dataset_files(data_dir) -> list[Path]
  async upload_dataset(data_dir, *, prefix, bucket=None, endpoint_url=None, access_key_id=None, secret_access_key=None, region="auto", concurrency=4) -> UploadReport  # env fallbacks R2_BUCKET_NAME / R2_WRITE_ACCESS_KEY_ID / R2_WRITE_SECRET_ACCESS_KEY
  upload_dataset_sync(data_dir, **kwargs) -> UploadReport

dataprep/pipeline/verify.py:
  @dataclass VerifyReport(dataset, merkle_root, num_shards, shards_hashed, tokens_total, failures; .ok)
  verify_local(data_dir, *, sample=None, seed=0) -> VerifyReport
  verify_sample(data_dir, n, *, seed=0) -> VerifyReport

dataprep/cli.py:
  main(argv=None) -> int; build_parser() -> ArgumentParser  # console script mok-data (already wired in pyproject)
  subcommands: tokenizer | download | dedup | tokenize | shard | manifest | upload | verify
  top-level: --smoke TEXT_FILE... --out DIR [--seq-len 128 --shard-sequences 32 --vocab-size 512]
```
Notes:
- Golden vectors pinned in tests/unit/test_dataprep_manifest.py (shard leaf aeb002c6... and merkle root d1d9efe6... on a fixed uint16 arange fixture) and tests/unit/test_dataprep_shard_writer.py (65536*4096*2 == 536870912 == 512 MiB), each marked '# consensus constant — change requires SPEC_VERSION bump'.

## compress

```
subnet/core/compress.py:
  SCALE_FLOOR: float (fp32-rounded 1e-12), SCALE_CEIL: float (1e4)
  packed_nbytes_12bit(count: int) -> int; packed_nbytes_2bit(count: int) -> int
  pack_12bit_indices(indices: Tensor[int, values in [0,4096)]) -> Tensor[uint8]  # 2 idx/3 bytes; odd tail -> 2 bytes
  unpack_12bit_indices(packed: Tensor[uint8], count: int) -> Tensor[int64]
  pack_2bit_values(codes: Tensor[uint8, values in [0,4)]) -> Tensor[uint8]  # 4 codes/byte, LSB-first
  unpack_2bit_values(packed: Tensor[uint8], count: int) -> Tensor[uint8]
  ChunkGeometry (frozen dataclass): orig_shape, mode('flat'|'grid'), rows, cols, pad_rows, pad_cols, n_chunks, chunk_elems; .numel, .padded_numel
  chunk_geometry(shape: tuple[int,...]|torch.Size, target_chunk: int) -> ChunkGeometry
  ChunkingTransformer(param_shapes: Mapping[str, Size|tuple], target_chunk: int = 64, *, use_dct: bool = False)
    .encode(name: str, tensor: Tensor) -> Tensor[n_chunks, chunk_elems]  # zero-pads partial chunks
    .decode(name: str, chunked: Tensor) -> Tensor[orig_shape]
    .geometry(name: str) -> ChunkGeometry; .names: tuple[str, ...]; .use_dct (always False; True raises)
  Quantizer(bins: int = 4, range_sigmas: float = 6.0)
    .quantize(vals: Tensor) -> (codes: Tensor[uint8], qparams: dict{shift: float, scale: float, lookup: Tensor[fp32, bins]})
    .dequantize(codes: Tensor[uint8], qparams: Mapping) -> Tensor[fp32]
  CompressedTensor (dataclass, eq=False): idxs_packed, codes_packed (uint8), qparams, n_chunks, chunk_elems, orig_shape: tuple, topk; .n_values
  TopKCompressor(transformer: ChunkingTransformer, quantizer: Quantizer, topk: int)  # quantizer.bins must be <= 4
    .compress(name: str, tensor: Tensor) -> CompressedTensor  # per-chunk top-k by |value|, indices sorted ascending (canonical)
    .decompress(name: str, ct: CompressedTensor) -> Tensor[fp32, orig_shape]
    .effective_topk(name: str) -> int  # min(topk, chunk_elems)
  ErrorFeedback(beta: float = 0.95)
    .update(name: str, delta: Tensor) -> Tensor  # m = beta*m + delta (cpu fp32), returns clone
    .subtract_transmitted(name: str, decompressed: Tensor) -> None
    .reset() -> None; .state_dict() -> dict[str, Tensor]; .load_state_dict(state: Mapping) -> None
    .merkle_root() -> str  # hex; 1 MiB chunk leaves via mok_core.data.merkle.MerkleTree
    .buffer(name: str) -> Tensor; .names: tuple[str, ...]

subnet/core/payload.py:
  MAGIC = b'MOKP'; WIRE_VERSION = 1; ZSTD_LEVEL = 3
  PayloadError(ValueError)
  assign_owned_params(names: Iterable[str], rank: int, world_size: int, is_expert_local: Callable[[str], bool]) -> set[str]
  PayloadMeta (frozen dataclass): sample_digest: str, sample_count: int, theta_end_hash: str, state_root: str, global_step: int, spec_version: int
  WindowPayload (dataclass, eq=False): uid: int, window: int, compressed: dict[str, CompressedTensor], dense: dict[str, Tensor[fp32]], metadata: PayloadMeta
  serialize(payload: WindowPayload) -> bytes  # deterministic: canonical JSON header + ordered blobs, zstd level-3 single-thread
  deserialize(data: bytes, *, max_bytes: int, max_decompressed_bytes: int | None = None) -> WindowPayload  # all bounds checked pre-allocation
  canonical_payload_hash(payload: WindowPayload) -> str  # hex blake2b-256 of serialized bytes
  validate_structure(payload, expected_param_shapes: Mapping[str, tuple], expected_dense: Mapping[str, tuple] | set[str], topk: int, *, target_chunk: int = 64) -> None
```
Notes:
- Wire format v1 (consensus surface): b'MOKP' + u8 version + zstd(level 3, no checksum, single thread) over [u32le header_len | canonical JSON header | ordered blobs]. deserialize enforces byte-for-byte canonical header re-serialization, declared zstd content size <= budget (default 4*max_bytes) before decompression, exact blob-length accounting before any tensor allocation, and rejects bool-as-int/unsorted names/pad-bit garbage.
- Golden consensus constants pinned in tests: 12-bit pack '001000ff0f80' (even) / '001000ff0f800700' (odd), 2-bit pack '9303', canonical_payload_hash 'aee9446706d2c5804c780ee0da1e3463d663b5d903c1949fa28c8ac6f9360414'. The payload hash binds the zstd frame bytes — zstandard is pinned at 0.25.0 in the venv/container; a zstd library bump that changes level-3 output is a SPEC_VERSION bump.

## outer

```
subnet/core/window_state.py:
- state_root(named_params: Iterable[tuple[str, Tensor]]) -> str  # delegates to mok_core.determinism.hash_named_tensors
- collect_digests(named: Iterable[tuple[str, Tensor]]) -> dict[str, bytes]
- divergence_report(expected_digests: Mapping[str, bytes], actual_digests: Mapping[str, bytes], limit: int = 16) -> list[DivergenceRecord]
- rank_parallel_state_root(named: Iterable[tuple[str, Tensor]], gather: GatherFn) -> str | None  # root on rank 0, None elsewhere; GatherFn = Callable[[list[tuple[str, bytes]]], list[list[tuple[str, bytes]]] | None]; raises ValueError on duplicate tensor ownership

subnet/core/outer_opt.py:
- deterministic_segment_mean(indices: list[Tensor], values: list[Tensor], numel: int) -> Tensor  # fp32 CPU dense; fp64 accumulation via stable-sort + CPU bincount, no atomics
- median_norm_clip_factors(norms: Tensor) -> Tensor  # fp32, factor_i = min(1, median/norm_i), zero-norm -> 1.0, even count -> lower median
- @dataclass(frozen=True) OuterReport(global_grad_l2: float, per_param_l2: dict[str, float], applied_peers: int)
- class ReplicatedOuterStep(cfg: OuterOptConfig, param_shapes: dict[str, torch.Size]):
    .apply(named_params: dict[str, Tensor], peer_sparse: dict[str, list[tuple[Tensor, Tensor]]], dense_contribs: dict[str, list[Tensor]], peer_norms: dict[str, Tensor]) -> OuterReport  # @torch.no_grad, in-place, cert-UID peer order, sorted-name param order
    .state_dict() -> dict[str, Tensor]  # cloned fp32 momentum
    .load_state_dict(state: Mapping[str, Tensor]) -> None  # strict names+shapes
    .param_names -> tuple[str, ...]

subnet/core/phase.py:
- @dataclass(frozen=True) PhaseConfig(name, data, lr: LRSpec, seq_len, tokens_per_rank_microbatch, rope_theta, grad_accum, capacity_multiplier, inner_steps, requires_restart)
- resolve_phase(manifest: RunManifest, cfg: RunConfig, window: int) -> PhaseConfig  # cumulative fold of all entries with start_window <= window
- lr_at(spec: LRSpec, global_inner_step: int, tokens_per_inner_step: int) -> float  # closed-form warmup/wsd_flat/wsd_linear_decay/const
- accum_at(window_cfg: WindowConfig, tokens_consumed: int) -> int  # exact integer floor ramp, monotonic

subnet/core/certificate.py:
- class CommitLike(Protocol): uid: int; payload_hash: str; in_gate: bool; valid: bool
- class WindowCertificate(FrozenModel): window: int; included_uids: tuple[int, ...]; payload_hashes: dict[int, str]; theta_start_root: str; leader_uid: int; leader_sig: str = ""
- certificate_message(cert: WindowCertificate) -> bytes  # raw 32-byte blake2b-256 canonical hash of the 5 unsigned fields (UNSIGNED_FIELDS)
- build_certificate(window, commits: Mapping[int, CommitLike], scores: Mapping[int, float], gather_count: int, reserve_count: int, theta_start_root: str, leader_uid: int, sign: Callable[[bytes], bytes]) -> WindowCertificate
- verify_certificate(cert, chain_commits: Mapping[int, CommitLike], verify_sig: Callable[[bytes, bytes], bool]) -> bool  # never raises; False on any defect
```
Notes:
- lr_at warmup convention: step s in [0, warmup) gets peak*(s+1)/warmup_steps — peak is reached at s = warmup_steps-1 and there is no dead zero-LR step 0. wsd_linear_decay measures decay progress in tokens: (s - warmup_steps)*tokens_per_inner_step against decay_total_tokens, clamped at 0.0. Anneal-phase LRSpec overlays must therefore set warmup_steps to the global step where decay begins. kind='const' ignores warmup entirely. Golden values pinned as consensus constants in test_phase.py.
- certificate_message wire rule (consensus constant, golden vector 1733d1d9...859e pinned in test_certificate.py): pydantic json-mode dump of {window, included_uids, payload_hashes, theta_start_root, leader_uid} (int dict keys become strings, tuples become lists) -> canonical_hash -> raw 32 digest bytes. Signature is over those 32 bytes.
- No compress.py/payload.py coupling: outer_opt operates on plain (indices, values) tensor pairs; the CompressedTensor contract is consumed upstream by exchange/window_runner which should dequantize into these pairs in certificate-UID order.

## trust

```
subnet/core/scoring.py:
  RANDOM_POOL_UID: int = 2**32 - 1
  gradient_score(loss_before: float, loss_after: float) -> float
  binary_indicator(improvement_own: float, improvement_random: float) -> int
  class BinaryEMA(alpha, threshold, warmup_windows): update(uid, indicator, window) -> float; value(uid) -> float; passes(uid, window) -> bool; reset(uid); state_dict()/load_state_dict(state)
  class OpenSkillBook(beta, tau): rate_window(scores: dict[int, float]) -> None; ordinal(uid) -> float; mu_sigma(uid) -> tuple[float, float]; reset(uid); state_dict()/load_state_dict(state)
  sync_score(avg_steps_behind: float, max_behind: float = 3.0) -> float
  final_score(uid: int, book: OpenSkillBook, bma: BinaryEMA, sync: float) -> float
  compute_weights(final_scores: dict[int, float], cfg: ScoringConfig, *, gather_count: int = 20, reserve_count: int = 10) -> dict[int, float]
  PlanFactory = Callable[[RunManifest, bytes, int, int, PhaseConfig], Any]
  class EvalPools(plan_factory: PlanFactory | None = None, *, world_size: int = 8):
    own_pool(manifest, run_seed, uid, window, phase: PhaseConfig, n_sequences, block_hash) -> list[tuple[int, int]]
    random_pool(manifest, run_seed, window, phase: PhaseConfig, n_sequences, block_hash) -> list[tuple[int, int]]

subnet/core/overlap.py:
  @dataclass OverlapPair(uid_a, uid_b, overlap: float)
  @dataclass OverlapReport(pairs: list[OverlapPair], pairs_checked: int, mean_overlap: float)
  @dataclass OverlapSeverity(level: str, multiplier: float, naughty: bool); __bool__ = any sanction
  index_overlap_report(peer_indices: dict[int, dict[str, LongTensor]], threshold: float) -> OverlapReport
  determine_offender(pair: OverlapPair, upload_ts: dict[int, float]) -> int
  severity(overlap: float) -> OverlapSeverity

subnet/core/slashing.py:
  MISSING_GRADIENT_LADDER = (0.75, 0.5, 0.0); SYNC_BEHIND_MULTIPLIER = INACTIVITY_MULTIPLIER = 0.75
  Protocol AuditReportLike(auditor_uid: int, match: bool)
  @dataclass SlashRecord(uid, window, reason, multiplier, naughty_until: int | None, detail); to_json()
  class SlashLedger(audit_cfg: AuditConfig | None = None, *, inactivity_reset_windows: int = 25, overlap_naughty_windows: int | None = None):
    missing_gradient(uid, window) -> SlashRecord; gradient_received(uid, window); invalid_payload(uid, window) -> SlashRecord; sync_behind(uid, window) -> SlashRecord; inactivity(uid, window) -> SlashRecord; overlap(uid, window, multiplier, naughty) -> SlashRecord; audit_verdicts(uid, window, reports: list[AuditReportLike]) -> SlashRecord | None; is_naughty(uid, window) -> bool; apply(uid, base_score, window) -> float; reset(uid); records: list[SlashRecord]; state_dict()/load_state_dict(state)

subnet/core/rollback.py:
  rollback_salt(run_seed: bytes, target_window: int) -> str  # blake2b-256(run_seed||b'rollback'||target le64) hex
  class SpikeDetector(threshold_nats, baseline_windows): observe(window, probe_loss) -> bool; reset()
  class RollbackVote(FrozenModel): voter_uid, stake, target_window, window_cast, sig
  class RollbackDecision(FrozenModel): target_window, void: VoidRange, reseed_salt_hex
  enum RollbackState: NORMAL/ALERTED/VOTING/PENDING (StrEnum)
  class RollbackStateMachine(cfg: RollbackConfig, run_seed: bytes):
    on_alert(window, checkpoint_window) -> bool; add_vote(vote, total_stake) -> bool; tick(window); maybe_activate(window) -> RollbackDecision | None; state; target_window; activation_window; yes_stake; history
```
Notes:
- Eval-pool sampling uses first-k-distinct rejection sampling over Philox integers (not Generator.choice) — memory-flat for huge dataset trees and immune to numpy choice-algorithm changes; it is a consensus function, golden-pinned in tests.
- random_pool keeps the own_pool signature minus uid (RANDOM_POOL_UID=2**32-1 sentinel replaces it in eval_seed); window is accepted for parity but the pool keys on block_hash, matching the seeding contract.
- Golden vectors pinned (consensus constants, SPEC_VERSION-bound): own_pool draw [(4,7),(7,5),(0,0),(1,9)] and random_pool draw [(3,12),(0,10),(1,7),(1,0),(3,0),(0,9)] for run_seed=0x00*32, block_hash=0x01*32; rollback_salt(0x00*32, 42) = 8468c1a72c7325cc485d05af14ebdf75c1f6de57692c1f247adcfffa7f6e724d.

## chain

```
mok_core.chain (all re-exported from package __init__):

windows.py (pure):
- window_of_block(block: int, start_block: int, blocks_per_window: int) -> int  # >=0; ValueError if block < start_block or bad params
- boundary_block(window: int, start_block: int, blocks_per_window: int) -> int  # first block of window; ValueError on window < 0
- blocks_into_window(block: int, start_block: int, blocks_per_window: int) -> int  # offset in [0, bpw)
- gate_deadline_s(boundary_block_ts: float, grace_s: float) -> float
- is_in_gate(object_ts: float, boundary_ts: float, grace_s: float) -> bool  # half-open [boundary_ts, boundary_ts+grace_s)

schemas.py (pydantic FrozenModels; each has .encode() -> str and classmethod .decode(wire: str), ValueError on garbage):
- WIRE_VERSION = 1; MAX_COMMITMENT_BYTES = 256; TAG_BUCKET/TAG_WINDOW/TAG_MANIFEST/TAG_VOTE ('MOKB'/'MOKW'/'MOKM'/'MOKV')
- BucketCommit(version: int = 1, creds: BucketCreds)  # WIRE_LEN 197
- WindowCommit(version: int = 1, window: int, payload_hash: hex64, state_root: hex64, theta_end_hash: hex64)  # WIRE_LEN 210
- ManifestCommit(version: int = 1, manifest_hash: hex64)  # WIRE_LEN 70
- VoteCommit(version: int = 1, kind: Literal['rollback','amendment'], target: int, payload_hash: hex64)  # WIRE_LEN 83
- Commitment = BucketCommit | WindowCommit | ManifestCommit | VoteCommit
- decode_commitment(wire: str) -> Commitment  # dispatch on 4-char tag

client.py:
- U16_MAX = 65535
- class ChainError(RuntimeError)
- class WindowSchedule(Protocol): start_block: int; blocks_per_window: int  # RunManifest satisfies it
- normalize_weights_u16(weights: Mapping[int, float]) -> tuple[list[int], list[int]]  # pure; (uids sorted, u16 vals), max->65535, drops <=0/non-finite/rounds-to-0
- class ChainClient(cfg: ChainConfig, *, wallet=None, subtensor_factory: Callable[[], Any] | None = None, keypair_factory: Callable[[str], Any] | None = None, backoff_base_s: float = 1.0)
  properties: subtensor, wallet, metagraph (all lazy; real `bittensor` imported only when nothing injected)
  sync_metagraph() -> None
  commit(data: str) -> None  # cfg.commit_retries attempts, exponential backoff backoff_base_s * 2**attempt; ChainError on exhaustion
  get_commitment(uid: int) -> str | None; get_all_commitments(block: int | None = None) -> dict[int, str]  # substrate query_map Commitments/CommitmentOf
  commit_bucket(creds: BucketCreds); get_bucket(uid) -> BucketCreds | None; get_all_buckets() -> dict[int, BucketCreds]; ensure_bucket_committed(creds) -> bool
  commit_window(WindowCommit); get_window_commits(window: int, uids: Iterable[int] | None = None) -> dict[int, WindowCommit]
  commit_manifest_hash(manifest_hash: str); get_manifest_hash(owner_uid: int) -> str | None
  commit_vote(VoteCommit); get_votes(kind=None, target=None, uids=None) -> dict[int, VoteCommit]
  set_weights(weights: dict[int, float], *, wait_for_inclusion: bool = False) -> bool
  current_block(*, force: bool = False) -> int  # cached ~cfg.block_time_s
  block_hash(block: int) -> bytes; block_timestamp(block: int) -> float  # substrate Timestamp.Now at block hash, ms -> s
  current_window(schedule: WindowSchedule) -> int; async wait_for_window(window: int, schedule: WindowSchedule, poll_s: float = 12.0) -> int
  uids() -> list[int]; hotkeys() -> list[str]; stakes() -> dict[int, float]; hotkey_of(uid) -> str | None; uid_of_hotkey(hotkey) -> int | None; my_uid() -> int | None
  sign(data: bytes) -> bytes; verify(hotkey_ss58: str, data: bytes, signature: bytes) -> bool
```
Notes:
- Wire format (consensus, golden-pinned): 4-char kind tag + 2-digit zero-padded version prefix, then fixed-width fields. BucketCommit: account_id(32) bucket_name(63) access_key_id(32) secret_access_key(64), variable fields right-padded with '~' (forbidden inside fields) = 197 chars. WindowCommit: window as 12-digit zero-padded decimal + 3x hex64 = 210 chars (asserted <= MAX_COMMITMENT_BYTES=256, including worst-case window 10^12-1). ManifestCommit 70, VoteCommit kind char R/A + 12-digit target + hex64 = 83. All hex fields must be 64 LOWERCASE hex chars (uppercase rejected). Golden strings pinned in tests/unit/test_chain_schemas.py with '# consensus constant — change requires SPEC_VERSION bump'.

## storage

```
mok_core.storage (all re-exported from package __init__):

keys.py (pure, consensus wire formats — golden-vector pinned):
  MANIFEST_KEY: str = "manifest.json"; MAX_WINDOW = 10**8; MAX_UID = 10**5
  KeyFormatError(ValueError)
  payload_key(window: int, uid: int, version: str) -> str        # payloads/w{w:08d}/uid{u:05d}-v{ver}.zst
  checkpoint_key(window: int, kind: str) -> str                  # checkpoints/w{w:08d}/{kind}
  shard_key(hash16: str) -> str                                  # shards/{16-lower-hex}.bin
  telemetry_key(window: int, uid: int) -> str                    # telemetry/w{w:08d}/uid{u:05d}.json
  certificate_key(window: int) -> str                            # certificates/w{w:08d}.json
  aggregator_key(window: int) -> str                             # aggregators/w{w:08d}.zst
  audit_report_key(window: int, auditor_uid: int, miner_uid: int) -> str  # audits/w{w:08d}/auditor{a:05d}-miner{m:05d}.json
  manifest_key() -> str
  attest_key(uid: int, nonce: str) -> str                        # attest/uid{u:05d}-{8..64 lower hex}.json
  payload_prefix(window) / telemetry_prefix(window) / audit_prefix(window) -> str   # for list_keys
  parse_payload_key/parse_checkpoint_key/parse_shard_key/parse_telemetry_key/parse_certificate_key/
  parse_aggregator_key/parse_audit_report_key/parse_attest_key(key: str) -> NamedTuple ref
  Refs (NamedTuples): PayloadRef(window, uid, version), CheckpointRef(window, kind), ShardRef(hash16),
  TelemetryRef(window, uid), CertificateRef(window), AggregatorRef(window),
  AuditReportRef(window, auditor_uid, miner_uid), AttestRef(uid, nonce)

client.py:
  StorageError(Exception); ObjectMissingError; ObjectTooLargeError; IntegrityError  (all subclass StorageError)
  @dataclass(frozen=True) GatherResult: ok: OrderedDict[int, bytes] (uid-ascending), failed: dict[int, str]; .uids -> list[int]
  StorageClient(creds: BucketCreds, cfg: StorageConfig, *, session_factory: Callable[[], Any] | None = None,
                endpoint_override: str | None = None, region_name: str = "auto", retry_attempts: int = 3,
                retry_base_delay_s: float = 0.5, part_concurrency: int = 8)
    async context manager (__aenter__/__aexit__) + aclose()
    async put_bytes(key: str, data: bytes) -> None                                  # own bucket
    async get_bytes(bucket: BucketCreds, key: str, *, expected_hash: str | None = None,
                    max_bytes: int | None = None) -> bytes                          # None -> cfg.max_payload_bytes
    async upload_file(key: str, path: str | os.PathLike[str]) -> None               # multipart above cfg.multipart_threshold_bytes
    async download_file(bucket, key, path, *, expected_hash: str | None = None,
                        max_bytes: int | None = None) -> None                       # ranged/resumable .part -> atomic os.replace
    async object_timestamp(bucket, key) -> float                                    # HEAD LastModified epoch; ObjectMissingError if absent
    async object_exists(bucket, key) -> bool
    async list_keys(bucket, prefix: str) -> list[str]                               # sorted, paginated
    async gather_bytes(peers: Mapping[int, BucketCreds], key_fn: Callable[[int], str], *,
                       expected_hashes: Mapping[int, str], deadline_s: float,
                       max_bytes: int | None = None) -> GatherResult                # deadline_s = per-fetch timeout
```
Notes:
- Key layouts are golden-vector pinned as consensus constants in test_storage_keys.py; fixed-width w{:08d}/uid{:05d} makes key sort order == numeric window order (asserted). checkpoint_key kind and payload version are validated slugs ([0-9A-Za-z._-]+, no slashes) so all keys parse round-trip unambiguously.
- gather_bytes wraps each get_bytes in asyncio.wait_for(deadline_s) - deadline_s is a PER-FETCH timeout, matching the task spec. Results are assembled by iterating sorted(peers) after asyncio.gather(return_exceptions=True), so ok is uid-ascending regardless of completion order (consensus-bearing for the outer merge). Failure reason strings are prefix-classified: 'timeout', 'missing: ...', 'integrity: ...', 'too_large: ...', 'error: <Type>: ...'. A uid absent from expected_hashes is fetched unverified rather than failed.
- aioboto3/aiobotocore/botocore/boto3/aiofiles are all imported function-locally; asserted via a fresh-interpreter test that importing mok_core.storage loads none of them. Note mok_core.determinism.hashing imports torch at module level (pre-existing, allowed per ground rules).

## model

```
mok_core.model (all re-exported from __init__):

attention.py:
  resolve_attention_backend() -> Literal['cudnn_det','flash_det']   # env MOK_ATTENTION_BACKEND override (ATTENTION_BACKEND_ENV)
  sdpa_backend(backend: str|None = None) -> ctx mgr yielding pinned name ('math' on CPU)
  rope_cos_sin(seq_len: int, theta: float, device, dtype, head_dim: int) -> (cos, sin)  # cached per (S, theta, device, dtype)
  apply_rope(x [B,S,n,hd], cos, sin) -> Tensor
  CausalSelfAttention(cfg: ModelConfig, rope_theta: float|None = None, dtype=torch.bfloat16); forward(x [B,S,H]) -> [B,S,H]

router.py:
  RouterOutput(weights fp32 [T,topk], experts int64 [T,topk], router_logits fp32 [T,E], load int64 [E])
  Router(cfg); forward(x [T,H]) -> RouterOutput; update_balance_bias_(load: Tensor, rate: float) -> None  # b -= rate*sign(load-mean)
  # persistent fp32 buffer 'balance_bias' [E] — selection-only bias; in state_root domain

losses.py:
  z_loss(logits) / router_z_loss(router_logits) -> fp32 scalar
  aux_load_loss(router_logits [T,E], load [E], top_k: int) -> fp32 scalar    # E * sum(f_i * P_i), ==1 at balance
  loss_head(lm_logits [B,S,V], targets int64 [B,S], router_stats: Sequence[RouterOutput], cfg) -> LossOutput(total, ce, aux, router_z, output_z)

moe.py:
  is_expert_local(name: str) -> bool           # marker EXPERT_MARKER = '.routed_'
  MoKMoELayer(cfg, layer_idx: int, mok_runtime: MoKRuntimeConfig|None = None, dtype=torch.bfloat16)
    .forward(x bf16 [T,H], backend: 'mok'|'reference') -> (y bf16 [T,H], RouterOutput)
    params: shared_gate/up/down, routed_gate/up/down (bf16 masters; aliases w_shared_*/w_routed_* as properties), .router, .quant_cache
  _MoKFunction (internal autograd.Function; single-use fwd_ctx guard; accumulates 6 weight grads into param.grad, returns d_x + d_router_weights)

quant.py:
  QuantizedRoutedWeights(12 tensors).forward_args() -> 3x(fp8,sc); .backward_args() -> (4-tuple, 4-tuple, (t_fp8,t_sc)); .tensors()
  MXFP8WeightManager(layers: Sequence[MoKMoELayer]); .requantize_(layer); .requantize_all_(); .layers

transformer.py:
  ModelOutput(logits [B,S,V], loss_inputs: tuple[RouterOutput,...]); .total_load() -> int64 [E]
  LMHead(cfg)  # fp32 master weight, bf16 matmul cast
  MoKBlock(cfg, layer_idx, rope_theta=None, mok_runtime=None, dtype=bf16)
  MoKTransformer(cfg, backend='reference'|'mok', mok_runtime=None, rope_theta=None, dtype=bf16)
    .forward(tokens int64 [B,S]) -> ModelOutput
    .iter_master_params() -> Iterator[(name, Tensor)]   # state_root domain: all params + balance_bias buffers
    .is_expert_local(name) -> bool; .param_shapes() -> dict[str, tuple]; .moe_layers() -> list[MoKMoELayer]
  COMPILE_ENV = 'MOK_COMPILE'

init.py:
  init_model(cfg, seed: int, device='cpu', backend='reference', mok_runtime=None) -> MoKTransformer  # seed_everything + fixed-order init; meta skips values

reference.py:
  reference_config(cfg) -> ModelConfig            # ep_size forced 1 via model_copy (bypasses EP validator by design)
  build_reference_model(cfg, seed, device='cpu') -> MoKTransformer
  evaluate_sequences(model, token_batches: Iterable[Tensor [B,S+1]|[S+1]], device='cpu') -> float  # token-weighted mean CE, no grad
```


# Round B additions

## subnet.core.engine.inner

```
subnet/core/zero1.py:
  Comm (runtime-checkable Protocol): broadcast(tensor, src_rank) -> None (overwrite in place from src); all_reduce(tensor) -> None (SUM in place)
  SingleProcessComm  # world_size==1 identity comm; broadcast validates src_rank==0
  TorchDistComm(group: Any | None = None)  # torch.distributed sum/broadcast; src_rank is group-relative (get_global_rank translation); dist imported lazily inside methods
  flat_grad_all_reduce(named_params: Mapping[str, Tensor], comm: Comm, world_size: int) -> None
    # THE fixed-order gradient reduction: ONE fp32 flat buffer in sorted-name order (missing grads contribute zeros and get a grad created), comm.all_reduce, div_(world_size), write back into p.grad in the grad dtype. world_size==1 round trip is bitwise lossless (tested incl. bf16 extremes/inf/nan).
  class Zero1Adam(named_params: Mapping[str, Parameter], *, rank: int, world_size: int, is_expert_local: Callable[[str], bool], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1, comm: Comm | None = None)
    # expert-local params (.routed_ marker) stepped fully locally on every rank; replicated params ZeRO-1 partitioned by deterministic name-sorted round-robin: owner(name_i) = i % world_size over sorted replicated names. comm defaults to SingleProcessComm (only legal when world_size==1; world_size>1 without comm raises).
    @classmethod fresh(named_params, inner: InnerOptConfig, *, rank, world_size, is_expert_local, comm=None) -> Zero1Adam   # the per-window Adam reset (protocol decision #1); state starts zero
    .step(lr: float) -> None  # AdamW math bitwise identical to torch.optim.AdamW(foreach=False) single-tensor non-capturable path (param.mul_(1-lr*wd); exp_avg.lerp_; exp_avg_sq.mul_().addcmul_(); python-float bias corrections; addcdiv_) — pinned by test vs torch 2.13. Params with grad None are skipped (step count does not advance, like torch lazy init). Then broadcasts EVERY replicated param.data from its owner in sorted-name order.
    .owned_names / .expert_names / .replicated_names -> tuple[str, ...]; .owner_of(name) -> int (replicated only, KeyError for expert names); .rank/.world_size/.betas/.eps/.weight_decay/.comm

subnet/core/inner_loop.py:
  IGNORE_INDEX = -100  # F.cross_entropy default ignore_index, honored by losses.loss_head
  @dataclass(frozen=True) WindowResult(entry_loss, mean_loss, final_loss: float; tokens: int (all ranks); grad_norm_mean: float (PRE-clip); capacity_util_max: float; router_entropy_mean: float; expert_load: int64 [E] CPU (summed over layers+steps); global_inner_steps_done: int (= global_inner_step0 + plan.inner_steps))
  class InnerLoop(model: MoKTransformer, cfg: RunConfig, phase: PhaseConfig, *, rank: int, world_size: int, comm: Comm, device: str | torch.device)
    .run_window(plan: WindowBatchPlan, shard_lookup: Callable[[int], ShardReader], window: int, global_inner_step0: int, tokens_consumed0: int, null_round: bool = False) -> WindowResult
    # Per inner step s: accum = clamp(accum_at(cfg.window, tokens_consumed), 1, plan.grad_accum); grads zeroed (set_to_none); per microbatch: plan.microbatch_tokens -> view [seqs_per_microbatch, seq_len] -> model forward -> loss_head(logits, shifted targets, loss_inputs, cfg.model) -> (total/accum).backward(). Targets = within-sequence shift with last column IGNORE_INDEX (keeps T=B*S >=512 and %256 for the MoK kernel; S-1 supervised positions/seq). After accum: flat_grad_all_reduce over NON-expert grads; fp32 global grad-norm clip at cfg.inner.grad_clip (coef = clip/(norm+1e-6) clamped at 1, torch convention; expert-local squared sums included via a 1-element all_reduce — the only other collective); lr = lr_at(phase.lr, global_inner_step0+s, cfg.tokens_per_inner_step); Zero1Adam.fresh(...).step(lr) unless null_round; MoeHealth.post_step(per-layer step loads, capacity_multiplier=phase.capacity_multiplier). Fresh Zero1Adam + fresh MoeHealth per window. bf16 autocast on CUDA, plain math on CPU (nullcontext — same code path). tokens counted as accum * tokens_per_rank_microbatch * world_size per step. Plan validated against (window, rank, world_size, phase.{inner_steps,grad_accum,seq_len,tokens_per_rank_microbatch}) — ValueError on mismatch.

subnet/core/pseudo_grad.py (the upstream implementation distributed.py:364-545 adapted, non-DTensor, credited in NOTICE):
  CpuSnapshot (frozen dataclass): tensors: dict[str, Tensor] (CPU clones in MASTER dtype), pinned: bool; .names (sorted), __len__, __contains__
    @classmethod take(named_params: Mapping | Iterable[tuple[str, Tensor]], *, pin: bool | None = None) -> CpuSnapshot  # pin=None resolves to torch.cuda.is_available(); pinned buffers + async D2H fenced with one synchronize; duplicate names raise
  restore_and_extract_delta(named_params, snapshot) -> dict[str, Tensor]  # Δ = θ_start − θ_end as contiguous fp32 CPU tensors AND params restored to θ_start in place (bitwise — snapshots keep master dtype). Names must match snapshot exactly; shape/dtype checked. Works over iter_master_params() incl. balance_bias buffers (state_root round-trip pinned by test).

subnet/core/moe_health.py:
  MoeHealth(model: MoKTransformer, cfg: RunConfig, *, capacity_multiplier: float | None = None)  # default cfg.mok.schedule_capacity_multiplier; requires model.cfg.num_experts == cfg.model.num_experts; cm must be > 0
    .post_step(router_loads: Sequence[int64 [E]]) -> float  # fixed order: (1) mok backend only: MXFP8WeightManager.requantize_ per layer (reference backend: no-op, mok never imported, quant_cache stays None — tested); (2) per layer in layer order: router.update_balance_bias_(load, cfg.model.bias_update_rate); (3) capacity util. Returns this step's util.
    .capacity_alert(threshold: float) -> bool  # running max util >= threshold
    .max_util / .capacity_multiplier
  # Util formula (documented contract): experts EP-blocked contiguously by PROTOCOL geometry from cfg.model (rank r hosts [r*E_local, (r+1)*E_local)); util_layer = max_r sum(load[block_r]) / (sum(load)/ep_size * capacity_multiplier); util = max over layers; 1/cm at perfect balance, ep_size/cm at total skew; 0.0 on zero load.
```

## subnet.core.engine.io

```
subnet/core/exchange.py (all async unless noted):
  ExchangeError(RuntimeError)
  UploadReceipt (frozen dataclass): payload_hash: hex64, key: str, committed: bool
  put_window_payload(storage: StorageClient, chain, payload: WindowPayload, *, version: int) -> UploadReceipt  # TWO-PHASE: chain.commit_window(WindowCommit(window, H(payload), meta.state_root, meta.theta_end_hash)) via asyncio.to_thread FIRST, then storage.put_bytes(keys.payload_key(window, uid, str(version)), serialize(payload)); chain failure propagates and NOTHING is uploaded
  AggregatorObject (dataclass): window: int, payloads: dict[int, bytes]; .serialize() -> bytes; @classmethod .deserialize(data, *, max_bytes, max_decompressed_bytes=None)  # wire v1: b'MOKA'|u8 ver|zstd(level3, u32le header_len|canonical JSON {window, entries:[{uid,nbytes,payload_hash}] uid-strictly-increasing}|blobs); bounds+canonical-header+exact-blob-accounting+per-entry blake2b all checked pre-copy
  put_aggregator_object(storage, window, payloads: Mapping[int, bytes]) -> str; get_aggregator_object(storage, bucket, window, *, max_bytes) -> AggregatorObject
  CertifiedGather (frozen dataclass): payloads: OrderedDict[int, WindowPayload] (uid-ascending), missing: dict[int, str]; .uids
  gather_certified(storage, cert: WindowCertificate, peer_buckets: Mapping[int, BucketCreds], *, expected_param_shapes, expected_dense, topk, version: int, deadline_s, max_bytes, target_chunk=64, leader_bucket: BucketCreds|None=None) -> CertifiedGather  # fetches exactly cert.included_uids with expected_hash=cert.payload_hashes[uid]; per-peer failures (+ uids with no bucket) retried once from leader aggregator mirror; every frame deserialized + validate_structure'd; failures land in missing with prefixed reasons ('no_bucket', 'missing:', 'integrity:', 'timeout', 'invalid:', '...; mirror: ...')
  gather_from_aggregator(storage, cert, leader_bucket, *, expected_param_shapes, expected_dense, topk, max_bytes, target_chunk=64) -> CertifiedGather  # mirror-only gather (catch-up path)
  put_certificate(storage, cert) -> str; get_certificate(storage, bucket, window, *, max_bytes=1MiB) -> WindowCertificate  # canonical JSON; sig/consensus checks stay the caller's job
  debug_key(window, uid) -> 'debug/w{:08d}/uid{:05d}.json' (key namespace owned here, validated like keys.py)
  put_debug_slices(storage, window, uid, named_params: Mapping|Iterable[pairs], elems=2) -> str; get_debug_slices(storage, bucket, window, uid, *, max_bytes=1MiB) -> dict[str, list[float]]  # the upstream implementation debug-dict: first-`elems` fp32 values per name
  put_telemetry(storage, window, uid, payload: Mapping, *, max_bytes=1MiB) -> str; get_telemetry(...) -> dict  # bounded canonical JSON
  put_audit_report(storage, report: Mapping, *, max_bytes=1MiB) -> str  # needs int window/auditor_uid/miner_uid; hotkey/sig fields pass through untouched
  list_audit_reports(storage, bucket, window, *, max_bytes=1MiB) -> list[dict]  # key-sorted; malformed or id-contradicting objects skipped, logged
  gate_check(storage, bucket, key, boundary_ts: float, grace_s: float) -> bool  # object_timestamp + windows.is_in_gate; missing object -> False
  Constants: AGGREGATOR_MAGIC=b'MOKA', AGGREGATOR_WIRE_VERSION=1, DEFAULT_JSON_MAX_BYTES=1<<20

subnet/core/checkpoint.py:
  CheckpointError(RuntimeError)
  CheckpointMeta (frozen dataclass): window, global_step, tokens_consumed, state_root, manifest_hash, spec_version; .to_dict(), .canonical() -> bytes, .from_dict(strict)
  Checkpointer(storage: StorageClient|None, local_dir, keep_local: int = 2, *, no_dist: bool = True)
    .window_dir(w) -> Path (local_dir/w{:08d})
    .save_local(window, named_params: Mapping|Iterable[pairs], outer_state: Mapping[str, Tensor], meta) -> Path  # sync; atomic tmp+rename; layout: model/ (DCP), outer_state.pt = torch.save({'outer': ...}), meta.json canonical; prunes after
    async .save(...) -> Path  # save_local + upload when storage wired
    async .upload(window)  # deterministic tar of model/ -> checkpoint_key(w,'model.tar') (multipart-capable upload_file) + outer_state.pt + meta.json objects
    .prune_local() -> list[int]; .local_windows() -> list[int]
    .load_local(window) -> (state_dict, outer_state, CheckpointMeta)  # DCP template rebuilt from .metadata (no model needed), dtypes/bytes exact
    async .load_latest(*, bucket: BucketCreds|None=None) -> tuple|None  # newest local, else newest complete remote in bucket (downloaded+extracted into local_dir)
  sparse_pairs_from_compressed(name, ct: CompressedTensor, compressor: TopKCompressor) -> (flat_idx int64, vals fp32)  # chunk->orig mapping, pad positions dropped; scatter == compressor.decompress bitwise (pinned)
  build_outer_inputs(payloads: Mapping[int, WindowPayload] uid-ascending, compressor, param_names) -> (peer_sparse, dense_contribs, peer_norms)  # EXACT inputs for ReplicatedOuterStep.apply; norms = l2 of what enters the merge (== norm of decompressed); window_runner MUST import this
  consensus_state_root(commits: Mapping[int, WindowCommit]) -> str|None  # majority, tie -> lexicographically smallest
  CatchUpDivergence (frozen): window, expected_root, actual_root, detail; CatchUpError(RuntimeError) with .divergence; CatchUpReport (frozen): applied_windows, skipped_void, unverified_windows, final_root
  async catch_up(model_params, outer_step, exchange, storage, chain, manifest, cfg: RunConfig, from_window, to_window, apply_fn: Callable[[CertifiedGather], None]|None = None, *, leader_bucket: BucketCreds, dense_names: Iterable[str]|None = None, max_bytes: int|None = None) -> CatchUpReport
```

## subnet.core.engine.runner+replay

```
subnet.core.window_runner: DENSE_SUFFIX='balance_bias'; RunState(global_step,global_inner_step,tokens_consumed) frozen; run_state_at(cfg,manifest,window,*,world_size)->RunState (consensus counters at window start; mirrors inner-loop accum-ramp token accounting; void windows skipped); WindowClock protocol {boundary_ts(window)->float, now()->float}; RunnerComm protocol {broadcast,all_reduce,gather_object(obj)->list|None, broadcast_object(obj,src)->Any, barrier()}; SingleNodeComm (ws==1, extends zero1.SingleProcessComm); TorchDistRunnerComm (extends TorchDistComm; dist.gather_object/broadcast_object_list/barrier); build_window_plan(manifest,phase,*,run_seed,uid,window,rank,world_size)->WindowBatchPlan; shared_master_root(model,*,rank,world_size,comm)->str|None (assign_owned_params ownership, rank-parallel, ==state_root for ws==1); TrainingArtifacts(uid,window,state_root_start,theta_end_root,theta_end_digests:dict[str,bytes],deltas,result:WindowResult,sample_digest,payload,payload_bytes,payload_hash); run_training_phase(model,cfg,manifest,phase,*,uid,window,rank,world_size,comm,shard_lookup,global_state,compressor=None,error_feedback=None,device='cpu',plan=None,run_seed=None)->TrainingArtifacts — phases 1-5, NO storage/chain/clock deps, restores theta_start bitwise (asserted on per-tensor digests), payload bytes deterministic on rank 0 when compressor+EF given (both-or-neither enforced); await_certificate(storage,leader_bucket,window,*,timeout_s,poll_s)->WindowCertificate|None; WindowOutcome(window,state_after,restart_required,desync,late_upload,reason,state_root_start,theta_end_root,state_root_after,payload_hash,upload_key,gather_uids,outer_report,train_result,checkpoint_saved,sync_divergences); WindowRunner(model,cfg,manifest,*,uid,rank,world_size,comm,storage,chain,shard_cache,fetch_fn,compressor,error_feedback,outer_step,checkpointer=None,metrics=None,clock,peer_buckets:Callable[[int],Mapping[int,BucketCreds]],leader_bucket:Callable[[int],BucketCreds],payload_version=1,device='cpu',self_leader=False,sign_fn=None,cert_poll_s=2.0,cert_timeout_s=180.0) with async run_window(window,global_state:RunState)->WindowOutcome; gate close = clock.boundary_ts(window+1)+cfg.window.upload_grace_s; checkpoint when window % cfg.window.checkpoint_every_windows == 0 with meta.state_root = post-outer-step root (theta_start(w+1) per layout contract). subnet.core.replay: PreconditionError; ReplayTask(miner_uid,window,commit:WindowCommit); AuditReport(miner_uid,window,theta_start_root,committed_theta_end,replayed_theta_end,match,divergences:list[dict],wall_time_s,auditor_uid,signature='') with .to_json(); report_message(report|dict)->bytes (raw 32B blake2b canonical hash of unsigned fields, certificate_message convention); sign_report(report,sign)->AuditReport; verify_report(report|dict,verify)->bool (never raises); WindowReplayer(model,cfg,manifest,*,comm,shard_lookup_factory:Callable[[WindowBatchPlan],ContextManager[Callable[[int],ShardReader]]],auditor_uid=0,rank=0,world_size=1,device='cpu').replay(task,*,global_state=None (defaults run_state_at), expected_digests=None)->AuditReport — PreconditionError unless replica root == commit.state_root; runs run_training_phase with the miner's uid; replica unchanged afterwards; divergences via window_state.divergence_report when miner per-tensor digests supplied, else single '<state_root>' record; audit_sampler(run_seed,block_hash,window,active_uids,rho,auditor_uids)->dict[int,list[tuple[int,int]]] — philox(audit_seed) Bernoulli-rho in sorted-uid order, round-robin over sorted auditors, golden-pinned.
```

## sft

```
sft.hf_model.configuration_mok_moe: MokMoeConfig(PretrainedConfig, model_type='mok_moe'; HF names: num_hidden_layers/num_attention_heads/num_key_value_heads/head_dim/num_experts/num_experts_per_tok/intermediate_size/max_position_embeddings/rope_theta/rms_norm_eps; pad/bos/eos=0/1/2, tie_word_embeddings=False, mok_provenance dict); .from_model_config(ModelConfig, **kw) / .to_model_config() (lazy mok_core bridges, repo-side only). sft.hf_model.modeling_mok_moe: MokMoeForCausalLM(MokMoePreTrainedModel, GenerationMixin) -> CausalLMOutputWithPast (labels -100-masked fp32 CE); MokMoeModel; MokMoeDecoderLayer/MokMoeAttention (GQA, torch SDPA only, RoPE fp32-built)/MokMoeSparseMoeBlock (op-for-op replica of MoKMoELayer._reference_forward)/MokMoeGate (fp32 router + persistent e_score_correction_bias buffer, selection-only, softmax over selected UNBIASED logits)/MokMoeMLP (SwiGLU); supports gradient checkpointing + DynamicCache; _keep_in_fp32_modules_strict=['mlp.gate.weight','e_score_correction_bias']. sft.convert_dcp_to_hf: convert(checkpoint_dir, out_dir, tokenizer_path=None, *, dtype='bfloat16', model_config: ModelConfig|None=None, check_state_root=True, max_shard_bytes=4GiB) -> ConversionReport(out_dir, weight_files, num_tensors, total_bytes, dtype, meta); pure helpers remap_state_dict(sd, cfg) (explicit mapping table in module docstring; qkv row-split, [E,I,H] expert unstack, balance_bias->e_score_correction_bias; strict both ways, raises ConversionError), apply_dtype_policy(hf_sd, dtype) (router+bias fp32, rest incl. lm_head cast), infer_hf_config(sd, *, head_dim=None), load_dcp_state_dict(model_dir), read_checkpoint_meta(dir), write_sharded_safetensors(...), write_tokenizer_files(...); constants META_KEYS, FP32_KEEP_SUFFIXES, PAD/BOS/EOS_TOKEN/_ID; main() CLI (--tokenizer --dtype --model-config --no-state-root-check). sft.verify_conversion: verify(checkpoint_dir, hf_dir, *, n_tokens=64, seq=32, seed=0) -> VerificationReport(max_abs_diff, argmax_agreement, n_positions, seq, dtype; .ok) — THE PARITY GATE, raises VerificationError unless max|logit diff|<2e-2 and argmax agreement>0.99 (both forwards under mok_core sdpa_backend() pin); main() CLI. sft.data_prep: CHAT_TEMPLATE/IM_START/IM_END/LABEL_IGNORE; render_chat(messages, tokenizer) -> (input_ids, labels) lists, loss on assistant body+im_end only; tulu3/openhermes(*, rows=None)/reasoning_traces(path, *, rows=None) -> iter of {'messages': [...]} (datasets lazy, rows injectable); text_ngrams/build_eval_ngrams/decontaminate(examples, eval_ngrams, n=13); pack_examples/pack_for_sft(pairs, seq_len=16384, pad_id=0) + SFTPackCollator(seq_len, pad_id).__call__ -> {input_ids, labels, attention_mask}. sft.sft_train: load_settings(yaml) -> SFTSettings; cosine_warmup_lambda(*, start_lr, peak_lr, warmup_steps, total_steps); build_mixture(settings); load_eval_ngrams(path, n); run(settings) (trl/transformers/datasets lazy); main() == mok-sft console script. sft.eval_select: PROBES (16 regex-judged instruction probes), judge_response, score_outputs(dict)->float, score_checkpoint(path, generate_fn), list_checkpoints(dir), pick_checkpoint(run_dir, *, generate_fn=None) -> (best_path, scores) (ties -> later), default_generate_fn, main() CLI. CHECKPOINT LAYOUT CONTRACT consumed exactly as specified; test_sft_convert.make_checkpoint_dir doubles as a layout pin.
```

## rl

```
rl.dpo_train: DPOSettings (frozen dataclass: model_dir, output_dir, ref_model_dir=None->model_dir, seed=42, epochs=1.0, lr=5e-7, beta=0.1, bf16, micro_batch_size=1, grad_accum=16, max_prompt_length=2048, max_length=4096, logging_steps, save_steps, eval_holdout_examples, max_examples, datasets); load_settings(path)->DPOSettings; render_messages(msgs)->str / render_prompt(msgs)->str (adds '<|im_start|>assistant\n') / render_completion(content)->str ('content<|im_end|>\n') — byte-identical to sft.data_prep.CHAT_TEMPLATE (jinja-verified); normalize_preference_row(row)->{prompt,chosen,rejected}|None (accepts message-list chosen/rejected [Tulu-3-pref, UltraFeedback-binarized] or bare prompt/chosen/rejected strings); normalize_preferences(ds)->Iterator (skips bad rows); build_preference_mixture(settings, *, tulu_rows=None, ultrafeedback_rows=None)->Iterator (lazy datasets); run(settings) (lazy trl DPOTrainer, ref_model=SFT ckpt); main(argv)->int. Constants: TULU3_PREF_DATASET='allenai/llama-3.1-tulu-3-8b-preference-mixture', ULTRAFEEDBACK_DATASET='HuggingFaceH4/ultrafeedback_binarized' split train_prefs. || rl.grpo_train (console script mok-rl): VllmSettings(server_mode=True, endpoints:tuple[str,...]; validates http(s) URLs); GRPOSettings(model_dir, output_dir, seed=42, group_size=8, lr=1e-6, kl_coef=0.04, bf16, micro_batch_size=8, grad_accum=8, max_prompt_length=1024, max_completion_length=2048, temperature=1.0, epochs, logging_steps, save_steps, max_prompts, code_timeout_s=6.0, code_sandbox='auto', vllm, datasets); load_settings(path); endpoint_for_rank(settings, rank=None)->str (RANK round-robin); reward_router(sample, completion, *, code_timeout_s=6.0, code_sandbox='auto')->float (tag 'math'->verify_math vs sample['reference_answer'], 'code'->verify_code vs sample['tests'], else ValueError); make_trl_reward_fn(*, code_timeout_s, code_sandbox)->fn(prompts, completions, **cols)->list[float] (TRL column-batched kwargs; str or conversational completions); build_rlvr_dataset(settings, *, gsm8k_rows/math_rows/mbpp_rows=None)->list[{prompt,tag,reference_answer,tests}] (uniform schema, seed-deterministic shuffle, max_prompts cap); run(settings) (lazy trl GRPOTrainer; GRPOConfig kwargs filtered via dataclasses.fields for cross-version vllm server-mode fields); main(argv)->int. || rl.rewards.math_reward: verify_math(completion, reference_answer)->float{0,1}; extract_final_answer(completion)->str|None (last \boxed{...} balanced > last '#### x' > last number line incl fractions/percent/$/commas); extract_boxed(text)->str|None; normalize_answer(s)->str (LaTeX \frac (nested)/\sqrt[n]/\pi/^/%/units/=RHS); sympy lazy: Expr.equals then simplify(a-b)==0, guarded vs huge exponents; float fallback rel 1e-6; MAX_ANSWER_CHARS=300. || rl.rewards.code_reward: verify_code(completion, tests, *, timeout_s=6.0, sandbox='auto'|'bwrap'|'rlimit', mem_bytes=1GiB, nproc=128)->float (fraction passed; one subprocess per test); extract_code(completion)->str|None (last ```python fence > last generic fence > raw); run_snippet(program, ...)->ExecResult(passed, returncode, timed_out, stdout, stderr); async run_code_batch(items[{completion,tests,timeout_s?}], *, concurrency=8, timeout_s, sandbox)->list[float]; resolve_sandbox/bwrap_available; rlimit fallback = python -I + RLIMIT_{CPU,AS,NPROC,FSIZE,NOFILE} + own session (SIGKILL group on timeout) + minimal env + tmpdir cwd + unshare -n when usable; bwrap path = --unshare-all, ro-binds, tmpfs /tmp. || rl.data.rlvr_math: build_math_prompts(*, n=None, seed=0, gsm8k_rows=None, math_rows=None)->list[{prompt(+MATH_PROMPT_SUFFIX boxed instruction), reference_answer, tag:'math'}]; gsm8k_reference/math_reference; deterministic_subsample[T](items, n, seed) (order-preserving, seed-pinned); GSM8K openai/gsm8k:main, MATH EleutherAI/hendrycks_math x 7 configs. || rl.data.rlvr_code: build_code_prompts(*, n=None, seed=0, mbpp_rows=None, selfgen_path=None)->list[{prompt, tests, tag:'code'}]; mbpp_item(row) (text+test_list+test_setup_code prefix); load_selfgen_jsonl(path) ({'prompt','tests'|'test'} lines, strict); format_code_prompt/CODE_PROMPT_TEMPLATE; MBPP google-research-datasets/mbpp. || rl.vllm_plugin.mok_moe_vllm: register_mok_moe() (idempotent; ModelRegistry.register_model('MokMoeForCausalLM', 'rl.vllm_plugin.mok_moe_vllm:MokMoeForCausalLM_vLLM'); raises ImportError 'vllm>=0.8 required' without vllm); MokMoeForCausalLM_vLLM via module __getattr__ (PEP 562, built on first access; QKVParallelLinear+get_rope(neox)+Attention, FusedMoE with custom_routing_function replicating selection-only e_score_correction_bias (softmax over UNBIASED selected fp32 logits), fp32 ReplicatedLinear gate carrying the bias param, shared expert added pre-all-reduce reduce_results=False DeepSeek-V3 style, ParallelLMHead+LogitsProcessor, load_weights->set[str]); pure tables (no vllm needed): STACKED_PARAMS_MAPPING (q/k/v->qkv_proj, gate/up->gate_up_proj), map_dense_name(name)->(vllm_name, shard|None), expert_params_mapping(num_experts)->[(target, 'experts.{e}.{proj}.', e, w1|w2|w3)], is_routed_expert_weight(name), vllm_param_of(name, num_experts); HF_ARCHITECTURE, VLLM_CLASS_PATH.
```

## release

```
## release

release/provenance.py:
  BUNDLE_SPEC_VERSION = 1; INDEX_FILENAME='index.json', MANIFEST_FILENAME='manifest.json', WINDOWS_FILENAME='windows.jsonl', AUDITS_FILENAME='audits.jsonl', EVALS_FILENAME='evals.json', WEIGHTS_DIRNAME='weights', REPLAY_DIRNAME='replay', REPLAY_SCRIPT_NAME='replay_window.py', WEIGHTS_REF_SUFFIX='.ref.json'; REQUIRED_FILES; AUDIT_REPORT_FIELDS (10-field plan AuditReport shape)
  BundleError(ValueError)
  is_hex64(v) -> bool; blake2b_hex(bytes) -> str; blake2b_file(path) -> str  # streaming, hashlib-only (no torch)
  bundle_root_hash(files: Mapping[str,str]) -> str  # blake2b-256 over sorted (relpath, digest): len(rel) le32 || rel utf8 || raw 32 digest bytes
  class WindowRecord(FrozenModel): window>=0, state_root hex64, certificate: dict|None, payload_hashes: dict[int,str] (hex64 values), telemetry_hash: hex64|None — bad hex/negative rejected at construction
  class BundleManifest(FrozenModel): spec_version, manifest_hash, files: dict[str,str], root_hash, built_at_block — model_validator recomputes root_hash from files (inconsistent index unconstructible); index.json == canonical_bytes(BundleManifest)
  audit_report_problems(report, *, where='audit report') -> list[str]  # field/type/hex checks + consistency: match == (committed_theta_end==replayed_theta_end), match => divergences empty
  audit_report_message(report) -> bytes  # 32 bytes: blake2b-256 of canonical_bytes(report minus 'signature') — what auditors sign
  audit_sort_key(report) -> (window, miner_uid, auditor_uid)
  build_bundle(out_dir, *, manifest: RunManifest, window_records, audit_reports, weights_files, eval_results, extra, built_at_block=0, copy_weights=True, include_replay_script=True) -> BundleManifest
    # deterministic (no timestamps); refuses non-empty out_dir; sorts records, rejects duplicate windows; validates every audit report + its window has a WindowRecord; manifest.json bytes hash == canonical manifest_hash; evals.json = canonical {'extra':..., 'results':...}; copy_weights=False writes weights/<name>.ref.json {blake2b, bytes, filename}

release/verify_bundle.py:
  SignatureVerifier = Callable[[bytes, bytes, int], bool]  # (message32, signature, auditor_uid)
  @dataclass(frozen=True) VerifyReport(ok, problems: list[str], files_checked, windows, audits); .to_json()
  verify(bundle_dir) -> VerifyReport  # never raises; CPU-only, hashlib-only. Checks: index shape (exact 5 keys, spec_version pin, hex64, path traversal), every listed file exists+hash matches, unlisted files flagged, root_hash recomputes, RunManifest parses + manifest_hash matches, windows.jsonl strictly monotonic WindowRecords, audits.jsonl well-formed + windows covered + (signature != '' AND mok_core.chain.verify_audit_signature importable -> checked; missing hook = optional-pass, hook exception/False = problem), evals shape, .ref.json shape
  build_parser(); main(argv=None) -> int  # 0 iff ok; --json full report; python -m release.verify_bundle <dir>

release/replay_window.py:
  DEFAULT_INIT_SEED = 42; ReplayCLIError(RuntimeError)
  build_parser()  # (--bundle DIR | --manifest PATH) required-xor, --window N, --miner-uid U, --theta-start DIR (all required), [--config RunConfig.yaml] [--backend reference|mok=mok] [--device=cuda] [--out report.json]
  load_manifest_arg(bundle, manifest_path) -> RunManifest
  report_to_dict(report) -> dict  # dict/dataclass/pydantic -> validated AuditReport wire dict
  main(argv=None) -> int  # 0 iff match, 1 mismatch, 2 error; report JSON to --out or stdout
  _replay(...)  # heavy path behind main(): load_run_config + config_hash must equal manifest.config_hash; init_model(cfg.model, seed=42, device, backend); theta_start via subnet.core.checkpoint.load_master_state(model, ckpt_dir) if present else direct DCP FileSystemReader on <dir>/model per checkpoint layout contract; subnet.core.replay.WindowReplayer(manifest=, cfg=, model=).replay(uid=, window=)

release/run_evals.py (console script mok-eval):
  DEFAULT_TASKS = (mmlu, gsm8k_cot, ifeval, arc_challenge, hellaswag, winogrande)
  extract_results(raw) -> dict[str, dict[str, float]]  # pure; accepts simple_evaluate() output or bare results; strips ',none', non-default filter -> 'metric/filter', drops alias/bools/strings
  results_to_markdown(results) -> str  # pure; sorted | Task | Metric | Value | table, 4-decimal floats
  humaneval_cmd(model_dir, *, metrics_out, n_samples=20, temperature=0.2, batch_size=10) -> list[str]  # pure; accelerate launch main.py --tasks humaneval ... for bigcode-evaluation-harness
  run_lm_eval(model_path, *, tasks, backend='vllm'|'hf', model_args, batch_size='auto', limit) -> results  # lazy lm_eval
  run_humaneval(model_dir, harness_dir, ...) -> {'humaneval': {...}}  # guarded: FileNotFoundError without harness checkout, CalledProcessError propagates
  build_parser(); main(argv=None) -> int  # writes evals.json (the build_bundle eval_results input) + optional --markdown-out

release/hf_upload.py:
  MODEL_CARD_TEMPLATE / DEFAULT_REPLAY_INSTRUCTIONS  # provenance-forward card; placeholders: model_name, benchmarks_table, provenance_root_hash, manifest_hash, replay_instructions
  UploadError(ValueError); @dataclass(frozen=True) PlannedOp(op, target, source)
  card_placeholders(template) -> set[str]; render_model_card(*, model_name, benchmarks_table, provenance_root_hash, manifest_hash, replay_instructions=..., card_template=...) -> str
  plan_release(hf_repo, dirs, *, private=True) -> list[PlannedOp]  # pure; validates 'org/name', dirs exist, path_in_repo legal; order: create_repo, upload_folder per sorted key ('' -> repo root '.'), upload_file README.md
  upload_release(hf_repo, dirs, *, card_template=..., model_name=None, benchmarks_table/provenance_root_hash/manifest_hash, replay_instructions, private=True, token=None, commit_message, dry_run=False) -> list[PlannedOp]  # dry_run never imports huggingface_hub; real path: HfApi repo_exists->create_repo, upload_folder per dir, README from rendered card bytes

All key names re-exported from release/__init__.py; `import release` loads no heavy deps (asserted in a smoke run).
```


# Round C additions

## fleet

```
## fleet (drop-in section for docs/INTERNAL_API.md)

fleet/attestation/challenge.py:
  ATTEST_DOMAIN=b'attest.v1'; DEFAULT_DEADLINE_S=420.0; DEFAULT_INNER_STEPS=20; BASE_CONFIG_PATH/TOY4L_CONFIG_PATH (resolved via the subnet package)
  AttestationChallenge(FrozenModel): challenge_id (16 lower hex), seed (63-bit), model_overlay: dict (toy4L.yaml verbatim), inner_steps=20, deadline_s, issued_block — validators on all fields
  make_challenge(block_hash: bytes32, issued_block, *, deadline_s=420, inner_steps=20) -> AttestationChallenge  # CONSENSUS: digest=blake2b256(block_hash||b'attest.v1'); challenge_id=digest[:8].hex(); seed=le64(digest[8:16])&(2^63-1); golden-pinned (block_hash=0x01*32 -> id 227a506a6bf08680, seed 4264113716381589857)
  toy4l_overlay() -> dict; challenge_run_config(challenge) -> RunConfig  # base.yaml deep_merge overlay deep_merge {'window':{'inner_steps':challenge.inner_steps}}, env-interpolated
  derive_expected(challenge, *, device, backend='reference', comm=None) -> str  # verifier precompute == run_reference().state_root (same code path); root broadcast to all ranks

fleet/attestation/reference_step.py:
  ATTEST_DATA_DOMAIN=b'attest.data.v1'; ATTEST_UID=0; ATTEST_WINDOW=0; ATTEST_NUM_SHARDS=4; ATTEST_DATA_WORLD_SIZE=8; ATTEST_TOKENIZER_HASH
  AttestationResponse(FrozenModel): challenge_id, state_root, wall_time_s, fingerprint: dict (environment_fingerprint().to_json())
  attestation_run_seed(challenge) -> bytes32  # CONSENSUS: blake2b256(b'attest.data.v1'||raw8(challenge_id)||le64(seed)); golden c780e611...21a2
  write_attestation_shards(challenge, cfg, out_dir) -> (DatasetShardIndex, DatasetManifestRef)  # philox(seed).integers(0,vocab,(rows,seq_len),uint16) row-major, 4 shards sized for ranks=8 (CONSENSUS wire); rank/platform-invariant bytes
  attestation_manifest(challenge, cfg, ref) -> RunManifest  # minimal, pure; prf=attestation_run_seed
  attestation_state_root(model, *, comm: RunnerComm, rank, world_size) -> str|None  # ws==1: hash_named_tensors(iter_master_params); ws>1: expert-local tensors gathered + cat(dim=0) in rank order (protocol EP blocking) == ep_size=1 layout; replicated from rank 0
  run_reference(challenge, *, backend='mok', device, comm=None) -> AttestationResponse  # init_model(seed) -> temp philox shards -> UNMODIFIED WindowBatchPlan+InnerLoop (20 steps) -> root; rank/ws from torch.distributed when initialized; response identical on every rank
  build_parser(); main(argv) -> int  # torchrun entry (entrypoint.sh attest): --challenge FILE|- --backend --device --out; enforce_determinism(); rank0 emits JSON

fleet/attestation/verify.py:
  REATTEST_DOMAIN=b'reattest.v1'; AttestationVerdict(FrozenModel): ok, reason
  judge(challenge, response, expected_root, *, received_ts, issued_ts) -> AttestationVerdict  # id match + 0<=elapsed<=deadline_s + case-insensitive root equality; collects all problems; never raises. Deadline doc: 420s only achievable on real 8xSM103 NVLink
  schedule_reattestation(run_seed, block_hash, active_uids, rate) -> list[int]  # CONSENSUS: philox(le64(blake2b256(run_seed||b'reattest.v1'||block_hash)[:8])&mask63).random(n) < rate over sorted uids; golden (run_seed=bytes(range(32)), bh=0x01*32, uids 0..19, rate .25) -> [8,11,16,17,18]
  Verifier(*, deadline_s=420, inner_steps=20): .issue(chain) -> challenge from chain.current_block()/block_hash(); .judge(...); .schedule_reattestation(...)

fleet/onboarding/preflight.py:
  REQUIRED_GPUS=8; GPU_NAME_TOKEN='B300'; REQUIRED_COMPUTE_CAP='10.3'; MIN_VRAM_BYTES=280e9; MIN_RAM_BYTES=1.2e12; MIN_NVME_FREE_BYTES=3e12
  GpuInfo(name, vram_bytes, compute_cap|None); PreflightCheck(FrozenModel: name, ok, detail); PreflightError
  PreflightReport(FrozenModel): checks; .ok; .failures(); .strict() raises PreflightError listing all failures
  parse_smi_xml(xml) -> list[GpuInfo]; parse_compute_caps_csv(csv) -> list[str]; parse_topo_matrix(text) -> {(i,j): link}; parse_meminfo_bytes(text) -> int
  run_preflight(*, smi_xml=None, compute_caps_csv=None, topo_text=None, meminfo_text=None, cache_dir=None, env=None, runner=None, disk_usage=None) -> PreflightReport  # None inputs probed live; probe failure -> failed check; compute cap: --query-gpu preferred, XML <compute_cap> fallback, neither -> fail

fleet/onboarding/wallet_setup.py:
  R2_ENV_VARS (read pair: R2_ACCOUNT_ID/R2_BUCKET_NAME/R2_READ_*); R2_WRITE_ENV_VARS; OnboardingError(RuntimeError); WalletError(OnboardingError)
  ensure_wallet(cfg: ChainConfig, *, interactive=False, wallet_factory=None) -> wallet  # bittensor lazy; non-interactive + missing keys raises with btcli instructions; interactive -> create_if_non_existent
  register(chain) -> uid  # idempotent; burned_register(wallet, netuid, wait both) via chain.subtensor, then re-sync
  bucket_creds_from_env(env=None) -> BucketCreds (READ pair, the on-chain one); write_creds_from_env(env=None) -> BucketCreds (private write pair)
  commit_bucket_credentials(chain, creds) -> bool  # ChainClient.ensure_bucket_committed

fleet/onboarding/init_publish.py:
  DEFAULT_INIT_SEED=42; INIT_WINDOW=0; InitPublishError
  async build_and_publish_init(cfg, storage|None, chain|None, *, local_dir, seed=42, device='cpu', backend='reference', manifest_hash='', spec_version=1) -> state_root  # init_model -> hash_named_tensors -> Checkpointer.save(window=0, masters, fresh ReplicatedOuterStep zeros, CheckpointMeta) -> upload when storage -> chain.commit_manifest_hash(root) via to_thread when chain
  async fetch_and_verify_init(storage, chain|None, expected_root, *, local_dir, bucket=None, owner_uid=None) -> (state, outer, meta)  # Checkpointer.load_latest; window==0 + meta.state_root + recomputed hash must ALL equal expected (bitwise); owner_uid -> chain.get_manifest_hash cross-check

fleet/calibration/local_harness.py (production loopback rig, promoted from the test_window_runner fixture pattern):
  MemoryStorage(root, creds, *, cfg: StorageConfig|None, clock=time.time)  # filesystem-backed StorageClient stand-in, async: put_bytes/get_bytes(expected_hash,max_bytes)/upload_file/download_file/object_exists/object_timestamp(injected clock)/list_keys(sorted)/gather_bytes(uid-ascending GatherResult, reason prefixes 'missing: '/'integrity: '/'too_large: '/'error: '), same mok_core.storage error types; cross-bucket reads via shared root; aenter/aexit/aclose parity
  ScriptedChain(uid=0): commit_window records; get_window_commits(window); block_hash deterministic blake2b; my_uid/sign/current_block
  LoopbackClock(seconds_per_window=1000, *, gate_offset_s=10): boundary_ts/now/enter_gate(w)
  make_compressor(model, cfg) / make_outer_step(model, cfg)  # DENSE_SUFFIX-aware run builders
  local_manifest(cfg, index, *, shard_path, run_seed, run_id='local-calibration') -> RunManifest  # ref derived from actual shard files
  LocalLoopbackHarness(model, cfg, manifest, index, *, shard_path, work_dir, uid=0, device='cpu', clock=None, metrics=None, checkpoint=True, cert_timeout_s=10)  # real WindowRunner self_leader=True over MemoryStorage+ScriptedChain; async run_window(window, state) enters the gate then delegates

fleet/calibration/rehearsal.py:
  CalibrationReport(FrozenModel): windows, loss_curve (final/window), entry_losses, window_wall_times, capacity_utils, state_roots (post-outer), determinism_check; .ok
  run_calibration_windows(n_windows, cfg, manifest, *, model, index, shard_path, work_dir, uid=0, start_window=0, device='cpu', timer=perf_counter, determinism_probe=True, metrics=None) -> CalibrationReport  # sync driver over the harness; non-clean window -> CalibrationError; determinism probe = follow-on window trained twice via run_training_phase (theta restored), roots must match

fleet/calibration/sweep.py:
  SweepPoint(FrozenModel: fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size); SweepResult(point, mok: MoKRuntimeConfig, mean_window_s, final_loss); DEFAULT_TUNED_PATH=subnet/configs/mok_tuned.yaml
  default_grid() -> 9 points (sms {24,36,48} x mb {2048,4096,8192}); apply_point(cfg, point) -> RunConfig (fully re-validated)
  run_sweep(cfg, manifest, *, model_factory, shard_path, points=None, windows_per_point=2, start_window=0, uid=0, device='cpu', timer=perf_counter) -> list[SweepResult]  # times run_training_phase windows (theta restored -> identical work per point)
  select_best(results) -> fastest, ties -> earlier; emit_tuned_yaml(mok, path=DEFAULT_TUNED_PATH, *, provenance) -> Path  # 'mok:' overlay + provenance comment, loadable by load_run_config

fleet/calibration/adam_ab.py:
  DEFAULT_K=5; DEFAULT_THRESHOLD_NATS=0.01
  ABReport(FrozenModel): n_windows, k, threshold_nats, losses_reset_every_window, losses_reset_every_k, delta_final_loss (=final(reset1)-final(resetK)), keep_reset_every_window (delta<threshold), recommendation ('inner.adam_reset_every_windows=1'|'=K')
  run_arm(model, cfg, manifest, *, n_windows, reset_every, shard_path, uid=0, start_window=0, device='cpu') -> list[float]  # injected-optimizer mirror of InnerLoop.run_window (reuses its _check_plan/_clip_grad_norm); reset_every=1 arm bitwise-pinned against real InnerLoop in tests
  run_adam_ab(n_windows, cfg, manifest, *, model, shard_path, k=5, threshold_nats=0.01, uid=0, start_window=0, device='cpu') -> ABReport  # both arms from deepcopies of the same theta

fleet/ops/healthcheck.py:
  GPU_QUERY_FIELDS; MAX_GPU_TEMP_C=88; MAX_CLOCK_OFFSET_S=1.0; MIN_DISK_FREE_BYTES=500e9
  HealthCheck/HealthReport(FrozenModel; .ok; .to_json())
  gpu_health(query_csv, *, max_temp_c) -> HealthCheck (ECC uncorrected>0 or over-temp fails; '[N/A]' ECC = note); nvlink_health(topo_text); clock_sync(text) (chronyc tracking offset OR timedatectl synchronized, auto-detected); async storage_reachability(storage, bucket, key='manifest.json') (missing object still reachable); disk_space(path, *, min_free_bytes, disk_usage)
  async run_healthchecks(*, gpu_csv/topo_text/clock_text=None -> live probes, storage/bucket, cache_dir, min_free_bytes, runner, disk_usage, clock) -> HealthReport
  build_parser(); main(argv) -> int  # loop CLI: one JSON report per --interval; --iterations N; canned-input flags --gpu-csv/--topo/--clock-status; exit 0 iff last report ok

fleet/container/: Dockerfile (2-stage FROM nvidia/cuda:13.0.0-devel-ubuntu22.04; py3.12 venv; torch==2.10.0+cu130; mixture-of-kittens @8f90b74 --no-build-isolation; dist/mok_subnet wheel; determinism env + TORCHINDUCTOR_CACHE_DIR baked, MAX_AUTOTUNE=0; LABEL org.mok.spec_version=1; ENTRYPOINT entrypoint.sh), entrypoint.sh (set -euo pipefail; MOK_CONTAINER_DIGEST required + /opt/mok/IMAGE_DIGEST self-check; roles miner|validator|auditor|attest|calibrate|healthcheck; torchrun --standalone --nproc-per-node=$MOK_NPROC_PER_NODE(8) for miner/attest; exec everywhere), compose.yml (profiles miner/validator/auditor + healthcheck sidecar; nvidia count:8 reservations; restart unless-stopped; env_file .env; shared shard-cache/wallet volumes)

fleet/cli.py (console scripts already wired in pyproject):
  attest_main: challenge (--block-hash+--block | --from-chain --config) | respond (--challenge --backend --device) | verify (--challenge --response --expected-root --issued-ts --received-ts; exit 0/1)
  onboard_main: --config/--overlay, --interactive, --skip-{preflight,wallet,register,bucket,init,attest}, --owner-uid, --expected-init-root, --local-dir, --backend, --device; JSON step lines; self-attest = run challenge twice (determinism) + deadline dry-run; exit 1 on failed attest
  init_publish_main: --config --local-dir --seed --backend --device --spec-version [--local-only skips R2+chain]; emits {'init_state_root': ...}
  calibrate_main: rehearse|sweep|adam-ab with common --config/--data-dir(shard_index.json + content-addressed shards)/--work-dir/--run-seed/--seed/--uid/--start-window/--backend/--device; sweep --sms/--minibatch/--windows-per-point/--tuned-out; rehearse exit 0 iff report.ok
```

## apps

```
subnet.miner.bootstrap (shared by all three roles): OWNER_UID=0; INIT_SEED=42; AUDITOR_COMMITMENT='auditor.v1' (plain chain-commitment tag; auditors never WindowCommit so the slot persists — trust model in module docstring); BootstrapError. Signer Protocol {hotkey; sign(bytes)->bytes; verify(hotkey,data,sig)->bool}; LocalSigner(hotkey) keyed-blake2b (harness only); ChainSigner(chain,hotkey). ChainWindowClock(chain,manifest,*,block_time_s=12) — WindowClock over on-chain boundary-block timestamps, cached, future boundaries extrapolated. Local-harness stack (public, reused by tests): LoopbackClock(genesis,window_s,now_ts).set(ts); MemoryStorage(creds,*,store,max_payload_bytes) — full StorageClient surface (put/get_bytes, upload/download_file, object_timestamp/exists, list_keys, gather_bytes->GatherResult) over a shared (bucket,key)->(bytes,ts) dict; ScriptedChain(*,clock,start_block,blocks_per_window,block_time_s=None,my_uid,buckets,stakes,manifest_hashes) — full ChainClient surface incl. commitment decode + per-window commit/vote HISTORY dicts (window_commits, votes, weights_calls public for tests); LocalHarness Protocol {cfg,manifest,creds,chain,storage,clock,data_dir,bucket_for(uid)}; _FallbackHarness.create(cfg,*,root,uid,...) seeds a synthetic verified dataset + canonical manifest into memory storage (used when fleet's LocalLoopbackHarness lacks the bootstrap `create` contract — it does; duck-checked). dataset_index_key(name)='datasets/<name>/shard_index.json' (owner bucket; content verified via verify_index_matches_ref so the key is not consensus). NodeContext dataclass(role,cfg,manifest,uid,signer,chain,storage,own_bucket,shard_caches:dict[name,ShardCache],shard_indexes,fetch_fns:dict[name,FetchFn],metrics,comm:RunnerComm,clock:WindowClock,rank,world_size,protocol_world_size,device,state_dir,local,dev_insecure) with .run_seed, .owner_bucket(), .peer_buckets(), .leader_uid()/.leader_bucket() (resolve_leader_uid: max stake, tie->lowest uid), async .aclose(). async bootstrap(role,argv,*,harness=None)->NodeContext — argparse(--config/--overlay*/--netuid/--network/--local-harness/--uid(local only)/--dev-insecure/--device/--state-dir), enforce_determinism() FIRST, load_run_config+overlays, RANK env -> dist.init_process_group+TorchDistRunnerComm else SingleNodeComm, manifest fetched from owner bucket verified against on-chain hash (canonical), config_hash(cfg)==manifest.config_hash enforced (warn in local/dev-insecure), assert_container_digest unless local/--dev-insecure, uid from chain.my_uid() or --uid, Metrics + per-dataset ShardCache/fetch_fn construction. Helpers: choose_backend(device) ('mok' iff cuda+wheel importable, loud fallback log), build_node_model, build_compressor, build_outer_step, load_master_state(model,state) bitwise strict, async materialize_replica(ctx,checkpointer)->(model,outer,from_window) — newest checkpoint > fleet.onboarding.fetch_and_verify_init(storage,None,manifest.init_checkpoint_hash,local_dir=state_dir/'init',bucket=owner) > fresh seed-42 init root-checked; from_window=-1 means θ_init (fleet stores θ_init AS window 0 — any restored root==init hash maps to -1); async catch_up_replica(ctx,model,outer,*,from_window,to_window)->CatchUpReport; auditor_uids_from_chain(chain); storage_fetch_fn(storage,bucket,index). ||| subnet.miner.app: RESTART_EXIT_CODE=3; MinerApp(ctx,*,self_leader=None(defaults ctx.local),max_windows=None,on_window=None,cert_poll_s,cert_timeout_s,catchup_retries,catchup_retry_s,window_poll_s) with async run()->int, stop(); rebuilds WindowRunner per window (phase-correct cache/fetch), run_state from consensus run_state_at, warmup_null_windows for fresh uids implemented by forcing the gate closed (_GateClock — trains full hot path, never publishes), prefetch task for window+1 shards concurrent with run_window, blocking chain calls via asyncio.to_thread, post-window debug slices + telemetry publication, desync->catch_up_replica with retries then run_state_at resync, restart_required->SystemExit(3), SIGTERM/SIGINT->final checkpoint+exit 0. ||| subnet.validator: ValidatorApp(ctx,*,max_windows,on_window,catchup_retries,catchup_retry_s,poll_s).run(); PAYLOAD_VERSION=1; processes window w after its gate closes (trails head by 1): gate_check+payload fetch/validate->SlashLedger events (missing/received/invalid/sync_behind on payload.meta.state_root mismatch), leader publishes certificate+aggregator FIRST, WindowEvaluator.evaluate_window (EvalRecord: own/random loss before/after applying decompressed Δ at outer.lr*eval_lr_factor to a scratch θ restored via CpuSnapshot, gradient_score+binary_indicator) -> BinaryEMA + OpenSkillBook, overlap check via sparse indices+object timestamps, catch_up(w-1,w) bitwise apply, sync scores from miner debug slices (match->1.0, divergent->sync_score(max), missing->sync_score(1)), leader: debug slices+checkpoint cadence+canonical probe loss (EvalPools.random_pool with fixed PROBE_BLOCK_HASH=bytes(32)) -> SpikeDetector -> RollbackStateMachine + chain.commit_vote, activated decision -> SystemExit(3); every scoring.windows_per_weights: weights.submit_weights (compute_weights ladder, skips empty); audit ingest of window w-report_deadline at apply_window=w; ValidatorState JSON persistence each window (ema/book/ledger state_dicts, probe-loss replay for SpikeDetector, rollback SM attribute-exact). evaluator.EvalRecord/WindowEvaluator; weights.weights_for/submit_weights; leader.CommitView/LeaderDuties(is_leader, publish_certificate, publish_debug_slices, maybe_checkpoint, observe_probe_loss)/PROBE_BLOCK_HASH; audit_ingest.AuditVerdict/collect_audit_verdicts/ingest_window_audits(storage,chain,window,ledger,*,apply_window,verify_signatures) — hotkey-signature-verified, one verdict per (auditor,miner), quorum via SlashLedger.audit_verdicts. ||| subnet.auditor: AUDIT_WALL_BUDGET_FRACTION=0.5; AuditorApp(ctx,*,max_windows,on_window,wall_budget_s,catchup_retries,catchup_retry_s,poll_s).run() — materialize_replica like a miner but NEVER trains; publishes AUDITOR_COMMITMENT; per window after gate close: audit_sampler(run_seed, block_hash(boundary(w+1)), w, commit uids, cfg.audit.probability, sorted auditors∪{self}) -> own tasks; per task: prefetch miner plan shards, WindowReplayer.replay with explicit run_state_at(protocol_world_size), sign_report(chain signer), put_audit_report to OWN bucket; wall-time budget skips + telemetry; then catch_up(w-1,w) applies the outer step. mok-miner/mok-validator/mok-auditor console scripts already wired in pyproject to subnet.<role>.main:main.
```

## gpu

```
tests/gpu/conftest.py: pytest_collection_modifyitems auto-applies @pytest.mark.gpu to every item under tests/gpu. Fixtures (all session-scoped): dist_ctx -> DistCtx(rank, world_size, local_rank, device, comm: TorchDistRunnerComm; .barrier()) — skips without torchrun env (RANK/WORLD_SIZE) or CUDA; runs enforce_determinism BEFORE CUDA context, torch.cuda.set_device(LOCAL_RANK), init_process_group('nccl', device_id=...), teardown barrier+destroy. mok_available -> imported `mok` package — skips unless CUDA + get_device_capability(0)==(10,3) + wheel importable. toy_cfg -> RunConfig from subnet/configs/base.yaml + toy4L.yaml via load_run_config (CPU-safe). shared_tmp -> job-keyed path identical on all ranks (TORCHELASTIC_RUN_ID/MASTER_PORT), rank-0 mkdir/rmtree + barriers. toy_dataset(dist_ctx, shared_tmp) -> ToyData(manifest: RunManifest, index: DatasetShardIndex, data_dir: Path) — rank-0 writes 8 shards x 128 seq x 4096 tok, barrier, all ranks verify. || tests/gpu/_synthetic.py (importable by tests AND a torchrun entry script for test_04): constants RUN_SEED=bytes(range(32)), INIT_SEED=42, UID=3, WINDOW=2, SEQ_LEN=4096, NUM_SHARDS=8, SEQS_PER_SHARD=128; shard_array/write_shard_files/build_index/build_manifest (test_inner_loop 4-token-cycle pattern at toy4L geometry); make_shard_lookup_factory(data_dir) -> WindowReplayer-compatible ctx factory; load_toy_run_config(*, inner_steps=None, routed_precision=None, adam_reset_every_windows=None) -> RunConfig; prepare_mok_model(model) (first MXFP8 requant, no-op bf16); build_mok_model(cfg, device) (init_model seed-42 backend='mok' + requant); run_toy_window(cfg, manifest, data_dir, *, rank, world_size, device, comm, window=2, uid=3, model=None) -> (state_root|None, model); CLI `torchrun --standalone --nproc-per-node=8 tests/gpu/_synthetic.py --data-dir D [--inner-steps 5]` prints STATE_ROOT=<hex64> on rank 0 (self-sys.path-bootstrapping). || test_02 exports reshard_reference_to_mok(ref_model, mok_model, rank) — the explicit ep=1 -> EP-N layout transform (replicated verbatim, `.routed_` tensors take block [rank*E_local,(rank+1)*E_local)); TOLERANCES = {'bf16': (0.05 loss, 0.999 cos, 0.02 norm-rtol), 'mxfp8': (0.25, 0.99, 0.10)}. test_01 exports mxfp8_quantize_reference(x_bf16) — the inlined 32-block quantize reference, verified bitwise-equal to mixture-of-kittens tests/utils. test_04 exports run_torchrun(script_args, *, nproc, extra_env, timeout_s) — nested --standalone torchrun helper with elastic-env scrubbing. || anneal/release_fork.py: fork_release(checkpoint_dir, out_dir, *, manifest: RunManifest, cfg: RunConfig, anneal_dataset='anneal', decay_tokens: int, effective_window: int, committed_block: int, phase_name='anneal_release', fork_run_id: str|None=None) -> ForkResult(manifest: RunManifest, manifest_path, runbook_path, decay_start_step, peak_lr). Loads checkpoint meta.json (CheckpointMeta), validates config_hash(cfg)==manifest.config_hash, meta.manifest_hash==manifest.manifest_hash(), dataset exists, decay_tokens>0, effective_window>meta.window; builds LRSpec(kind='wsd_linear_decay', peak_lr=<pre-fork phase LR>, warmup_steps=run_state_at(cfg, manifest, effective_window, world_size=cfg.model.ep_size).global_inner_step, decay_total_tokens=decay_tokens) inside a PhaseEntry(data=anneal_dataset) applied via manifest.with_amendment(kind='phase', ...); writes out_dir/manifest.json = canonical_bytes(forked) (file hash == manifest_hash; FileExistsError on overwrite) + out_dir/RELEASE_FORK.md operator runbook (release-branch launch vs continue-main). ForkError(ValueError); load_checkpoint_meta(dir) -> CheckpointMeta; MANIFEST_FILENAME/RUNBOOK_FILENAME/DEFAULT_PHASE_NAME/META_FILENAME; build_parser(); main(argv) -> int (0 ok, 2 precondition failure): --checkpoint --out --manifest --config [--overlay ...] --decay-tokens --effective-window --committed-block [--anneal-dataset --phase-name --fork-run-id]; `python -m anneal.release_fork`.
```
