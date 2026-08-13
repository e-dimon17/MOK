"""Dataset layer: Merkle commitments, the assignment PRF, verified shard IO,
precomputed window schedules, and the local shard cache."""

from .assignment import (
    effective_run_seed,
    sample_order,
    sequences_per_window,
    shard_ids,
    tokens_per_shard,
)
from .download import FetchFn, ShardCache, ShardCacheError, ShardVerificationError
from .merkle import MerkleTree, Proof
from .shards import (
    DatasetShardIndex,
    ShardReader,
    shard_filename,
    shard_leaf_hash,
    verify_index_matches_ref,
)
from .window_dataset import WindowBatchPlan

__all__ = [
    "DatasetShardIndex",
    "FetchFn",
    "MerkleTree",
    "Proof",
    "ShardCache",
    "ShardCacheError",
    "ShardReader",
    "ShardVerificationError",
    "WindowBatchPlan",
    "effective_run_seed",
    "sample_order",
    "sequences_per_window",
    "shard_filename",
    "shard_ids",
    "shard_leaf_hash",
    "tokens_per_shard",
    "verify_index_matches_ref",
]
