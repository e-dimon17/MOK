#!/usr/bin/env bash
# Step E context extension = step C's miner under the 16k overlay (fresh MoK
# workspace materializes from num_local_tokens=16384; clean relaunch required).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m C.miner.main \
  --config C/configs/base.yaml --overlay E/configs/context16k.yaml --overlay C/configs/mok_tuned.yaml "$@"
