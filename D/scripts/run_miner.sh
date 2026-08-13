#!/usr/bin/env bash
# Step D anneal = step C's miner under the anneal overlay. The authoritative
# switch is the on-chain manifest phase entry; this wrapper serves rehearsals.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m C.miner.main \
  --config C/configs/base.yaml --overlay D/configs/anneal.yaml --overlay C/configs/mok_tuned.yaml "$@"
