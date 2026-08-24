#!/usr/bin/env bash
# Quality anneal = the training-run miner under the anneal overlay. The authoritative
# switch is the on-chain manifest phase entry; this wrapper serves rehearsals.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m subnet.miner.main \
  --config subnet/configs/base.yaml --overlay anneal/configs/anneal.yaml --overlay subnet/configs/mok_tuned.yaml "$@"
