#!/usr/bin/env bash
# Miner entrypoint — one process per GPU, WORLD == the in-node MoK EP group.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m C.miner.main \
  --config C/configs/base.yaml --overlay C/configs/bulk.yaml --overlay C/configs/mok_tuned.yaml "$@"
