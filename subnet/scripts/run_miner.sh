#!/usr/bin/env bash
# Miner entrypoint — one process per GPU, WORLD == the in-node MoK EP group.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m subnet.miner.main \
  --config subnet/configs/base.yaml --overlay subnet/configs/bulk.yaml --overlay subnet/configs/mok_tuned.yaml "$@"
