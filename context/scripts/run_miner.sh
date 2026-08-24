#!/usr/bin/env bash
# Context extension = the training-run miner under the 16k overlay (fresh MoK
# workspace materializes from num_local_tokens=16384; clean relaunch required).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m subnet.miner.main \
  --config subnet/configs/base.yaml --overlay context/configs/context16k.yaml --overlay subnet/configs/mok_tuned.yaml "$@"
