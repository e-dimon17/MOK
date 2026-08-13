#!/usr/bin/env bash
# Audit validator — identical Tier-A node (8xB300), bitwise replay duty.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m C.auditor.main \
  --config C/configs/base.yaml --overlay C/configs/bulk.yaml --overlay C/configs/mok_tuned.yaml "$@"
