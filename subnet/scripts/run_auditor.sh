#!/usr/bin/env bash
# Audit validator — identical Tier-A node (8xB300), bitwise replay duty.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec torchrun --standalone --nproc-per-node=8 -m subnet.auditor.main \
  --config subnet/configs/base.yaml --overlay subnet/configs/bulk.yaml --overlay subnet/configs/mok_tuned.yaml "$@"
