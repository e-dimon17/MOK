#!/usr/bin/env bash
# Scoring validator — single process, reference backend (>=141 GB GPU).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python -m C.validator.main \
  --config C/configs/base.yaml --overlay C/configs/bulk.yaml "$@"
