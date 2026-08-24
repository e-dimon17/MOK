#!/usr/bin/env bash
# Scoring validator — single process, reference backend (>=141 GB GPU).
set -euo pipefail
cd "$(dirname "$0")/../.."
exec python -m subnet.validator.main \
  --config subnet/configs/base.yaml --overlay subnet/configs/bulk.yaml "$@"
