#!/usr/bin/env bash
# ============================================================================
# Role dispatch for the blessed container (playbook step B).
#   entrypoint.sh <miner|validator|auditor|attest|calibrate|healthcheck> [args...]
# 8-GPU roles launch under torchrun --nproc-per-node=8; control-plane roles run
# a single python. Env is validated up front so a misconfigured node fails in
# milliseconds with instructions, not minutes into a window.
# ============================================================================
set -euo pipefail

ROLE="${1:-}"
if [ -z "${ROLE}" ]; then
    echo "entrypoint: no role given. Usage: <miner|validator|auditor|attest|calibrate|healthcheck> [args...]" >&2
    exit 64
fi
shift

require_env() {
    local missing=0
    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            echo "entrypoint: required env ${var} is unset (see .env.example)" >&2
            missing=1
        fi
    done
    [ "${missing}" -eq 0 ] || exit 64
}

# --- container digest self-check -------------------------------------------
# MOK_CONTAINER_DIGEST must be set (the manifest pins it; mok_core enforces it
# in-process too). When the image carries its own digest record
# (/opt/mok/IMAGE_DIGEST, written by the release pipeline after push), the two
# must agree — catching a stale env file pointing at an old image.
require_env MOK_CONTAINER_DIGEST
if [ -f /opt/mok/IMAGE_DIGEST ]; then
    BAKED_DIGEST="$(cat /opt/mok/IMAGE_DIGEST)"
    if [ "${BAKED_DIGEST}" != "${MOK_CONTAINER_DIGEST}" ]; then
        echo "entrypoint: MOK_CONTAINER_DIGEST=${MOK_CONTAINER_DIGEST} != baked image digest ${BAKED_DIGEST}" >&2
        echo "entrypoint: refusing to start a mismatched image (lockstep would break)" >&2
        exit 65
    fi
fi

NPROC="${MOK_NPROC_PER_NODE:-8}"
TORCHRUN=(torchrun --standalone --nproc-per-node="${NPROC}")

case "${ROLE}" in
    miner)
        require_env R2_ACCOUNT_ID R2_BUCKET_NAME \
            R2_WRITE_ACCESS_KEY_ID R2_WRITE_SECRET_ACCESS_KEY \
            R2_READ_ACCESS_KEY_ID R2_READ_SECRET_ACCESS_KEY \
            BT_WALLET_NAME BT_WALLET_HOTKEY BT_NETWORK BT_NETUID
        exec "${TORCHRUN[@]}" -m C.miner.main "$@"
        ;;
    validator)
        require_env R2_ACCOUNT_ID R2_BUCKET_NAME \
            R2_WRITE_ACCESS_KEY_ID R2_WRITE_SECRET_ACCESS_KEY \
            BT_WALLET_NAME BT_WALLET_HOTKEY BT_NETWORK BT_NETUID
        exec python -m C.validator.main "$@"
        ;;
    auditor)
        require_env R2_ACCOUNT_ID R2_BUCKET_NAME \
            BT_WALLET_NAME BT_WALLET_HOTKEY BT_NETWORK BT_NETUID
        exec python -m C.auditor.main "$@"
        ;;
    attest)
        # 8-rank reference run; rank 0 prints the AttestationResponse JSON.
        exec "${TORCHRUN[@]}" -m B.attestation.reference_step "$@"
        ;;
    calibrate)
        # rehearse | sweep | adam-ab (single process drives all 8 GPUs via EP).
        exec mok-calibrate "$@"
        ;;
    healthcheck)
        exec python -m B.ops.healthcheck "$@"
        ;;
    *)
        echo "entrypoint: unknown role '${ROLE}' (want miner|validator|auditor|attest|calibrate|healthcheck)" >&2
        exit 64
        ;;
esac
