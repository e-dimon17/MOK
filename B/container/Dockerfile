# syntax=docker/dockerfile:1
# ============================================================================
# The ONE blessed image (playbook step B). Its sha256 digest is pinned in the
# on-chain RunManifest (container_digest); every miner/validator/auditor runs
# THIS image or fails `assert_container_digest` at startup. Reproducibility
# levers: pinned base image, pinned torch wheel, pinned mixture-of-kittens
# commit, the repo wheel built from one git commit, and a PRE-BAKED inductor
# FX-graph cache (runtime autotune stays off — TORCHINDUCTOR_MAX_AUTOTUNE=0 —
# because benchmark-based tuning is timing-dependent and breaks lockstep).
#
# Build (from the repo root, wheel first):
#   python -m build --wheel
#   docker build -f B/container/Dockerfile -t mok-subnet:latest .
# ============================================================================

ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu22.04
ARG MOK_COMMIT=8f90b74
ARG TORCH_SPEC=torch==2.10.0+cu130

# ---------------------------------------------------------------------------
# Stage 1 — builder: python 3.12 venv + torch cu130 + MoK kernel + our wheel
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS builder
ARG MOK_COMMIT
ARG TORCH_SPEC
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common git curl ca-certificates build-essential ninja-build \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/mok/venv
ENV PATH=/opt/mok/venv/bin:$PATH
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Pinned torch cu130 wheel (consensus: AdamW kernel parity is pinned to this torch)
RUN pip install --no-cache-dir ${TORCH_SPEC} --index-url https://download.pytorch.org/whl/cu130

# Pinned MoK megakernel commit (SM103-only build; --no-build-isolation reuses the venv torch)
RUN git clone https://github.com/cursor/mixture-of-kittens /opt/mok/mixture-of-kittens \
    && cd /opt/mok/mixture-of-kittens \
    && git checkout ${MOK_COMMIT} \
    && git submodule update --init --recursive \
    && pip install --no-cache-dir --no-build-isolation .

# The subnet wheel (one wheel == one spec surface == one digest)
COPY dist/mok_subnet-*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/mok_subnet-*.whl && rm -rf /tmp/wheels

# ---------------------------------------------------------------------------
# Stage 2 — runtime: same CUDA base (inductor needs nvcc), venv copied over,
# determinism env pinned, inductor cache dir baked.
# ---------------------------------------------------------------------------
FROM ${CUDA_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates chrony \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/mok/venv /opt/mok/venv

# Determinism pins (mirrors mok_core.determinism.env._REQUIRED_ENV) + the
# baked inductor FX-graph cache location. The cache is populated by running
# the toy attestation graph at build time on a build node with GPUs; on a
# GPU-less builder the step is a no-op and the first fleet run warms it.
ENV PATH=/opt/mok/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    NCCL_ALGO=Ring \
    NCCL_PROTO=Simple \
    TORCHINDUCTOR_MAX_AUTOTUNE=0 \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    TORCHINDUCTOR_CACHE_DIR=/opt/mok/inductor-cache
RUN mkdir -p /opt/mok/inductor-cache \
    && python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

COPY B/container/entrypoint.sh /opt/mok/entrypoint.sh
RUN chmod +x /opt/mok/entrypoint.sh

LABEL org.mok.spec_version="1" \
      org.mok.role-dispatch="miner|validator|auditor|attest|calibrate|healthcheck" \
      org.opencontainers.image.title="mok-subnet" \
      org.opencontainers.image.description="Verified decentralized MoE pretraining (MoK-54B) — blessed runtime"

ENTRYPOINT ["/opt/mok/entrypoint.sh"]
CMD ["healthcheck", "--iterations", "1"]
