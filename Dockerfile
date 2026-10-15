# syntax=docker/dockerfile:1
# Ultra-fast, Multi-Stage Production CUDA Dockerfile using Astral uv
# Eliminates duplicate 3GB cuDNN apt packages and compiler bloat from runtime image.

ARG CUDA=12.6.1
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

# --------------------------------------------------------------------------- #
# Stage 1: Builder with uv
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA}-base-ubuntu22.04 AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION
ARG TORCH_INDEX

# Grab uv from Astral official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        git \
        build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN uv venv /opt/venv --python python${PYTHON_VERSION}
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install torch --index-url ${TORCH_INDEX}

WORKDIR /build
COPY pyproject.toml README.md ./
COPY app ./app
COPY quan_transcribe ./quan_transcribe
COPY samples ./samples

ARG INSTALL_LM=true
ARG INSTALL_VAD=true
RUN extras="" ; \
    if [ "$INSTALL_LM" = "true" ]; then extras="${extras}lm"; fi ; \
    if [ "$INSTALL_VAD" = "true" ]; then extras="${extras:+$extras,}vad"; fi ; \
    if [ -n "$extras" ]; then \
        uv pip install ".[${extras}]" ; \
    else \
        uv pip install . ; \
    fi

RUN rm -rf /root/.cache/uv

# --------------------------------------------------------------------------- #
# Stage 2: Minimal Production Runtime Image
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA}-base-ubuntu22.04 AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
    && apt-get purge -y software-properties-common \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY app ./app
COPY quan_transcribe ./quan_transcribe
COPY samples ./samples

ENV HF_HOME=/cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=qnguyen3/QUAN-Transcribe-1.0 \
    SERVED_MODEL_NAME=QUAN-Transcribe-1.0 \
    HOST=0.0.0.0 \
    PORT=8000 \
    PRELOAD=true \
    LOG_LEVEL=info \
    OMP_NUM_THREADS=8

HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=5 \
    CMD curl -fsS http://127.0.0.1:${PORT}/ready || exit 1

EXPOSE 8000

ENTRYPOINT ["quan-asr-server", "--serve"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
