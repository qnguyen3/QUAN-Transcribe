#!/usr/bin/env bash
# Build the QUAN-ASR Server image.
#
#   ./scripts/build.sh                     # -> quan-asr-server:1.0.0
#   TAG=ghcr.io/me/quan-asr-server:1.0.0 ./scripts/build.sh
#   INSTALL_LM=false ./scripts/build.sh    # ~3x faster build, no beam search
#   CUDA=12.4.1 ./scripts/build.sh         # different CUDA runtime base
#
# Set INSTALL_LM=false to skip building kenlm from source (it needs a C++
# toolchain). The server then only supports greedy decoding.
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${TAG:-quan-asr-server:1.0.0}"
INSTALL_LM="${INSTALL_LM:-true}"
CUDA="${CUDA:-12.6.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

echo "==> building ${TAG}"
echo "    CUDA runtime : ${CUDA}"
echo "    torch index  : ${TORCH_INDEX}"
echo "    kenlm        : ${INSTALL_LM}"

exec docker build \
    -f Dockerfile \
    --build-arg "CUDA=${CUDA}" \
    --build-arg "TORCH_INDEX=${TORCH_INDEX}" \
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
    --build-arg "INSTALL_LM=${INSTALL_LM}" \
    -t "${TAG}" \
    .
