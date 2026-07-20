#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_name="${IMAGE_NAME:-maiagent-muq-audio:torch2.11.0-cu128}"
platform="${PLATFORM:-linux/amd64}"

docker build \
  --progress "${BUILD_PROGRESS:-plain}" \
  --platform "${platform}" \
  --file "${repo_root}/docker/Dockerfile" \
  --tag "${image_name}" \
  --build-arg "INCLUDE_MUQ_WEIGHTS=${INCLUDE_MUQ_WEIGHTS:-1}" \
  --build-arg "MUQ_MODEL_ID=${MUQ_MODEL_ID:-OpenMuQ/MuQ-large-msd-iter}" \
  --build-arg "PYPI_INDEX_URL=${PYPI_INDEX_URL:-https://pypi.org/simple}" \
  --build-arg "PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  "${repo_root}"
