#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-maiagent-muq-audio:torch2.11.0-cu128}"
gpu_devices="${GPU_DEVICES:-all}"

docker run --rm \
  --gpus "${gpu_devices}" \
  --entrypoint python \
  "${image_name}" \
  /opt/maiagent/scripts/check_cuda_runtime.py
