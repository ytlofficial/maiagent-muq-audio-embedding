#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-maiagent-muq-audio:torch2.5.1-cu124}"
gpu_devices="${GPU_DEVICES:-all}"

docker run --rm \
  --gpus "${gpu_devices}" \
  --entrypoint python \
  "${image_name}" \
  -c "import torch, torchaudio, muq; assert torch.cuda.is_available(); print({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'torchaudio': torchaudio.__version__, 'muq': 'ok'})"
