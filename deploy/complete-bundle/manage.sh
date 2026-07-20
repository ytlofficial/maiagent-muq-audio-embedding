#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
data_root="${bundle_dir}/data"
config_path="${bundle_dir}/training.yaml"
image_name="maiagent-muq-audio:torch2.11.0-cu128"
image_archive="${bundle_dir}/image/maiagent-muq-audio-torch2.11.0-cu128-linux-amd64.tar"
image_checksum="${image_archive}.sha256"
gpu_devices="${GPU_DEVICES:-all}"
shm_size="${SHM_SIZE:-20g}"
tensorboard_port="${TENSORBOARD_PORT:-6006}"
run_as_host_user="${RUN_AS_HOST_USER:-1}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

require_docker() {
  require_command docker
  docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
}

require_image() {
  docker image inspect "${image_name}" >/dev/null 2>&1 || \
    die "image ${image_name} is not loaded; run ./manage.sh load"
}

check_data_layout() {
  local required=(
    "datasets/audio_embedding_charts_1000_300_300_train.csv"
    "datasets/audio_embedding_charts_1000_300_300_validation.csv"
    "datasets/audio_embedding_charts_1000_300_300_test.csv"
    "outputs/lancedb/simai_pattern_chunks"
    "outputs/audio_chunks/simai_audio_chunks"
  )
  local item
  [[ -f "${config_path}" ]] || die "training config is missing: ${config_path}"
  for item in "${required[@]}"; do
    [[ -e "${data_root}/${item}" ]] || die "data path is missing: ${data_root}/${item}"
  done
}

verify_image_archive() {
  [[ -f "${image_archive}" ]] || die "image archive is missing: ${image_archive}"
  [[ -f "${image_checksum}" ]] || die "image checksum is missing: ${image_checksum}"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "${image_archive}")" && sha256sum -c "$(basename "${image_checksum}")")
  elif command -v shasum >/dev/null 2>&1; then
    (cd "$(dirname "${image_archive}")" && shasum -a 256 -c "$(basename "${image_checksum}")")
  else
    die "sha256sum or shasum is required"
  fi
}

common_run_args() {
  RUN_ARGS=(
    --rm
    --init
    --shm-size "${shm_size}"
    --volume "${data_root}:/workspace"
    --volume "${config_path}:/deploy/training.yaml:ro"
    --workdir /workspace
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env PYTHONUNBUFFERED=1
  )
  if [[ "${run_as_host_user}" = "1" ]]; then
    RUN_ARGS+=(--user "$(id -u):$(id -g)" --env HOME=/tmp)
  fi
}

run_training() {
  local use_gpu="$1"
  shift
  check_data_layout
  require_image
  common_run_args
  if [[ "${use_gpu}" = "1" ]]; then
    RUN_ARGS+=(--gpus "${gpu_devices}")
  fi
  docker run "${RUN_ARGS[@]}" "${image_name}" \
    --config /deploy/training.yaml \
    --data-root /workspace \
    "$@"
}

check_gpu() {
  require_image
  docker run --rm \
    --gpus "${gpu_devices}" \
    --entrypoint python \
    "${image_name}" \
    /opt/maiagent/scripts/check_cuda_runtime.py
}

print_help() {
  cat <<'EOF'
Usage: ./manage.sh ACTION [training CLI overrides]

Actions:
  verify       Verify the bundled image and required data layout.
  load         Verify and load the bundled Docker image.
  doctor       Check Docker, CUDA, DB joins, and every audio path.
  dry-run      Validate all metadata and files without loading MuQ or CUDA.
  smoke        Run a small one-epoch CUDA training test.
  train        Start full training from training.yaml.
  resume PATH  Resume a checkpoint path relative to the bundled data root.
  tensorboard  Serve run logs on TENSORBOARD_PORT (default 6006).
  config       Print resolved paths and training configuration.
  shell        Open a CUDA shell with bundled data mounted at /workspace.
EOF
}

action="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${action}" in
  verify)
    verify_image_archive
    check_data_layout
    echo "bundle_verify_complete"
    ;;
  load)
    require_docker
    verify_image_archive
    docker image load --input "${image_archive}"
    ;;
  doctor)
    require_docker
    check_data_layout
    check_gpu
    run_training 0 --dry-run "$@"
    echo "doctor_complete"
    ;;
  dry-run)
    require_docker
    run_training 0 --dry-run "$@"
    ;;
  smoke)
    require_docker
    check_gpu
    run_training 1 --smoke-test "$@"
    ;;
  train)
    require_docker
    run_training 1 "$@"
    ;;
  resume)
    [[ $# -gt 0 ]] || die "resume requires a checkpoint path relative to the bundled data root"
    checkpoint="$1"
    shift
    [[ "${checkpoint}" != /* ]] || die "use a checkpoint path relative to the bundled data root"
    output_dir="$(dirname "${checkpoint}")"
    require_docker
    run_training 1 --resume "${checkpoint}" --output-dir "${output_dir}" "$@"
    ;;
  tensorboard)
    require_docker
    check_data_layout
    require_image
    common_run_args
    docker run "${RUN_ARGS[@]}" \
      --publish "${tensorboard_port}:6006" \
      --entrypoint tensorboard \
      "${image_name}" \
      --logdir /workspace/outputs/audio_embedding_runs \
      --host 0.0.0.0 \
      --port 6006
    ;;
  config)
    echo "BUNDLE_DIR=${bundle_dir}"
    echo "DATA_ROOT=${data_root}"
    echo "IMAGE_NAME=${image_name}"
    echo "IMAGE_ARCHIVE=${image_archive}"
    echo "GPU_DEVICES=${gpu_devices}"
    echo "SHM_SIZE=${shm_size}"
    echo "TENSORBOARD_PORT=${tensorboard_port}"
    echo
    sed -n '1,240p' "${config_path}"
    ;;
  shell)
    require_docker
    check_data_layout
    require_image
    common_run_args
    RUN_ARGS+=(--gpus "${gpu_devices}")
    docker run "${RUN_ARGS[@]}" --entrypoint bash "${image_name}"
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    print_help >&2
    die "unknown action: ${action}"
    ;;
esac
