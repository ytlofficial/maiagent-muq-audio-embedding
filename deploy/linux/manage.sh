#!/usr/bin/env bash
set -euo pipefail

deploy_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${deploy_dir}/../.." && pwd)"
env_file="${DEPLOY_ENV_FILE:-${deploy_dir}/deploy.env}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

if [[ -f "${env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${env_file}"
  set +a
fi

image_name="${IMAGE_NAME:-maiagent-muq-audio:torch2.11.0-cu128}"
config_file="${CONFIG_FILE:-../../configs/training.example.yaml}"
data_root="${DATA_ROOT:-}"
gpu_devices="${GPU_DEVICES:-all}"
shm_size="${SHM_SIZE:-16g}"
tensorboard_port="${TENSORBOARD_PORT:-6006}"
run_as_host_user="${RUN_AS_HOST_USER:-1}"
include_weights="${INCLUDE_MUQ_WEIGHTS:-1}"
model_id="${MUQ_MODEL_ID:-OpenMuQ/MuQ-large-msd-iter}"
pypi_index_url="${PYPI_INDEX_URL:-https://pypi.org/simple}"
pytorch_index_url="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
hf_hub_offline="${HF_HUB_OFFLINE:-1}"
transformers_offline="${TRANSFORMERS_OFFLINE:-1}"

resolve_deploy_path() {
  local path="$1"
  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
  else
    printf '%s\n' "${deploy_dir}/${path}"
  fi
}

config_path="$(resolve_deploy_path "${config_file}")"

require_docker() {
  require_command docker
  docker info >/dev/null 2>&1 || die "Docker daemon is unavailable"
}

require_image() {
  docker image inspect "${image_name}" >/dev/null 2>&1 || \
    die "image ${image_name} is unavailable; run ${0} build"
}

resolve_data_root() {
  [[ -n "${data_root}" ]] || die "set DATA_ROOT in ${env_file}"
  [[ "${data_root}" = /* ]] || die "DATA_ROOT must be an absolute path"
  [[ -d "${data_root}" ]] || die "DATA_ROOT is not a directory: ${data_root}"
  data_root="$(cd "${data_root}" && pwd)"
}

check_runtime_paths() {
  resolve_data_root
  [[ -f "${config_path}" ]] || die "training config not found: ${config_path}"
}

build_image() {
  require_docker
  docker build \
    --platform linux/amd64 \
    --file "${repo_root}/docker/Dockerfile" \
    --tag "${image_name}" \
    --build-arg "INCLUDE_MUQ_WEIGHTS=${include_weights}" \
    --build-arg "MUQ_MODEL_ID=${model_id}" \
    --build-arg "PYPI_INDEX_URL=${pypi_index_url}" \
    --build-arg "PYTORCH_INDEX_URL=${pytorch_index_url}" \
    "${repo_root}"
}

common_run_args() {
  RUN_ARGS=(
    --rm
    --init
    --shm-size "${shm_size}"
    --volume "${data_root}:/workspace"
    --volume "${config_path}:/deploy/training.yaml:ro"
    --workdir /workspace
    --env "HF_HUB_OFFLINE=${hf_hub_offline}"
    --env "TRANSFORMERS_OFFLINE=${transformers_offline}"
    --env PYTHONUNBUFFERED=1
  )
  if [[ "${run_as_host_user}" = "1" ]]; then
    RUN_ARGS+=(
      --user "$(id -u):$(id -g)"
      --env HOME=/tmp
    )
  fi
}

run_training() {
  local use_gpu="$1"
  shift
  check_runtime_paths
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
    /opt/maiagent/scripts/check_rtx5090_runtime.py
}

print_help() {
  cat <<'EOF'
Usage: deploy/linux/manage.sh ACTION [training CLI overrides]

Actions:
  build        Build the CUDA image from this repository.
  doctor       Check Docker, CUDA, runtime paths, DB joins, and audio paths.
  dry-run      Validate train/validation/test metadata without MuQ or CUDA.
  smoke        Run a small one-epoch CUDA training test.
  train        Start full training from the YAML configuration.
  resume PATH  Resume a checkpoint path relative to DATA_ROOT.
  tensorboard  Serve all run logs on TENSORBOARD_PORT.
  config       Print resolved deployment and training configuration.
  shell        Open a CUDA shell with DATA_ROOT mounted at /workspace.
  help         Show this message.
EOF
}

action="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${action}" in
  build)
    build_image
    ;;
  doctor)
    require_docker
    check_runtime_paths
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
    [[ $# -gt 0 ]] || die "resume requires a checkpoint path relative to DATA_ROOT"
    checkpoint="$1"
    shift
    [[ "${checkpoint}" != /* ]] || die "use a checkpoint path relative to DATA_ROOT"
    output_dir="$(dirname "${checkpoint}")"
    require_docker
    run_training 1 --resume "${checkpoint}" --output-dir "${output_dir}" "$@"
    ;;
  tensorboard)
    require_docker
    check_runtime_paths
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
    echo "DEPLOY_ENV_FILE=${env_file}"
    echo "IMAGE_NAME=${image_name}"
    echo "CONFIG_FILE=${config_path}"
    echo "DATA_ROOT=${data_root:-<not set>}"
    echo "GPU_DEVICES=${gpu_devices}"
    echo "SHM_SIZE=${shm_size}"
    echo "TENSORBOARD_PORT=${tensorboard_port}"
    echo "INCLUDE_MUQ_WEIGHTS=${include_weights}"
    echo
    sed -n '1,240p' "${config_path}"
    ;;
  shell)
    require_docker
    check_runtime_paths
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
