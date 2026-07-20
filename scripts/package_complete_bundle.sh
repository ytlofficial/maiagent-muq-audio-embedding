#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_source_root="${DATA_SOURCE_ROOT:-}"
output_root="${OUTPUT_ROOT:-${repo_root}/packages}"
bundle_name="${BUNDLE_NAME:-maiagent-muq-audio-complete-torch2.11.0-cu128-linux-amd64}"
bundle_dir="${output_root}/${bundle_name}"
transfer_dir="${output_root}/${bundle_name}-transfer"
image_name="${IMAGE_NAME:-maiagent-muq-audio:torch2.11.0-cu128}"
image_archive_name="maiagent-muq-audio-torch2.11.0-cu128-linux-amd64.tar"
image_dir="${bundle_dir}/image"
image_archive="${image_dir}/${image_archive_name}"
part_size="${PART_SIZE:-1900m}"
stage_mode="${STAGE_MODE:-hardlink}"
transfer_mode="${TRANSFER_MODE:-split}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

for command in docker python3 tar gzip split rsync; do
  require_command "${command}"
done

[[ -n "${data_source_root}" ]] || die "set DATA_SOURCE_ROOT to the private training data root"
[[ "${data_source_root}" = /* ]] || die "DATA_SOURCE_ROOT must be absolute"
[[ -d "${data_source_root}" ]] || die "DATA_SOURCE_ROOT is not a directory: ${data_source_root}"
[[ ! -e "${bundle_dir}" ]] || die "bundle output already exists: ${bundle_dir}"
[[ ! -e "${transfer_dir}" ]] || die "transfer output already exists: ${transfer_dir}"
[[ "${transfer_mode}" = "split" || "${transfer_mode}" = "single" ]] || \
  die "TRANSFER_MODE must be split or single"

required_data=(
  "datasets/audio_embedding_charts_1000_300_300_train.csv"
  "datasets/audio_embedding_charts_1000_300_300_validation.csv"
  "datasets/audio_embedding_charts_1000_300_300_test.csv"
  "outputs/lancedb/simai_pattern_chunks"
  "outputs/audio_chunks/simai_audio_chunks"
)
for relative_path in "${required_data[@]}"; do
  [[ -e "${data_source_root}/${relative_path}" ]] || \
    die "required training data is missing: ${data_source_root}/${relative_path}"
done

docker image inspect "${image_name}" >/dev/null 2>&1 || \
  die "Docker image is unavailable: ${image_name}"

mkdir -p \
  "${bundle_dir}/data/datasets" \
  "${bundle_dir}/data/outputs/lancedb" \
  "${bundle_dir}/data/outputs/audio_chunks" \
  "${image_dir}" \
  "${transfer_dir}"

install -m 0644 "${repo_root}/deploy/complete-bundle/README.md" "${bundle_dir}/README.md"
install -m 0644 "${repo_root}/deploy/complete-bundle/CUDA_COMPATIBILITY.md" "${bundle_dir}/CUDA_COMPATIBILITY.md"
install -m 0755 "${repo_root}/deploy/complete-bundle/manage.sh" "${bundle_dir}/manage.sh"
install -m 0644 "${repo_root}/deploy/complete-bundle/manage.cmd" "${bundle_dir}/manage.cmd"
install -m 0644 "${repo_root}/deploy/complete-bundle/manage.ps1" "${bundle_dir}/manage.ps1"
install -m 0644 "${repo_root}/configs/training.example.yaml" "${bundle_dir}/training.yaml"

for split_file in "${data_source_root}"/datasets/audio_embedding_charts_1000_300_300*; do
  [[ -f "${split_file}" ]] || continue
  install -m 0644 "${split_file}" "${bundle_dir}/data/datasets/$(basename "${split_file}")"
done

stage_tree() {
  local source="$1"
  local destination="$2"
  if [[ "${stage_mode}" = "hardlink" ]]; then
    if cp -al "${source}" "${destination}" 2>/dev/null; then
      return
    fi
    echo "hardlink staging unavailable for ${source}; falling back to rsync copy" >&2
  elif [[ "${stage_mode}" != "copy" ]]; then
    die "STAGE_MODE must be hardlink or copy"
  fi
  mkdir -p "${destination}"
  rsync -a "${source}/" "${destination}/"
}

stage_tree \
  "${data_source_root}/outputs/lancedb/simai_pattern_chunks" \
  "${bundle_dir}/data/outputs/lancedb/simai_pattern_chunks"
stage_tree \
  "${data_source_root}/outputs/audio_chunks/simai_audio_chunks" \
  "${bundle_dir}/data/outputs/audio_chunks/simai_audio_chunks"

echo "saving_image=${image_name}"
docker image save --output "${image_archive}" "${image_name}"
chmod 0644 "${image_archive}"

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
  else
    shasum -a 256 "${path}" | awk '{print $1}'
  fi
}

image_sha256="$(sha256_file "${image_archive}")"
printf '%s  %s\n' "${image_sha256}" "${image_archive_name}" > "${image_archive}.sha256"

docker run --rm --entrypoint python "${image_name}" -c \
  "import json, torch, torchvision, torchaudio, muq, lancedb, librosa; print(json.dumps({'python': __import__('sys').version.split()[0], 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'torchaudio': torchaudio.__version__, 'cuda_runtime': torch.version.cuda, 'cuda_compiled': torch._C._cuda_getCompiledVersion(), 'compiled_arches': str(torch._C._cuda_getArchFlags()).split(), 'cudnn': torch.backends.cudnn.version(), 'muq': '0.1.0', 'lancedb': lancedb.__version__, 'librosa': librosa.__version__}, sort_keys=True))" \
  > "${bundle_dir}/runtime-versions.json"
docker run --rm --entrypoint python "${image_name}" \
  /opt/maiagent/scripts/check_cuda_runtime.py --static-only \
  > "${bundle_dir}/cuda-static-report.json"

python3 - "${bundle_dir}/data" "${bundle_dir}/data-inventory.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])

def inventory(relative):
    path = root / relative
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "path": relative,
        "file_count": len(files),
        "size_bytes": sum(item.stat().st_size for item in files),
    }

groups = [
    inventory("datasets"),
    inventory("outputs/lancedb/simai_pattern_chunks"),
    inventory("outputs/audio_chunks/simai_audio_chunks"),
]
payload = {
    "groups": groups,
    "total_file_count": sum(group["file_count"] for group in groups),
    "total_size_bytes": sum(group["size_bytes"] for group in groups),
}
output.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
PY

image_id="$(docker image inspect "${image_name}" --format '{{.Id}}')"
image_size="$(docker image inspect "${image_name}" --format '{{.Size}}')"
image_architecture="$(docker image inspect "${image_name}" --format '{{.Architecture}}')"
image_archive_size="$(wc -c < "${image_archive}" | tr -d ' ')"
created_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

python3 - \
  "${bundle_dir}/manifest.json" \
  "${bundle_name}" \
  "${created_at}" \
  "${image_name}" \
  "${image_id}" \
  "${image_architecture}" \
  "${image_size}" \
  "${image_archive_name}" \
  "${image_archive_size}" \
  "${image_sha256}" \
  "${bundle_dir}/runtime-versions.json" \
  "${bundle_dir}/data-inventory.json" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    bundle_name,
    created_at,
    image_name,
    image_id,
    image_architecture,
    image_size,
    image_archive,
    image_archive_size,
    image_sha256,
    runtime_path,
    inventory_path,
) = sys.argv[1:]

payload = {
    "bundle": bundle_name,
    "created_at": created_at,
    "self_contained": True,
    "network_required_at_runtime": False,
    "image": {
        "name": image_name,
        "id": image_id,
        "architecture": f"linux/{image_architecture}",
        "size_bytes": int(image_size),
        "archive": f"image/{image_archive}",
        "archive_size_bytes": int(image_archive_size),
        "archive_sha256": image_sha256,
        "muq_weights_embedded": True,
    },
    "runtime": json.loads(Path(runtime_path).read_text(encoding="utf-8")),
    "data": json.loads(Path(inventory_path).read_text(encoding="utf-8")),
    "pipeline": {
        "teacher_dimension": 512,
        "audio_sample_rate": 24000,
        "segments_per_chart": 5,
        "train_charts": 1000,
        "validation_charts": 300,
        "test_charts": 300,
        "automatic_final_test": True,
        "balanced_profile": {
            "epochs": 30,
            "batch_size": 32,
            "validation_batch_size": 64,
            "num_workers": 6,
            "learning_rate": 0.001,
            "retrieval_block_size": 512,
            "shared_memory": "20g",
            "doctor_audio_batch_size": 8,
            "doctor_audio_seconds": 10,
        },
    },
    "entrypoints": {
        "windows": "manage.cmd",
        "linux": "manage.sh",
    },
    "target": {
        "host": "Windows 10/11 or Linux x86_64",
        "container": "linux/amd64",
        "gpu": "NVIDIA GPU compatible with CUDA 12.8",
        "minimum_wsl": "2.1.5",
        "static_compatibility_report": "cuda-static-report.json",
    },
}
Path(output).write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
PY

install -m 0644 "${repo_root}/deploy/transfer/README.md" "${transfer_dir}/README.md"
install -m 0755 "${repo_root}/deploy/transfer/restore.sh" "${transfer_dir}/restore.sh"
install -m 0644 "${repo_root}/deploy/transfer/restore.cmd" "${transfer_dir}/restore.cmd"
install -m 0644 "${repo_root}/deploy/transfer/restore.ps1" "${transfer_dir}/restore.ps1"

archive_stem="${bundle_name}.tar.gz"
export COPYFILE_DISABLE=1
if [[ "${transfer_mode}" = "single" ]]; then
  archive_path="${transfer_dir}/${archive_stem}"
  echo "creating_single_archive=${archive_path}"
  tar -C "${output_root}" -cf - "${bundle_name}" | gzip -1 > "${archive_path}"
  archive_sha256="$(sha256_file "${archive_path}")"
  printf '%s  %s\n' "${archive_sha256}" "${archive_stem}" > "${transfer_dir}/archive.sha256"
else
  part_prefix="${transfer_dir}/${archive_stem}.part-"
  echo "creating_transfer_parts=${part_prefix}*"
  tar -C "${output_root}" -cf - "${bundle_name}" | gzip -1 | \
    split -b "${part_size}" -d -a 3 - "${part_prefix}"

  : > "${transfer_dir}/parts.sha256"
  for part in "${part_prefix}"*; do
    part_sha256="$(sha256_file "${part}")"
    printf '%s  %s\n' "${part_sha256}" "$(basename "${part}")" >> "${transfer_dir}/parts.sha256"
  done
fi

python3 - "${transfer_dir}" "${bundle_name}" "${part_size}" "${transfer_mode}" <<'PY'
import json
import sys
from pathlib import Path

transfer_dir = Path(sys.argv[1])
bundle_name = sys.argv[2]
part_size = sys.argv[3]
transfer_mode = sys.argv[4]
if transfer_mode == "single":
    archive = transfer_dir / f"{bundle_name}.tar.gz"
    payload = {
        "bundle": bundle_name,
        "archive_format": "single tar.gz",
        "archive": archive.name,
        "total_size_bytes": archive.stat().st_size,
        "sha256_file": "archive.sha256",
    }
else:
    parts = sorted(transfer_dir.glob(f"{bundle_name}.tar.gz.part-*"))
    payload = {
        "bundle": bundle_name,
        "archive_format": "tar.gz split into ordered binary parts",
        "part_size_setting": part_size,
        "part_count": len(parts),
        "total_size_bytes": sum(part.stat().st_size for part in parts),
        "parts": [
            {"name": part.name, "size_bytes": part.stat().st_size}
            for part in parts
        ],
    }
(transfer_dir / "transfer-manifest.json").write_text(
    json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
    encoding="utf-8",
)
PY

if command -v perl >/dev/null 2>&1; then
  perl -pi -e 's/\r?\n/\r\n/g' "${transfer_dir}/restore.cmd" "${transfer_dir}/restore.ps1"
fi

echo "bundle_dir=${bundle_dir}"
echo "transfer_dir=${transfer_dir}"
echo "image_sha256=${image_sha256}"
echo "transfer_mode=${transfer_mode}"
if [[ "${transfer_mode}" = "single" ]]; then
  echo "transfer_archive=${transfer_dir}/${archive_stem}"
else
  echo "transfer_parts=$(find "${transfer_dir}" -type f -name '*.part-*' | wc -l | tr -d ' ')"
fi
