#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
IMPORT_SCRIPT="${SCRIPT_DIR}/build_segment_chunk_audio_lancedb.py"
FFMPEG="${SIMAI_AUDIO_FFMPEG:-${REPO_ROOT}/.venv/bin/ffmpeg}"

is_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing virtualenv python: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${IMPORT_SCRIPT}" ]]; then
  echo "missing audio import script: ${IMPORT_SCRIPT}" >&2
  exit 1
fi

if [[ ! -x "${FFMPEG}" ]]; then
  echo "missing ffmpeg executable: ${FFMPEG}" >&2
  echo "install it with: ${PYTHON} -m pip install imageio-ffmpeg" >&2
  exit 1
fi

args=(
  --ranges-dir "${SIMAI_AUDIO_RANGES_DIR:-${REPO_ROOT}/outputs/segmentation_ranges/charts}"
  --measures-dir "${SIMAI_AUDIO_MEASURES_DIR:-${REPO_ROOT}/outputs/simai_measures/charts}"
  --chartdata-root "${SIMAI_AUDIO_CHARTDATA_ROOT:-${REPO_ROOT}/chartdata-rebuilt}"
  --audio-filename "${SIMAI_AUDIO_FILENAME:-track.mp3}"
  --fallback-audio-filename "${SIMAI_AUDIO_FALLBACK_FILENAME:-track.ogg}"
  --audio-out-dir "${SIMAI_AUDIO_OUT_DIR:-${REPO_ROOT}/outputs/audio_chunks/simai_audio_chunks}"
  --audio-suffix "${SIMAI_AUDIO_SUFFIX:-.mp3}"
  --converted-source-dir "${SIMAI_AUDIO_CONVERTED_SOURCE_DIR:-${REPO_ROOT}/outputs/audio_chunks/converted_sources}"
  --db-path "${SIMAI_AUDIO_DB_PATH:-${REPO_ROOT}/outputs/lancedb/simai_pattern_chunks}"
  --table "${SIMAI_AUDIO_TABLE:-simai_audio_chunks}"
  --index-table "${SIMAI_AUDIO_INDEX_TABLE:-simai_audio_chunk_index}"
  --mode "${SIMAI_AUDIO_MODE:-overwrite}"
  --batch-size "${SIMAI_AUDIO_BATCH_SIZE:-128}"
  --progress-every "${SIMAI_AUDIO_PROGRESS_EVERY:-25}"
  --ffmpeg "${FFMPEG}"
  --mp3-quality "${SIMAI_AUDIO_MP3_QUALITY:-2}"
)

if is_enabled "${SIMAI_AUDIO_REUSE_EXISTING:-1}"; then
  args+=(--reuse-existing-audio)
fi

if is_enabled "${SIMAI_AUDIO_SKIP_MISSING:-1}"; then
  args+=(--skip-missing-audio)
fi

if is_enabled "${SIMAI_AUDIO_STORE_BYTES:-0}"; then
  args+=(--store-audio-bytes)
fi

if is_enabled "${SIMAI_AUDIO_NO_INDEX_TABLE:-0}"; then
  args+=(--no-index-table)
fi

if [[ -n "${SIMAI_AUDIO_LIMIT_REPORTS:-}" ]]; then
  args+=(--limit-reports "${SIMAI_AUDIO_LIMIT_REPORTS}")
fi

if [[ -n "${SIMAI_AUDIO_ONLY_SOURCE_FORMAT:-}" ]]; then
  args+=(--only-source-audio-format "${SIMAI_AUDIO_ONLY_SOURCE_FORMAT}")
fi

if is_enabled "${SIMAI_AUDIO_LOG_SOURCE_FORMAT_SKIPS:-0}"; then
  args+=(--log-source-format-skips)
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" "${IMPORT_SCRIPT}" "${args[@]}" "$@"
