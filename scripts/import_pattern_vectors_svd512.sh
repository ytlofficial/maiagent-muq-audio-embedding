#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
REBUILD_SCRIPT="${SCRIPT_DIR}/rebuild_full_pattern_embedding_lancedb_svd512.py"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing virtualenv python: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${REBUILD_SCRIPT}" ]]; then
  echo "missing rebuild script: ${REBUILD_SCRIPT}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" "${REBUILD_SCRIPT}" \
  --components "${SIMAI_VECTOR_COMPONENTS:-512}" \
  --batch-size "${SIMAI_VECTOR_BATCH_SIZE:-256}" \
  --progress-every "${SIMAI_VECTOR_PROGRESS_EVERY:-50}" \
  "$@"
