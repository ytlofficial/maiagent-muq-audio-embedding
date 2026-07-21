#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "missing virtualenv python: ${PYTHON}" >&2
  exit 1
fi

cd "${REPO_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/rebuild_full_pattern_embedding_lancedb_svd512.py" "$@"
