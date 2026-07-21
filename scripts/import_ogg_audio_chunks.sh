#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SIMAI_AUDIO_ONLY_SOURCE_FORMAT="${SIMAI_AUDIO_ONLY_SOURCE_FORMAT:-ogg}"
exec "${SCRIPT_DIR}/import_audio_chunks.sh" "$@"
