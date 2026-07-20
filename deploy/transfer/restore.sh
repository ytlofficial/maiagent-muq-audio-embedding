#!/usr/bin/env bash
set -euo pipefail

transfer_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-.}"

verify_checksum() {
  local checksum_file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "${transfer_dir}" && sha256sum -c "${checksum_file}")
  elif command -v shasum >/dev/null 2>&1; then
    (cd "${transfer_dir}" && shasum -a 256 -c "${checksum_file}")
  else
    echo "error: sha256sum or shasum is required" >&2
    exit 1
  fi
}

mkdir -p "${destination}"
if [[ -f "${transfer_dir}/archive.sha256" ]]; then
  verify_checksum archive.sha256
  archive_name="$(awk 'NR == 1 {print $2}' "${transfer_dir}/archive.sha256")"
  [[ -f "${transfer_dir}/${archive_name}" ]] || {
    echo "error: archive is missing: ${archive_name}" >&2
    exit 1
  }
  tar -xzf "${transfer_dir}/${archive_name}" -C "${destination}"
else
  [[ -f "${transfer_dir}/parts.sha256" ]] || {
    echo "error: archive.sha256 or parts.sha256 is required" >&2
    exit 1
  }
  verify_checksum parts.sha256
  parts=("${transfer_dir}"/*.tar.gz.part-*)
  [[ -e "${parts[0]}" ]] || {
    echo "error: no archive parts found" >&2
    exit 1
  }
  cat "${parts[@]}" | tar -xzf - -C "${destination}"
fi
echo "restored_to=$(cd "${destination}" && pwd)"
