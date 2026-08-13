#!/usr/bin/env bash

set -euo pipefail

BASE_URL="https://huggingface.co/datasets/pfdelta/pfdelta/resolve/main"
ROOT_DIR="${1:-data/pfdelta_data}"

shift || true

if [ "$#" -eq 0 ]; then
  CASES=("case30" "case118")
else
  CASES=("$@")
fi

download_and_extract() {
  local name="$1"
  local archive_path="${ROOT_DIR}/${name}.tar.gz"
  local extract_path="${ROOT_DIR}/${name}"

  mkdir -p "${ROOT_DIR}"

  if [ -d "${extract_path}" ]; then
    echo "[skip] ${extract_path} already exists"
    return 0
  fi

  echo "[download] ${name}.tar.gz"
  wget -c -O "${archive_path}" "${BASE_URL}/${name}.tar.gz"

  echo "[extract] ${archive_path}"
  tar -xzf "${archive_path}" -C "${ROOT_DIR}"
}

download_and_extract "shuffle_files"

for case_name in "${CASES[@]}"; do
  download_and_extract "${case_name}"
done

echo "[done] data is available under ${ROOT_DIR}"
