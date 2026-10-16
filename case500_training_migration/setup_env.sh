#!/usr/bin/env bash
# Recreate the tested liuy Python/CUDA package family in this package's venv.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.11}
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This installer targets Linux x86_64 with NVIDIA GPUs.' >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Use Python 3.11 (set PYTHON_BIN to its executable).')
PY
if [[ ! -d "$ROOT/.venv" ]]; then
  "$PYTHON_BIN" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit('Existing .venv is not Python 3.11; use a clean package/environment.')
PY
"$ROOT/.venv/bin/python" -m pip install 'pip<26'
"$ROOT/.venv/bin/python" -m pip install 'torch==2.1.2' --index-url https://download.pytorch.org/whl/cu121
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/python" "$ROOT/scripts/check_environment.py"
echo 'Environment ready. No training was started.'
