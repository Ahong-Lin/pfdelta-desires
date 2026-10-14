#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# PYTHON_BIN selects the base interpreter during setup. Runtime must use the
# environment where setup_env.sh actually installed this bundle's dependencies.
python_bin="$bundle_dir/.venv/bin/python"
[[ -x "$python_bin" ]] || { echo 'Run bash setup_env.sh first to create the project virtual environment.' >&2; exit 1; }
exec "$python_bin" "$bundle_dir/scripts/run_pipeline.py" "$@"
