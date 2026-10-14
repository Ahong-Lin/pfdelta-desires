#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON_BIN:-python3}
julia_bin=${JULIA_BIN:-julia}
if [[ -z "${JULIA_BIN:-}" && -x "$bundle_dir/.runtime/julia-1.9.4/bin/julia" ]]; then
  julia_bin="$bundle_dir/.runtime/julia-1.9.4/bin/julia"
fi
"$python_bin" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11), "Use Python 3.10 or 3.11 (set PYTHON_BIN)."'
if ! command -v "$julia_bin" >/dev/null 2>&1; then
  echo 'Julia 1.9.4 is required. On Linux x86_64 run: bash scripts/install_julia.sh' >&2
  exit 1
fi
"$julia_bin" --startup-file=no -e 'VERSION == v"1.9.4" || error("This bundle locks Julia 1.9.4; set JULIA_BIN to that version.")'
if [[ ! -x "$bundle_dir/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$bundle_dir/.venv"
fi
"$bundle_dir/.venv/bin/python" -m pip install -r "$bundle_dir/requirements.txt"
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-"$bundle_dir/.julia_depot"}
export JULIA_NUM_THREADS=${JULIA_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=1
"$julia_bin" --startup-file=no --project="$bundle_dir/data_generation" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using PowerModels, Ipopt, JSON; println("Julia environment ready")'
"$bundle_dir/.venv/bin/python" -m unittest discover -s "$bundle_dir/tests" -v
echo 'Environment ready. Next: bash run.sh plan; then a smoke run on persistent storage.'
