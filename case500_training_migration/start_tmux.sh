#!/usr/bin/env bash
# Usage: ./start_tmux.sh --manifest Results/0-1labels/_control/runs.json --gpus 0,1,2
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/.venv/bin/python"}
SESSION_NAME=${SESSION_NAME:-case500_endpoint}
command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
PYTHON_EXEC=$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session already exists: tmux attach -t $SESSION_NAME" >&2
    exit 1
fi
# Argument validation and read-only preview before starting the persistent shell.
"$PYTHON_EXEC" "$ROOT/scripts/manage_runs.py" "$@" --dry-run
# A pre-existing tmux daemon may hold an older CUDA allocation. Explicitly
# set or unset both variables in the new pane, matching this caller exactly.
RUN_ENV=(env -u CUDA_VISIBLE_DEVICES -u CUDA_DEVICE_ORDER)
if [[ ${CUDA_VISIBLE_DEVICES+x} ]]; then
    RUN_ENV+=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
if [[ ${CUDA_DEVICE_ORDER+x} ]]; then
    RUN_ENV+=("CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER")
fi
printf -v RUN_COMMAND '%q ' "${RUN_ENV[@]}" "$PYTHON_EXEC" "$ROOT/scripts/manage_runs.py" "$@"
printf -v ROOT_QUOTED '%q' "$ROOT"
tmux new-session -d -s "$SESSION_NAME" "cd $ROOT_QUOTED && $RUN_COMMAND; status=\$?; echo Queue_exit_code=\$status; exec bash"
echo "Started. Attach: tmux attach -t $SESSION_NAME"
echo "Detach without stopping: Ctrl-b, then d."
