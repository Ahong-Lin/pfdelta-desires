#!/usr/bin/env bash
set -euo pipefail
bundle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash start_tmux.sh /persistent/path/case500 [session-name] [extra run.sh arguments]' >&2
  exit 2
fi
command -v tmux >/dev/null || { echo 'tmux is not installed; ask the server administrator to install it.' >&2; exit 1; }
output_dir=$1
session_name=${2:-case500_generate}
shift
if [[ $# -gt 0 ]]; then shift; fi
[[ "$session_name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid tmux session name' >&2; exit 2; }
if tmux has-session -t "=$session_name" 2>/dev/null; then
  echo "Session exists: tmux attach -t $session_name" >&2
  exit 1
fi
[[ "$output_dir" = /* ]] || { echo 'Use an absolute path on persistent storage.' >&2; exit 2; }
mkdir -p "$output_dir/logs"
log_file="$output_dir/logs/launcher_$(date +%Y%m%d_%H%M%S).log"
printf -v command_line '%q ' bash "$bundle_dir/run.sh" full "$output_dir" "$@"
printf -v quoted_log '%q' "$log_file"
command_line+=" 2>&1 | tee $quoted_log; result=\${PIPESTATUS[0]}; printf '\\nPipeline exit: %s\\n' \"\$result\"; exec bash"
# Explicitly pass environment overrides into the new tmux session, even when
# tmux's server was started before the user's exports.
env_args=()
for key in JULIA_BIN PYTHON_BIN JULIA_DEPOT_PATH JULIA_NUM_THREADS; do
  if [[ -n "${!key:-}" ]]; then env_args+=(env "$key=${!key}"); fi
done
printf -v tmux_command '%q ' "${env_args[@]}" bash -o pipefail -c "$command_line"
tmux new-session -d -s "$session_name" "$tmux_command"
echo "Started: tmux attach -t $session_name"
echo 'Detach without stopping: Ctrl+b, release, then d'
echo "Log: $log_file"
