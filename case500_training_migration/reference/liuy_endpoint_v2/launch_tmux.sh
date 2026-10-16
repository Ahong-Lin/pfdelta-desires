#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PFDELTA_PYTHON:-/data1/liuy/miniconda3/envs/pfdelta311/bin/python}"
SESSION="case118_endpoint_v2"
ACTION="${1:-start}"
CONTROL="$REPO/../Results/0-1labels/_endpoint_v2_control"
case "$ACTION" in
  start)
    command -v tmux >/dev/null
    test -x "$PYTHON"
    mkdir -p "$CONTROL"
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      echo "Session already exists: $SESSION"
      echo "Attach with: tmux attach -t $SESSION"
      echo "If its window has exited, use this script with restart."
      exit 0
    fi
    printf -v RUN_COMMAND '%q %q %q' bash "$SCRIPT_DIR/launch_tmux.sh" inside
    tmux new-session -d -s "$SESSION" -n queue -c "$REPO" "$RUN_COMMAND"
    tmux set-window-option -t "$SESSION:queue" remain-on-exit on
    echo "Started tmux session: $SESSION"
    echo "It prepares caches first, then runs 3 jobs on each of GPU 0, 1 and 2."
    echo "Attach: tmux attach -t $SESSION"
    echo "Detach safely: Ctrl+b, then d (do not press Ctrl+c)."
    ;;
  inside)
    cd "$REPO"
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    "$PYTHON" -u "$SCRIPT_DIR/manage_endpoint_experiments.py" run --prepare 2>&1 | tee -a "$CONTROL/launcher.log"
    ;;
  status)
    "$PYTHON" "$SCRIPT_DIR/manage_endpoint_experiments.py" status
    nvidia-smi
    ;;
  attach) exec tmux attach -t "$SESSION" ;;
  stop) "$PYTHON" "$SCRIPT_DIR/manage_endpoint_experiments.py" stop ;;
  restart)
    if tmux has-session -t "=$SESSION" 2>/dev/null; then
      DEAD="$(tmux list-panes -t "$SESSION" -F '#{pane_dead}')"
      if [[ "$DEAD" != "1" ]]; then
        echo "Session is still alive. Use status/attach; no second queue was started." >&2
        exit 1
      fi
      tmux kill-session -t "$SESSION"
    fi
    exec bash "$SCRIPT_DIR/launch_tmux.sh" start
    ;;
  *) echo "Usage: bash $0 {start|status|attach|stop|restart}" >&2; exit 2 ;;
esac
