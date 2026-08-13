#!/usr/bin/env bash
set -u

run_one() {
  cfg="$1"
  seed="$2"
  run_name="${cfg}_seed${seed}"
  log="queue_logs/${run_name}.log"

  echo "[$(date '+%F %T')] START ${cfg} seed=${seed} run_name=${run_name}"

  python main.py \
    --config "$cfg" \
    --seed "$seed" \
    --functional.run_location="$cfg" \
    --functional.run_name="$run_name" \
    --functional.wandb_enabled=true \
    --functional.wandb_project=pfdelta-case57 \
    --functional.wandb_group=flowcan-case57 \
    > "$log" 2>&1

  status=$?
  echo "[$(date '+%F %T')] DONE ${cfg} seed=${seed} status=${status}"
  return "$status"
}

export -f run_one
mkdir -p queue_logs

printf "%s %s\n" \
  flowcan_case57_task_1_1 1 \
  flowcan_case57_task_1_1 2 \
  flowcan_case57_task_1_1 3 \
  flowcan_case57_task_2_2 1 \
  flowcan_case57_task_2_2 2 \
  flowcan_case57_task_2_2 3 \
  flowcan_case57_task_2_3 1 \
  flowcan_case57_task_2_3 2 \
  flowcan_case57_task_2_3 3 \
| xargs -n 2 -P 3 bash -c 'run_one "$1" "$2"' _
