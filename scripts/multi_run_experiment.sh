#!/usr/bin/env bash
# Usage: bash multi_run_experiment.sh <exp_path> <task> <num_runs> <num_workers> [extra_args...]
#
# Runs <num_runs> independent multi-agent experiments for <task>.
# Each run spawns <num_workers> parallel worker agents internally.
#
# Example:
#   bash multi_run_experiment.sh logs/cifar10 cifar10 3 4 --llm-name qwen2.5-7b-instruct

set -euo pipefail

exp_path=$1
task=$2
num_runs=$3
num_workers=$4
shift 4

extra_args="${*}"
python=$(which python)

echo "exp_path:    $exp_path"
echo "task:        $task"
echo "num_runs:    $num_runs"
echo "num_workers: $num_workers"
echo "extra_args:  $extra_args"
echo "Logs saved to $exp_path/run_<timestamp>/"

for (( i=0; i<num_runs; i++ ))
do
    ts=$(date +%s)
    run_dir="$exp_path/run_$ts"
    work_dir="workspaces/$run_dir"

    mkdir -p "$run_dir" "$work_dir"

    echo "[Run $((i+1))/$num_runs] log → $run_dir/log"

    $python -u -m MLAgentBench.prepare_task "$task" "$python"

    $python -u -m MLAgentBench.runner \
        --python "$python" \
        --task "$task" \
        --log-dir "$run_dir" \
        --work-dir "$work_dir" \
        --num-workers "$num_workers" \
        $extra_args \
        > "$run_dir/log" 2>&1 &

    sleep 2
done

wait
echo "All $num_runs runs complete."
