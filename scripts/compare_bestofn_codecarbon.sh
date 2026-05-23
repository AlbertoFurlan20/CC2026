#!/usr/bin/env bash

# Best-of-N comparison with CodeCarbon coverage.
# Fixed (top_p, temperature); sweeps vLLM n_samples ∈ {1, 3}.
#
# Usage:
#   bash scripts/compare_bestofn_codecarbon.sh
#
# Env overrides:
#   STORAGE_DIR     base dir for logs/workspace/results (default: repo root)
#   TASK            MLAgentBench task name             (default: cifar10)
#   LLM_NAME        primary LLM name                   (default: llama-3.1-8B-Instruct)
#   FAST_LLM_NAME   fast LLM name                      (default: llama-3.1-8B-Instruct)
#   MAX_STEPS       max agent steps per run            (default: 30)
#   TOP_P           fixed top_p                        (default: 0.9)
#   TEMPERATURE     fixed temperature                  (default: 0.7)
#   N_RUNS          repetitions per n_samples value    (default: 3)
#   N_SAMPLES_LIST  space-separated N values           (default: "1 3")
#   EXP_NAME        experiment name                    (default: bestofn)
#   OUTPUT_DIR      results subdir                     (default: results)
#   PYTHON          python binary                      (default: python)

set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_DIR="${STORAGE_DIR:-${REPO_ROOT}}"

TASK="${TASK:-cifar10}"
LLM_NAME="${LLM_NAME:-llama-3.1-8B-Instruct}"
FAST_LLM_NAME="${FAST_LLM_NAME:-llama-3.1-8B-Instruct}"
MAX_STEPS="${MAX_STEPS:-30}"
TOP_P="${TOP_P:-0.9}"
TEMPERATURE="${TEMPERATURE:-0.7}"
N_RUNS="${N_RUNS:-3}"
N_SAMPLES_LIST="${N_SAMPLES_LIST:-1 3}"
EXP_NAME="${EXP_NAME:-bestofn}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

echo "Using STORAGE_DIR = ${STORAGE_DIR}"
mkdir -p "${STORAGE_DIR}/logs" "${STORAGE_DIR}/workspace"

RESULTS_DIR="${STORAGE_DIR}/${OUTPUT_DIR}/${EXP_NAME}_codecarbon"
RESULTS_FILE="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}"

if [ ! -f "${RESULTS_FILE}" ]; then
    echo "n_samples,run,top_p,temperature,run_id,log_dir,work_dir,score,wall_time_s,avg_time_per_step,emissions_kg,energy_kwh,cc_duration_s,status" \
        > "${RESULTS_FILE}"
fi

run_once() {
    local n_samples="$1"
    local run_idx="$2"

    local tp_safe="${TOP_P//./}"
    local tm_safe="${TEMPERATURE//./}"
    local ts run_id log_dir work_dir stdout_file
    ts=$(date +%s)
    run_id="bon_n${n_samples}_r${run_idx}_tp${tp_safe}_tm${tm_safe}_${ts}"
    log_dir="${STORAGE_DIR}/logs/${run_id}"
    work_dir="${STORAGE_DIR}/workspace/${run_id}"
    stdout_file="${log_dir}/stdout.txt"
    mkdir -p "${log_dir}" "${work_dir}"

    local start_time end_time wall_time status score avg_step
    local emissions_kg energy_kwh cc_duration_s
    start_time=$(date +%s)
    status="ok"

    ${PYTHON} -m MLAgentBench.runner \
        --task "${TASK}" \
        --log-dir "${log_dir}" \
        --work-dir "${work_dir}" \
        --llm-name "${LLM_NAME}" \
        --fast-llm-name "${FAST_LLM_NAME}" \
        --edit-script-llm-name "${LLM_NAME}" \
        --max-steps "${MAX_STEPS}" \
        --use-codecarbon \
        --top-p "${TOP_P}" \
        --temperature "${TEMPERATURE}" \
        --n-samples "${n_samples}" \
        > "${stdout_file}" 2>&1 \
        || status="failed"

    end_time=$(date +%s)
    wall_time=$(( end_time - start_time ))

    score="NA"
    if [ "${status}" = "ok" ]; then
        local eval_json="${log_dir}/eval.json"
        ${PYTHON} -m MLAgentBench.eval \
            --log-folder "${log_dir}" \
            --task "${TASK}" \
            --output-file "${eval_json}" \
            --eval-intermediate \
            > "${log_dir}/eval.log" 2>&1 || true

        score=$(EVAL_JSON="${eval_json}" ${PYTHON} - <<'PYEOF'
import json, os
try:
    d = json.load(open(os.environ["EVAL_JSON"]))
    vals = []
    for v in d.values():
        if not isinstance(v, dict):
            continue
        fs = v.get("final_score", -1)
        if isinstance(fs, (int, float)) and fs >= 0:
            vals.append(fs)
        else:
            step_scores = [s for s in (v.get("score") or []) if isinstance(s, (int, float)) and s >= 0]
            if step_scores:
                vals.append(max(step_scores))
    print(max(vals) if vals else "NA")
except Exception:
    print("NA")
PYEOF
        )
    fi

    avg_step=$(awk "BEGIN { if (${MAX_STEPS}+0 > 0) printf \"%.3f\", ${wall_time}/${MAX_STEPS}; else print \"NA\" }")
    echo "${score}" > "${log_dir}/.score"

    read -r emissions_kg energy_kwh cc_duration_s < <(CC_DIR="${log_dir}/codecarbon" ${PYTHON} - <<'PYEOF'
import csv, glob, os
cc_dir = os.environ["CC_DIR"]
em = en = du = 0.0
found = False
for path in glob.glob(os.path.join(cc_dir, "*.csv")):
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                found = True
                try: em += float(row.get("emissions") or 0)
                except (TypeError, ValueError): pass
                try: en += float(row.get("energy_consumed") or 0)
                except (TypeError, ValueError): pass
                try: du += float(row.get("duration") or 0)
                except (TypeError, ValueError): pass
    except Exception:
        continue
print(f"{em:.9f} {en:.9f} {du:.3f}" if found else "NA NA NA")
PYEOF
)

    echo "${n_samples},${run_idx},${TOP_P},${TEMPERATURE},${run_id},${log_dir},${work_dir},${score},${wall_time},${avg_step},${emissions_kg},${energy_kwh},${cc_duration_s},${status}" \
        >> "${RESULTS_FILE}"

    echo "[n=${n_samples} run=${run_idx}] score=${score} time=${wall_time}s emissions=${emissions_kg}kg energy=${energy_kwh}kWh status=${status}"
    echo "  log_dir: ${log_dir}"
    echo "  work_dir: ${work_dir}"
}

read -ra N_LIST <<<"${N_SAMPLES_LIST}"
N_TOTAL=$(( ${#N_LIST[@]} * N_RUNS ))

echo ""
echo "=== Best-of-N (CodeCarbon): n_samples ∈ {${N_SAMPLES_LIST}} × ${N_RUNS} runs = ${N_TOTAL} experiments ==="
echo "=== Fixed: top_p=${TOP_P} temperature=${TEMPERATURE} task=${TASK} ==="
echo "=== Results CSV: ${RESULTS_FILE} ==="
echo "=== Log dir: ${STORAGE_DIR}/logs ==="
echo "=== Work dir: ${STORAGE_DIR}/workspace ==="
echo ""

for n_samples in "${N_LIST[@]}"; do
    echo ""
    echo "--- n_samples=${n_samples} ---"
    for (( run_idx = 1; run_idx <= N_RUNS; run_idx++ )); do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting n=${n_samples} run=${run_idx}..."
        run_once "${n_samples}" "${run_idx}"
        sleep 1
    done
done

echo ""
echo "=== All runs complete. Results: ${RESULTS_FILE} ==="
echo ""
column -t -s ',' "${RESULTS_FILE}" 2>/dev/null \
    || cat "${RESULTS_FILE}"
