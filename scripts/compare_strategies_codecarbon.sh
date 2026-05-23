#!/usr/bin/env bash

# Usage: bash scripts/compare_strategies_codecarbon.sh [TASK] [LLM_NAME] [FAST_LLM] [MAX_STEPS]
# - Example: bash scripts/compare_strategies_codecarbon.sh cifar10 llama-3.1-8B-Instruct llama-3.1-8B-Instruct 30
#
# Variant of compare_strategies.sh that enables --use-codecarbon for every run
# and aggregates per-run CodeCarbon CSVs into the comparison results file.

set -euo pipefail

PYTHON="${PYTHON:-python}"

TASK="${1:-cifar10}"
LLM_NAME="${2:-llama-3.1-8B-Instruct}"
FAST_LLM_NAME="${3:-llama-3.1-8B-Instruct}"
MAX_STEPS="${4:-30}"

TOP_P_VALUES=(0.7 0.9 1.0)
BEST_OF_VALUES=(1 3)

RESULTS_DIR="results"
RESULTS_FILE="${RESULTS_DIR}/comparison_codecarbon.csv"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${REPO_ROOT}/${RESULTS_DIR}"

if [ ! -f "${REPO_ROOT}/${RESULTS_FILE}" ]; then
    echo "strategy,top_p,best_of,n_samples,run_id,score,wall_time_s,emissions_kg,energy_kwh,cc_duration_s,status" \
        > "${REPO_ROOT}/${RESULTS_FILE}"
fi

# region run_once [STRATEGY RUN_ID EXTRA_ARGS TOP_P BEST_OF N_SAMPLES]
run_once() {
    local strategy="$1"
    local run_id="$2"
    local extra_args="$3"
    local top_p="${4:-default}"
    local best_of="${5:-default}"
    local n_samples="${6:-default}"

    local log_dir="${REPO_ROOT}/logs/${run_id}"
    local work_dir="${REPO_ROOT}/workspace/${run_id}"
    local stdout_file="${log_dir}/stdout.txt"

    mkdir -p "${log_dir}" "${work_dir}"

    local start_time end_time wall_time status score
    local emissions_kg energy_kwh cc_duration_s

    start_time=$(date +%s)
    status="ok"

    # shellcheck disable=SC2086
    ${PYTHON} -m MLAgentBench.runner \
        --task "${TASK}" \
        --log-dir "${log_dir}" \
        --work-dir "${work_dir}" \
        --llm-name "${LLM_NAME}" \
        --fast-llm-name "${FAST_LLM_NAME}" \
        --edit-script-llm-name "${LLM_NAME}" \
        --max-steps "${MAX_STEPS}" \
        --use-codecarbon \
        ${extra_args} \
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

    echo "${score}" > "${log_dir}/.score"

    # Aggregate CodeCarbon CSVs (one per LLM step + env-action CSVs)
    read -r emissions_kg energy_kwh cc_duration_s < <(CC_DIR="${log_dir}/codecarbon" ${PYTHON} - <<'PYEOF'
import csv, glob, os, sys
cc_dir = os.environ["CC_DIR"]
em_sum = 0.0
en_sum = 0.0
du_sum = 0.0
found = False
for path in glob.glob(os.path.join(cc_dir, "*.csv")):
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                found = True
                try:
                    em_sum += float(row.get("emissions") or 0)
                except (TypeError, ValueError):
                    pass
                try:
                    en_sum += float(row.get("energy_consumed") or 0)
                except (TypeError, ValueError):
                    pass
                try:
                    du_sum += float(row.get("duration") or 0)
                except (TypeError, ValueError):
                    pass
    except Exception:
        continue
if not found:
    print("NA NA NA")
else:
    print(f"{em_sum:.9f} {en_sum:.9f} {du_sum:.3f}")
PYEOF
)

    echo "${strategy},${top_p},${best_of},${n_samples},${run_id},${score},${wall_time},${emissions_kg},${energy_kwh},${cc_duration_s},${status}" \
        >> "${REPO_ROOT}/${RESULTS_FILE}"

    echo "[${strategy}] run_id=${run_id} score=${score} time=${wall_time}s emissions=${emissions_kg}kg energy=${energy_kwh}kWh status=${status}"
}
# endregion

# region best-of-n n=1
echo ""
echo "=== Best-of-N n=1 (1 run, CodeCarbon) ==="
run_once "best_of_n_1" "bon1_cc_$(date +%s)" "" "default" "default" "default"
# endregion

# region best-of-n n=3
echo ""
echo "=== Best-of-N n=3 (3 runs, pick best, CodeCarbon) ==="
best_score=""
best_run=""
for i in 1 2 3; do
    ts=$(date +%s)
    run_id="bon3_cc_${ts}_${i}"
    run_once "best_of_n_3" "${run_id}" "" "default" "default" "default"

    row_score=$(cat "${REPO_ROOT}/logs/${run_id}/.score" 2>/dev/null || echo "NA")
    row_score=$(echo "${row_score}" | tr -d '[:space:]')
    if [ "${row_score}" != "NA" ] && [[ "${row_score}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        if [ -z "${best_score}" ] || \
            awk "BEGIN{exit !(${row_score}+0 > ${best_score}+0)}"; then
            best_score="${row_score}"
            best_run="${run_id}"
        fi
    fi
    sleep 1
done
if [ -z "${best_run}" ]; then
    echo "[best_of_n_3] no scored winner (all runs returned score=NA — check eval.log in each run dir)"
else
    echo "[best_of_n_3] winner → run_id=${best_run} score=${best_score}"
fi
# endregion

# region gridsearch
echo ""
echo "=== GridSearch (${#TOP_P_VALUES[@]} top_p × ${#BEST_OF_VALUES[@]} best_of = $((${#TOP_P_VALUES[@]} * ${#BEST_OF_VALUES[@]})) runs, CodeCarbon) ==="
echo "NOTE: some vLLM versions ignore 'best_of' server-side — verify in run logs if best_of>1 has effect"
for top_p in "${TOP_P_VALUES[@]}"; do
    for best_of in "${BEST_OF_VALUES[@]}"; do
        ts=$(date +%s)
        # Replace dot in top_p for run_id (0.7 → 07)
        tp_safe="${top_p//./}"
        run_id="gs_cc_tp${tp_safe}_bo${best_of}_${ts}"
        extra_args="--top-p ${top_p} --best-of ${best_of}"
        run_once "gridsearch" "${run_id}" "${extra_args}" "${top_p}" "${best_of}" "1"
        sleep 1
    done
done

# endregion

echo ""
echo "=== All runs complete. Results: ${RESULTS_FILE} ==="
echo ""
column -t -s ',' "${REPO_ROOT}/${RESULTS_FILE}" 2>/dev/null \
    || cat "${REPO_ROOT}/${RESULTS_FILE}"
