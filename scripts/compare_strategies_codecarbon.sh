#!/usr/bin/env bash

# Usage: bash scripts/compare_strategies_codecarbon.sh [CONFIG_JSON]
# - Example: bash scripts/compare_strategies_codecarbon.sh configs/comparison_grid.json
#
# Variant of compare_strategies.sh that enables --use-codecarbon for every run
# and aggregates per-run CodeCarbon CSVs into the summary file.

set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG_JSON="${1:-configs/comparison_grid.json}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_DIR="${STORAGE_DIR:-${REPO_ROOT}}"
mkdir -p "${STORAGE_DIR}/logs" "${STORAGE_DIR}/workspace"

if [ ! -f "${REPO_ROOT}/${CONFIG_JSON}" ]; then
    echo "config not found: ${REPO_ROOT}/${CONFIG_JSON}" >&2
    exit 1
fi

# region load grid from JSON helper
META_TMP="$(mktemp)"
COMBOS_TMP="$(mktemp)"
trap 'rm -f "${META_TMP}" "${COMBOS_TMP}"' EXIT

${PYTHON} "${REPO_ROOT}/scripts/emit_grid_combos.py" "${REPO_ROOT}/${CONFIG_JSON}" \
    > "${COMBOS_TMP}" 2> "${META_TMP}"

# shellcheck disable=SC1090
source <(grep -E '^[A-Z_]+=' "${META_TMP}")

N_RUNS="${N_RUNS:-1}"
EXP_NAME="${EXP_NAME:-experiment}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"
TASK="${FIXED_TASK}"
LLM_NAME="${FIXED_LLM}"
FAST_LLM_NAME="${FIXED_FAST_LLM}"
MAX_STEPS="${FIXED_MAX_STEPS}"
# endregion

RESULTS_DIR="${STORAGE_DIR}/${OUTPUT_DIR}/${EXP_NAME}_codecarbon"
RESULTS_FILE="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}"

if [ ! -f "${RESULTS_FILE}" ]; then
    echo "config_idx,run,top_p,temperature,run_id,score,wall_time_s,avg_time_per_step,emissions_kg,energy_kwh,cc_duration_s,status" \
        > "${RESULTS_FILE}"
fi

# region run_once [CONFIG_IDX RUN_IDX TOP_P TEMPERATURE]
run_once() {
    local config_idx="$1"
    local run_idx="$2"
    local top_p="$3"
    local temperature="$4"

    local tp_safe="${top_p//./}"
    local tm_safe="${temperature//./}"
    local ts run_id log_dir work_dir stdout_file
    ts=$(date +%s)
    run_id="cc_c${config_idx}_r${run_idx}_tp${tp_safe}_tm${tm_safe}_${ts}"
    log_dir="${STORAGE_DIR}/logs/${run_id}"
    work_dir="${STORAGE_DIR}/workspace/${run_id}"
    stdout_file="${log_dir}/stdout.txt"
    mkdir -p "${log_dir}" "${work_dir}"

    local extra_args="--top-p ${top_p} --temperature ${temperature}"

    local start_time end_time wall_time status score avg_step
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

    avg_step=$(awk "BEGIN { if (${MAX_STEPS}+0 > 0) printf \"%.3f\", ${wall_time}/${MAX_STEPS}; else print \"NA\" }")
    echo "${score}" > "${log_dir}/.score"

    # Aggregate CodeCarbon CSVs (one per LLM step + env heavy-action CSVs)
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

    echo "${config_idx},${run_idx},${top_p},${temperature},${run_id},${score},${wall_time},${avg_step},${emissions_kg},${energy_kwh},${cc_duration_s},${status}" \
        >> "${RESULTS_FILE}"

    echo "[cfg=${config_idx} run=${run_idx}] tp=${top_p} tm=${temperature} score=${score} time=${wall_time}s emissions=${emissions_kg}kg energy=${energy_kwh}kWh status=${status}"
}
# endregion

# region main loop
N_CONFIGS=$(wc -l < "${COMBOS_TMP}" | tr -d '[:space:]')
echo ""
echo "=== Grid (CodeCarbon): ${N_CONFIGS} configs × ${N_RUNS} runs = $((N_CONFIGS * N_RUNS)) experiments ==="
echo "=== Config: ${CONFIG_JSON} ==="
echo "=== Output: ${RESULTS_FILE} ==="

while IFS=$'\t' read -r config_idx col1 col2; do
    top_p="${col1#top_p=}"
    temperature="${col2#temperature=}"

    echo ""
    echo "--- Config ${config_idx}/${N_CONFIGS}: top_p=${top_p} temperature=${temperature} ---"

    for (( run_idx = 1; run_idx <= N_RUNS; run_idx++ )); do
        run_once "${config_idx}" "${run_idx}" "${top_p}" "${temperature}"
        sleep 1
    done
done < "${COMBOS_TMP}"
# endregion

echo ""
echo "=== All runs complete. Results: ${RESULTS_FILE} ==="
echo ""
column -t -s ',' "${RESULTS_FILE}" 2>/dev/null \
    || cat "${RESULTS_FILE}"
