#!/usr/bin/env bash

# Usage: bash scripts/compare_strategies.sh [CONFIG_JSON]
# - Example: bash scripts/compare_strategies.sh configs/comparison_grid.json
#
# JSON-driven strategy dispatcher. Bayesian configs are delegated to
# run_bayesian.py; the existing grid branch appends one long-format row per
# (config × run) to results/<exp_name>__summary.csv.

set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG_JSON="${1:-configs/comparison_grid.json}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_DIR="${STORAGE_DIR:-${REPO_ROOT}}"
mkdir -p "${STORAGE_DIR}/logs" "${STORAGE_DIR}/workspace"

if [[ "${CONFIG_JSON}" = /* ]]; then
    CONFIG_PATH="${CONFIG_JSON}"
else
    CONFIG_PATH="${REPO_ROOT}/${CONFIG_JSON}"
fi

# Guard: STORAGE_DIR on docker overlay FS = not bind-mounted, data lost on container rm.
if df -T "${STORAGE_DIR}" 2>/dev/null \
    | awk 'NR > 1 && $2 == "overlay" { found=1 } END { exit(found ? 0 : 1) }'; then
    echo "ERROR: STORAGE_DIR=${STORAGE_DIR} is on docker overlay FS (not bind-mounted from host)." >&2
    echo "       Data will be lost when container is removed." >&2
    echo "       Fix: add '-v /data:/data' to docker run, or set STORAGE_DIR to a bind-mounted path." >&2
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "config not found: ${CONFIG_PATH}" >&2
    exit 1
fi

# Route before loading grid-specific metadata. Existing configs without a
# search.method remain grid configs for backwards compatibility.
SEARCH_METHOD=$("${PYTHON}" - "${CONFIG_PATH}" <<'PYEOF'
import json
import sys

config_path = sys.argv[1]
try:
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
except (OSError, json.JSONDecodeError) as exc:
    print(f"failed to read config {config_path}: {exc}", file=sys.stderr)
    raise SystemExit(2)

if not isinstance(config, dict):
    print("config root must be an object", file=sys.stderr)
    raise SystemExit(2)

search = config.get("search", {})
if not isinstance(search, dict):
    print("config field 'search' must be an object", file=sys.stderr)
    raise SystemExit(2)

method = search.get("method", "grid")
if not isinstance(method, str) or not method.strip():
    print("config field 'search.method' must be a non-empty string", file=sys.stderr)
    raise SystemExit(2)

print(method.strip().lower())
PYEOF
)

case "${SEARCH_METHOD}" in
    grid)
        ;;
    bayesian)
        exec "${PYTHON}" "${REPO_ROOT}/scripts/run_bayesian.py" "${CONFIG_PATH}"
        ;;
    *)
        echo "unsupported search.method '${SEARCH_METHOD}'; expected 'grid' or 'bayesian'" >&2
        exit 2
        ;;
esac

# region load grid from JSON helper
META_TMP="$(mktemp)"
COMBOS_TMP="$(mktemp)"
cleanup() {
    local status=$?
    rm -f -- "${META_TMP}" "${COMBOS_TMP}"
    trap - EXIT
    exit "${status}"
}
trap cleanup EXIT

${PYTHON} "${REPO_ROOT}/scripts/emit_grid_combos.py" "${CONFIG_PATH}" \
    > "${COMBOS_TMP}" 2> "${META_TMP}"

# Parse the fixed, known metadata keys as data. Avoid `source`/`eval`: besides
# executing config-derived text, process substitution can race on older Bash.
N_RUNS=1
EXP_NAME=experiment
OUTPUT_DIR=results
FIXED_TASK=cifar10
FIXED_LLM=llama-3.1-8B-Instruct
FIXED_FAST_LLM=llama-3.1-8B-Instruct
FIXED_MAX_STEPS=30
while IFS='=' read -r key value; do
    case "${key}" in
        N_RUNS) N_RUNS="${value}" ;;
        EXP_NAME) EXP_NAME="${value}" ;;
        OUTPUT_DIR) OUTPUT_DIR="${value}" ;;
        FIXED_TASK) FIXED_TASK="${value}" ;;
        FIXED_LLM) FIXED_LLM="${value}" ;;
        FIXED_FAST_LLM) FIXED_FAST_LLM="${value}" ;;
        FIXED_MAX_STEPS) FIXED_MAX_STEPS="${value}" ;;
    esac
done < "${META_TMP}"

TASK="${FIXED_TASK}"
LLM_NAME="${FIXED_LLM}"
FAST_LLM_NAME="${FIXED_FAST_LLM}"
MAX_STEPS="${FIXED_MAX_STEPS}"
# endregion

RESULTS_DIR="${STORAGE_DIR}/${OUTPUT_DIR}/${EXP_NAME}"
RESULTS_FILE="${RESULTS_DIR}/summary.csv"
mkdir -p "${RESULTS_DIR}"

if [ ! -f "${RESULTS_FILE}" ]; then
    echo "config_idx,run,top_p,temperature,run_id,log_dir,work_dir,score,wall_time_s,avg_time_per_step,status" \
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
    run_id="c${config_idx}_r${run_idx}_tp${tp_safe}_tm${tm_safe}_${ts}"
    log_dir="${STORAGE_DIR}/logs/${run_id}"
    work_dir="${STORAGE_DIR}/workspace/${run_id}"
    stdout_file="${log_dir}/stdout.txt"
    mkdir -p "${log_dir}" "${work_dir}"

    local extra_args="--top-p ${top_p} --temperature ${temperature}"

    local start_time end_time wall_time status score avg_step
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
    echo "${config_idx},${run_idx},${top_p},${temperature},${run_id},${log_dir},${work_dir},${score},${wall_time},${avg_step},${status}" \
        >> "${RESULTS_FILE}"

    echo "[cfg=${config_idx} run=${run_idx}] tp=${top_p} tm=${temperature} score=${score} time=${wall_time}s status=${status}"
    echo "  log_dir: ${log_dir}"
    echo "  work_dir: ${work_dir}"
}
# endregion

# region main loop
N_CONFIGS=$(wc -l < "${COMBOS_TMP}" | tr -d '[:space:]')
echo ""
echo "=== Grid: ${N_CONFIGS} configs × ${N_RUNS} runs = $((N_CONFIGS * N_RUNS)) experiments ==="
echo "=== Config: ${CONFIG_JSON} ==="
echo "=== Results CSV: ${RESULTS_FILE} ==="
echo "=== Log dir: ${STORAGE_DIR}/logs ==="
echo "=== Work dir: ${STORAGE_DIR}/workspace ==="
echo ""

while IFS=$'\t' read -r config_idx col1 col2; do
    # parse "key=val" pairs
    top_p="${col1#top_p=}"
    temperature="${col2#temperature=}"

    echo ""
    echo "--- Config ${config_idx}/${N_CONFIGS}: top_p=${top_p} temperature=${temperature} ---"

    for (( run_idx = 1; run_idx <= N_RUNS; run_idx++ )); do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting cfg=${config_idx} run=${run_idx}..."
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
