#!/usr/bin/env python3
"""Read comparison grid JSON, emit cartesian-product rows for shell loops.

Output (stdout): TSV with columns
    config_idx<TAB>key1=val1<TAB>key2=val2<TAB>...
One row per (cell, run) pair, run index appended as last col.

Companion env metadata lines printed to stderr in KEY=value form so the shell
can `eval` them (n_runs, fixed.task, fixed.llm_name, fixed.fast_llm_name,
fixed.max_steps, experiment.name, output_dir).
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: emit_grid_combos.py <config.json>", file=sys.stderr)
        return 2

    cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())

    grid = cfg.get("grid", {}) or {}
    fixed = cfg.get("fixed", {}) or {}
    execution = cfg.get("execution", {}) or {}
    experiment = cfg.get("experiment", {}) or {}

    n_runs = int(execution.get("n_runs", 1))

    print(f"N_RUNS={n_runs}", file=sys.stderr)
    print(f"EXP_NAME={experiment.get('name', 'experiment')}", file=sys.stderr)
    print(f"OUTPUT_DIR={experiment.get('output_dir', 'results')}", file=sys.stderr)
    print(f"FIXED_TASK={fixed.get('task', 'cifar10')}", file=sys.stderr)
    print(f"FIXED_LLM={fixed.get('llm_name', 'llama-3.1-8B-Instruct')}", file=sys.stderr)
    print(f"FIXED_FAST_LLM={fixed.get('fast_llm_name', fixed.get('llm_name', 'llama-3.1-8B-Instruct'))}", file=sys.stderr)
    print(f"FIXED_MAX_STEPS={fixed.get('max_steps', 30)}", file=sys.stderr)

    if not grid:
        combos = [{}]
    else:
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        combos = [dict(zip(keys, vs)) for vs in product(*values)]

    for idx, combo in enumerate(combos, start=1):
        kv = "\t".join(f"{k}={v}" for k, v in combo.items())
        print(f"{idx}\t{kv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
