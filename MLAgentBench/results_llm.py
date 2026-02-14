import json
import os
import glob
import argparse
from pathlib import Path

import pandas as pd


# ─────────────────────────────────────────────
# Default configuration
# (can be overridden with CLI arguments)
# ─────────────────────────────────────────────

#DEFAULT_BASE_DIR = "/mnt/user-data/uploads"
DEFAULT_BASE_DIR = "."
DEFAULT_OUT_XLSX = "output_exel/results.xlsx"

# Name “Short” of tasks ( must adjust to real filenames)
TASKS = [
    "cifar10", "babylm", "clrs", "fathomnet", "feedback",
    "house", "identify", "imdb", "ogbn", "parkinson",
    "spaceship", "vector"
]

# Key of folder names → name of the model
MODELS = {
    "llama": "Llama 3.1 8B",
    "qwen":  "Qwen2.5 7B",
}

# prefix of folder names → type of prompt
#   user1_task_model/...
#   userN_task_model/...
PROMPTS = {
    "user1": "Rigid",    # prompt rígid / strict
    "userN": "Original",  # prompt original
}

USERS = ["user_1", "user_2", "user_3"]


# ─────────────────────────────────────────────
# FUNCTION: find the JSON of an experiment
# ─────────────────────────────────────────────

def find_json(base, prompt_prefix, task, model, user):
    """
    Searches for resultados_llm.json in paths like:
        {base}/{prompt_prefix}_{task}_{model}/{user}/resultados_llm.json
    and some variants.
    """
    base = Path(base)

    candidates = [
        base / f"{prompt_prefix}_{task}_{model}" / user / "resultados_llm.json",
        base / f"{prompt_prefix}_{task}_{model}" / user / "resultado_llm.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Search flexible in case of changes in folder/file names, looking for patterns like:
    pattern = str(base / f"*{task}*{model}*" / user / "*result*llm*.json")
    matches = glob.glob(pattern, recursive=False)
    if matches:
        return Path(matches[0])

    return None


# ─────────────────────────────────────────────
# FUNCTION: Extract key metrics from a resultados_llm.json file
# ─────────────────────────────────────────────

def extract_metrics(json_path: Path):
    """
    Reads a resultados_llm.json and returns a flat dict with:
      - final_score
      - completed (bool)
      - times
      - energy and emissions (CodeCarbon)
      - number of steps
      - error / submitted_final_answer flags
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # JSON has a single key (the internal path), we access its value
    if isinstance(raw, dict):
        data = list(raw.values())[0]
    else:
        raise ValueError(f"Unexpected format in {json_path}")

    final_score   = data.get("final_score", None)
    total_time    = data.get("total_time", None)
    error         = data.get("error", "")
    submitted     = data.get("submitted_final_answer", False)

    extra = data.get("extra", {})
    cc    = extra.get("codecarbon", {})
    totals = cc.get("totals", {})

    emissions_kg   = totals.get("emissions_kg", None)
    energy_kwh     = totals.get("energy_kwh", None)
    gpu_energy_kwh = totals.get("gpu_energy_kwh", None)
    cpu_energy_kwh = totals.get("cpu_energy_kwh", None)
    ram_energy_kwh = totals.get("ram_energy_kwh", None)
    duration_s     = totals.get("duration_s", None)
    n_steps        = len(cc.get("per_step", []))

    # -1.0 usually indicates no valid submission, we can use it to determine if the task was completed or not
    completed = (final_score is not None and final_score != -1.0)

    return {
        "final_score":     final_score,
        "completed":       completed,
        "total_time_s":    total_time,
        "duration_cc_s":   duration_s,
        "n_steps":         n_steps,
        "emissions_kg":    emissions_kg,
        "energy_kwh":      energy_kwh,
        "gpu_energy_kwh":  gpu_energy_kwh,
        "cpu_energy_kwh":  cpu_energy_kwh,
        "ram_energy_kwh":  ram_energy_kwh,
        "error":           error,
        "submitted":       submitted,
        "json_path":       str(json_path),
    }


# ─────────────────────────────────────────────
# PRINCIPAL LOOP OF AGGREGATION AND TABLE GENERATION
# ─────────────────────────────────────────────

def build_dataframe(base_dir: str) -> tuple[pd.DataFrame, list[dict]]:
    """
    Goes through PROMPTS x MODELS x TASKS x USERS and builds:
        - df: DataFrame with all found runs
        - missing: list of dicts with combinations without file
    """

    records = []
    missing = []

    for prompt_prefix, prompt_label in PROMPTS.items():
        for model_key, model_label in MODELS.items():
            for task in TASKS:
                for user in USERS:
                    json_path = find_json(base_dir, prompt_prefix, task, model_key, user)

                    if json_path is None:
                        missing.append({
                            "prompt_prefix": prompt_prefix,
                            "prompt":        prompt_label,
                            "model_key":     model_key,
                            "model":         model_label,
                            "task":          task,
                            "user":          user,
                        })
                        continue

                    try:
                        metrics = extract_metrics(json_path)
                    except Exception as e:
                        print(f"  ❌ Reading ERROR  {json_path}: {e}")
                        continue

                    records.append({
                        "prompt_prefix": prompt_prefix,
                        "prompt":        prompt_label,
                        "model_key":     model_key,
                        "model":         model_label,
                        "task":          task,
                        "user":          user,
                        **metrics,
                    })

    df = pd.DataFrame(records)
    return df, missing


# ─────────────────────────────────────────────
# TABLE GENERATION AND MAIN
# ─────────────────────────────────────────────

def make_tables(df: pd.DataFrame):
    """
    Recibe el DataFrame completo y devuelve:
      t1, t2, t3, t4  (las tablas de resumen)
    """

    # ── Table 1: score per task, modelo and prompt ──────────────────────────
    t1 = df.groupby(["prompt", "model", "task"])["final_score"].agg(
        mean="mean",
        min="min",
        max="max",
        std="std",
        n="count",
    ).round(4)

    # ── Table 2: Completeness rate ──────────────────────────────────────
    t2 = df.groupby(["prompt", "model", "task"])["completed"].agg(
        completadas=lambda x: int(x.sum()),
        total="count",
        tasa_pct=lambda x: round(x.mean() * 100, 1),
    )

    # ── Table 3: average energy and emissions per task and model ─────────────────────────────
    t3 = df.groupby(["prompt", "model", "task"])[
        ["energy_kwh", "gpu_energy_kwh", "emissions_kg", "total_time_s", "n_steps"]
    ].mean().round(5)

    # ── Table 4: global summary by model and prompt ───────────────────────
    def _score_mean_without_minus1(x):
        x = x[x != -1.0]
        return round(x.mean(), 4) if len(x) > 0 else "N/A"

    def _score_max_without_minus1(x):
        x = x[x != -1.0]
        return round(x.max(), 4) if len(x) > 0 else "N/A"

    t4 = df.groupby(["prompt", "model"]).agg(
        tasks_completados=("completed", "sum"),
        total_corridas=("completed", "count"),
        tasa_completitud_pct=("completed", lambda x: round(x.mean() * 100, 1)),
        score_promedio=("final_score", _score_mean_without_minus1),
        score_max=("final_score", _score_max_without_minus1),
        tiempo_promedio_s=("total_time_s", lambda x: round(x.mean(), 1)),
        energia_promedio_kwh=("energy_kwh", lambda x: round(x.mean(), 5)),
        emisiones_promedio_kg=("emissions_kg", lambda x: round(x.mean(), 5)),
    )

    return t1, t2, t3, t4


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Add results from results_llm.json by task, model, and prompt type."
    )
    parser.add_argument(
        "--base-dir",
        default=DEFAULT_BASE_DIR,
        help=f"Base folder where the userX_task_model folders are located (default: {DEFAULT_BASE_DIR})",
    )
    parser.add_argument(
        "--out-xlsx",
        default=DEFAULT_OUT_XLSX,
        help=f"Excel output path (default: {DEFAULT_OUT_XLSX})",
    )

    args = parser.parse_args()

    base_dir = args.base_dir
    out_xlsx = args.out_xlsx

    print(f"📂 Reading results from: {base_dir}")
    df, missing = build_dataframe(base_dir)

    if df.empty:
        print("⚠️  Results not found. No results_llm.json files were found.")
        print(f"   Check that BASE_DIR is correct: {base_dir}")
        if os.path.isdir(base_dir):
            print("\n   Contenido de BASE_DIR:")
            for p in sorted(Path(base_dir).iterdir()):
                print(f"     {p.name}")
        return

    print(f"✅ {len(df)} loaded register ({len(missing)} combinations without results_llm.json)\n")

    # Built Tables
    t1, t2, t3, t4 = make_tables(df)

    # Show a smal resume on console
    print("=" * 70)
    print("TABLE 1 — Final Score average per task, model and prompt")
    print("=" * 70)
    print(t1.to_string())
    print()

    print("=" * 70)
    print("TABLE 2 — Completeness rate (% of runs with submission)")
    print("=" * 70)
    print(t2.to_string())
    print()

    print("=" * 70)
    print("TABLE 3 — Average energy and emissions per task and model")
    print("=" * 70)
    print(t3.to_string())
    print()

    print("=" * 70)
    print("TABLE 4 — Global summary by model and prompt type")
    print("=" * 70)
    print(t4.to_string())
    print()

    # Creates foledr if not exists and export to Excel
    out_path = Path(out_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pases missing to a DataFrame to export it as well (optional)
    df_missing = pd.DataFrame(missing)

    # Export to Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Complete_Data", index=False)
        t1.to_excel(writer, sheet_name="Score_per_task")
        t2.to_excel(writer, sheet_name="Completeness_rate")
        t3.to_excel(writer, sheet_name="Energy_emissions")
        t4.to_excel(writer, sheet_name="Global_summary")
        if not df_missing.empty:
            df_missing.to_excel(writer, sheet_name="Missing_runs", index=False)

    print(f"✅ Exported Excel file to: {out_path}")

    # Show some of the missing combinations if there are any
    if missing:
        print(f"\n⚠️  {len(missing)} combinations without results_llm.json (first 20):")
        for m in missing[:20]:
            print(f"   {m['prompt']} | {m['model']} | {m['task']} | {m['user']}")
        if len(missing) > 20:
            print(f"   ... y {len(missing) - 20} more")


if __name__ == "__main__":
    main()
