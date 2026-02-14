# plot_codecarbon.py
# Usage:
#   python MLAgentBench/plot_codecarbon.py --exp-dir ./<Experiment_Folder> 
#   python MLAgentBench/plot_codecarbon.py --compare ./exp1 ./exp2 --out-dir ./multi_plots
# File must contains the env_log/ and the CodeCarbon CSVs in:
#   <exp>/codecarbon/  and/or  <exp>/env_log/codecarbon/

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================== Helpers ========================

def _find_cc_csvs(exp_dir: Path):
    """
    Searches for CodeCarbon CSVs in:
    -<exp_dir>/codecarbon
    -<exp_dir>/env_log/codecarbon
    """
    candidates = [exp_dir / "codecarbon", exp_dir / "env_log" / "codecarbon"]
    files = []
    for d in candidates:
        if d.exists():
            files += list(d.glob("*.csv"))
    return files


def _guess_task_from_trace(exp_dir: Path) -> Optional[str]:
    """
    Tries to infer the task name from env_log/trace.json.
    e.g., task, task_name, benchmark, dataset, name fields.
    Returns the first non-empty string found.
    """
    tj = exp_dir / "env_log" / "trace.json"
    if not tj.exists():
        return None
    try:
        tr = json.load(open(tj, "r"))
    except Exception:
        return None

    for k in ["task", "task_name", "benchmark", "benchmark_name", "dataset", "name"]:
        v = tr.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    meta = tr.get("meta", {})
    if isinstance(meta, dict):
        for k in ["task", "task_name", "benchmark", "benchmark_name", "dataset", "name"]:
            v = meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _guess_task_from_cc_csv(exp_dir: Path) -> Optional[str]:
    """
    Tries to infer the task by reading 'project_name' from any CodeCarbon CSV.
    e.g. project_name = "cifar10-Execute Script" -> "cifar
    """
    for p in _find_cc_csvs(exp_dir):
        try:
            df = pd.read_csv(p, nrows=1)
            if "project_name" in df.columns:
                pn = str(df["project_name"].iloc[0])
                return pn.split("-", 1)[0].strip() if pn else None
        except Exception:
            pass
    return None

def _guess_agent_from_cc_csv(exp_dir: Path) -> Optional[str]:
    """
    Tries to infer the agent/LLM by reading 'project_name' from any CodeCarbon CSV.
    
    Assuming formats like:
        - "cifar10-ResearchAgent-llama-3.1-8B-Instruct"
        - "task-Agent-Model-with-dashes"
    From there we return the "Agent" part (the second part).
    """
    for p in _find_cc_csvs(exp_dir):
        try:
            df = pd.read_csv(p, nrows=1)
        except Exception:
            continue

        if "project_name" not in df.columns:
            continue

        pn = str(df["project_name"].iloc[0] or "").strip()
        if not pn:
            continue

        parts = pn.split("-")
        # ej: ["cifar10", "ResearchAgent", "llama", "3.1", "8B", "Instruct"]
        if len(parts) >= 3:
            # modelo = everithing that comes after of the task and the agent
            model = "-".join(parts[2:])
            return model.strip()
        elif len(parts) == 2:
            # Simple case: "<something>-<model>"
            return parts[1].strip()

    return None

def normalize_agent_name(name: str) -> str:
    """
    Normalizes agent/model names to canonical aliases for plotting.
    for example:
     - 'Llama 3.1 8B' for any variant of llama-3.1-8B
     - 'GPT-4o' for any variant of gpt-4o
     - 'ResearchAgent-Llama 3.1 8B' for any variant of ResearchAgent-llama-3.1-8B-Instruct
     - 'qwen2.5-7b-isntruct' becomes 'Qwen2.5 7B'
     - 'qwen2.5-14b' becomes 'Qwen2.5 14B'
    """
    if not isinstance(name, str):
        return "Unknown"

    raw = name.strip()
    low = raw.lower()

    # --- Llama 3.1 8B ---
    if "llama" in low and "3.1" in low and "8b" in low:
        return "Llama 3.1 8B"

    # --- Qwen ---
    if "qwen" in low:
        # Qwen 2.5 7B ("qwen2.5-7b-instruct", "qwen-2.5-7B-Instruct", etc)
        if "2.5" in low and "7b" in low:
            return "Qwen2.5 7B"
        
        # Qwen 2.5 14B ("qwen2.5-14b", "qwen-2.5-14B", etc)
        if "2.5" in low and "14b" in low:
            return "Qwen2.5 14B"
        
        # try other qwen models or sizes
        return "Qwen (other)"
        
    # --- In any case of using again GPT (using again openAI APIs) ---
    if "gpt" in low and "4o" in low:
        return "GPT-4o"
    if "gpt" in low and "4" in low and "mini" in low:
        return "GPT-4.1 mini"

    # It coudl be adding any other APIs like Claude, etc.

    # If there is no match any alias, return name as is
    return raw

def _guess_task_from_folder(exp_dir: Path) -> str:
    """
    Heuristic to guess task from folder name.
    Adds here the aliases you use in your runs.
    """
    name = exp_dir.name.lower()
    if "cifar" in name:      return "CIFAR-10"
    if "imdb" in name:       return "IMDB"
    if "vector" in name:     return "Vectorization"
    if "house" in name:      return "House-Price"
    if "bibtex" in name:     return "BibTeX"
    if "titanic" in name:    return "Spaceship-Titanic"
    return "Unknown"


def infer_task_name(exp_dir: Path) -> str:
    """
    Infers the task name robustly:
     1) trace.json
     2) CodeCarbon CSVs
     3) folder name
    Includes an alias map to canonical names.
    """
    t = (_guess_task_from_trace(exp_dir)
         or _guess_task_from_cc_csv(exp_dir)
         or _guess_task_from_folder(exp_dir))

    if not isinstance(t, str) or not t.strip():
        return "Unknown"

    alias = {
        "cifar10": "CIFAR-10", "cifar-10": "CIFAR-10",
        "imdb": "IMDB",
        "vectorization": "Vectorization",
        "house-price": "House-Price", "house_price": "House-Price",
        "bibtex": "BibTeX",
        "spaceship-titanic": "Spaceship-Titanic", "titanic": "Spaceship-Titanic",
    }
    key = t.strip().lower()
    return alias.get(key, t.strip())


def infer_agent_name(exp_dir: Path) -> str:
    """
    tries to infer the agent/model name used in the experiment.
      1) trace.json (agent, agent_name, llm_name, fast_llm_name)
      2) CodeCarbon CSVs (project_name)
      3) folder name heuristics
      Then normalizes the name to a canonical alias. like 'Llama 3.1 8B', 'GPT-4o', etc.
    """
    raw_agent = None

    # 1) trace.json
    tj = exp_dir / "env_log" / "trace.json"
    if tj.exists():
        try:
            tr = json.load(open(tj, "r"))
            for k in ["agent", "agent_name", "llm_name", "fast_llm_name"]:
                v = tr.get(k)
                if isinstance(v, str) and v.strip():
                    raw_agent = v.strip()
                    return normalize_agent_name(raw_agent)
            meta = tr.get("meta", {})
            if isinstance(meta, dict):
                for k in ["agent", "agent_name", "llm_name", "fast_llm_name"]:
                    v = meta.get(k)
                    if isinstance(v, str) and v.strip():
                        raw_agent = v.strip()
                        return normalize_agent_name(raw_agent)
        except Exception:
            pass

    # 2) project_name de CodeCarbon → model (llama-3.1-8B-Instruct, etc.)
    raw_agent = _guess_agent_from_cc_csv(exp_dir)
    if isinstance(raw_agent, str) and raw_agent.strip():
        return normalize_agent_name(raw_agent)

    # 3) fallback file name (old runs / without CSV completely deleted)
    name = exp_dir.name.lower()
    if "gpt4o" in name or "gpt-4o" in name or "gpt4" in name:
        raw_agent = "GPT-4o" if "4o" in name else "GPT-4"
        return normalize_agent_name(raw_agent)
    if "claude" in name:
        raw_agent = "Claude"
        return normalize_agent_name(raw_agent)
    if "langchain" in name:
        raw_agent = "LangChainAgent"
        return normalize_agent_name(raw_agent)
    if "autogpt" in name:
        raw_agent = "AutoGPT"
        return normalize_agent_name(raw_agent)
    if "baseline" in name:
        raw_agent = "Baseline"
        return normalize_agent_name(raw_agent)

    
    return normalize_agent_name("Unknown")


def _pick(summed: pd.Series, candidates, default=0.0):
    """
    Searches robustly for a numeric value in 'summed' with possible column name variants
    (e.g., 'cpu_energy(kWh)').
    """
    for c in candidates:
        if c in summed:
            try:
                return float(summed[c])
            except Exception:
                pass
    bases = [c.replace("(kWh)", "").replace("(kW)", "").strip() for c in candidates]
    for col in summed.index:
        col_norm = col.replace("(kWh)", "").replace("(kW)", "").strip()
        if col_norm in bases:
            try:
                return float(summed[col])
            except Exception:
                pass
    return default


def _parse_one_csv(path: Path):
    """
    Reads a CodeCarbon CSV and returns an aggregated record per file.
    Infers step/source from the name:
    - step_0001_LLM_attempt_1.csv
    - step_0002_Execute_Script.csv
    """
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    s = df.sum(numeric_only=True)

    name = path.name
    step, source = None, "unknown"
    m1 = re.match(r"step_(\d+)_LLM(?:_attempt_(\d+))?\.csv", name)
    m2 = re.match(r"step_(\d+)_(.+)\.csv", name)
    if m1:
        step = int(m1.group(1))
        att = int(m1.group(2)) if m1.group(2) else 1
        source = f"agent_llm_attempt_{att}"
    elif m2:
        step = int(m2.group(1))
        source = f"env_{m2.group(2)}"
    if step is None:
        step = -1

    rec = dict(
        step=step,
        source=source,
        file=str(path),
        duration_s=_pick(s, ["duration", "duration_sec", "duration_s"]),
        emissions_kg=_pick(s, ["emissions", "emissions_kg"]),
        cpu_energy_kwh=_pick(s, ["cpu_energy", "cpu_energy_kWh", "cpu_energy(kWh)"]),
        gpu_energy_kwh=_pick(s, ["gpu_energy", "gpu_energy_kWh", "gpu_energy(kWh)"]),
        ram_energy_kwh=_pick(s, ["ram_energy", "ram_energy_kWh", "ram_energy(kWh)"]),
        energy_kwh=_pick(s, ["energy_consumed", "energy_kwh", "energy_consumed(kWh)"]),
    )
    if rec["energy_kwh"] == 0.0:
        rec["energy_kwh"] = rec["cpu_energy_kwh"] + rec["gpu_energy_kwh"] + rec["ram_energy_kwh"]
    return rec


def _clean_action_name(src_or_obj) -> str:
    """
    Returns a short/pretty action name.
    -for 'env_{...}' tries to extract "name".
    -for 'agent_llm_attempt_x' -> 'LLM (attempt x)'.
    -if no JSON, replaces '_' with space.
    """
    if isinstance(src_or_obj, dict):
        for k in ("name", "tool", "action", "action_name"):
            if isinstance(src_or_obj.get(k), str):
                return src_or_obj[k]

    s = str(src_or_obj)
    if s.startswith("env_"):
        s = s[4:]

    m = re.match(r"agent_llm_attempt_(\d+)$", s)
    if m:
        return f"LLM (attempt {m.group(1)})"

    m = re.search(r'"name"\s*:\s*"([^"]+)"', s) or re.search(r"'name'\s*:\s*'([^']+)'", s)
    if m:
        return m.group(1)

    s = s.replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return textwrap.shorten(s, width=40, placeholder="…")


def load_codecarbon(exp_dir: Path):
    """
    adds CodeCarbon metrics.
    - per_step: sum by step
    - per_source_step: sum by (step, source)
    """
    files = _find_cc_csvs(exp_dir)
    rows = []
    for p in files:
        r = _parse_one_csv(p)
        if r:
            rows.append(r)
    if not rows:
        return None, None

    det = pd.DataFrame(rows).sort_values(["step", "source"], ignore_index=True)

    per_step = (
        det.groupby("step")[["duration_s", "emissions_kg",
                             "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh", "energy_kwh"]]
          .sum()
          .reset_index()
          .sort_values("step")
    )

    per_source_step = (
        det.groupby(["step", "source"])[["duration_s", "emissions_kg",
                                         "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh", "energy_kwh"]]
          .sum()
          .reset_index()
          .sort_values(["step", "source"])
    )
    return per_step, per_source_step


def build_step_labels(exp_dir: Path, per_source_step: pd.DataFrame) -> dict[int, str]:
    """
    Assigns a label per step:
      1) Uses the env action (env_*) with highest energy at that step.
      2) Fills gaps with info from env_log/trace.json, adjusting offset 0/1.
    Returns a dict {step: label}.
    """
    labels: dict[int, str] = {}

    # 1) Labels from environment acctions (env_...)
    env_rows = per_source_step[per_source_step["source"].str.startswith("env_")]
    if not env_rows.empty:
        env_pick = (
            env_rows.sort_values(["step", "energy_kwh"], ascending=[True, False])
            .groupby("step")
            .first()
            .reset_index()
        )
        for _, row in env_pick.iterrows():
            labels[int(row["step"])] = _clean_action_name(row["source"])

    # 2) Labels from trace.json, inferring offset (0-based vs 1-based)
    trace_path = exp_dir / "env_log" / "trace.json"
    if trace_path.exists():
        try:
            tr = json.load(open(trace_path, "r"))
            trace_steps = tr.get("steps", [])
            if trace_steps:
                steps_in_cc = sorted(per_source_step["step"].unique().tolist())
                has_zero = 0 in steps_in_cc

                # if CodeCarbon has step 0, we assume it's 0-based -> offset=0
                # if not, we assume 1-based -> offset=1
                offset = 0 if has_zero else 1

                for idx, st in enumerate(trace_steps):
                    step_no = idx + offset
                    labels.setdefault(
                        int(step_no),
                        _clean_action_name(st.get("tool") or st.get("action") or st),
                    )
        except Exception:
            pass

    return labels



# ================================ Plots (per-exp) ================================

def bars_by_step(per_step: pd.DataFrame, step2label: dict, outdir: Path, title: str):
    
    """
    Bar plot of energy consumption by step, with breakdown CPU/GPU/RAM/Total.
    For the agent LLM view (steps labeled by action taken).
    """

    # Ajust for the steps that actually exist in the env / trace - just to visualize without any retries "action"
    if step2label:
        # last step that appears in the env / trace
        max_env_step = max(int(s) for s in step2label.keys())
        # reaming only steps <= last step in env / trace
        per_step = per_step[per_step["step"] <= max_env_step].copy()
        per_step = per_step.sort_values("step")
    
    x = np.arange(len(per_step))
    cpu = per_step["cpu_energy_kwh"].to_numpy()
    gpu = per_step["gpu_energy_kwh"].to_numpy()
    ram = per_step["ram_energy_kwh"].to_numpy()
    tot = per_step["energy_kwh"].to_numpy()

    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, tot, width, label="Total")
    ax.bar(x - 0.5*width, cpu, width, label="CPU")
    ax.bar(x + 0.5*width, gpu, width, label="GPU")
    ax.bar(x + 1.5*width, ram, width, label="RAM")

    steps = per_step["step"].tolist()
    tick_labels = [step2label.get(int(s), f"Step {int(s)}") for s in steps]
    tick_labels = [t if len(t) <= 22 else (t[:19] + "…") for t in tick_labels]

    ax.set_title(title or "Energy by Step (Agent LLM view)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Energy (kWh)")
    ax.set_xticks(x); ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.legend(title="Metric"); ax.grid(axis="y", linestyle="--", alpha=0.3)

    out = outdir / "energy_by_step.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out}")


def bars_by_action(per_source_step: pd.DataFrame, outdir: Path, title: str):
    """
    Bar plot of energy consumption by tool / action for the environment (aggregated over steps).
    """
    
    df = per_source_step.copy()
    df["Action"] = df["source"].apply(_clean_action_name)
    # Excluye intentos del LLM si no quieres verlos en "por acción":
    df = df[~df["Action"].str.match(r"^\s*LLM\b", na=False)]

    agg = (df.groupby("Action")[["energy_kwh", "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh"]]
             .sum()
             .reset_index()
             .sort_values("energy_kwh", ascending=False))

    x = np.arange(len(agg))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, agg["energy_kwh"],      width, label="Total")
    ax.bar(x - 0.5*width, agg["cpu_energy_kwh"],  width, label="CPU")
    ax.bar(x + 0.5*width, agg["gpu_energy_kwh"],  width, label="GPU")
    ax.bar(x + 1.5*width, agg["ram_energy_kwh"],  width, label="RAM")

    ticks = [a if len(a) <= 22 else (a[:19] + "…") for a in agg["Action"]]
    ax.set_xticks(x); ax.set_xticklabels(ticks, rotation=30, ha="right")
    ax.set_xlabel("Tool / Action"); ax.set_ylabel("Energy (kWh)")
    ax.set_title(title or "Energy Consumption by Tool / Action (Env_view)")
    ax.legend(title="Metric"); ax.grid(axis="y", linestyle="--", alpha=0.3)

    out = outdir / "energy_by_tool.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out}")


def lines_energy_by_source(per_source_step: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for src, g in per_source_step.groupby("source"):
        g = g.sort_values("step")
        ax.plot(g["step"], g["energy_kwh"], marker="o", label=_clean_action_name(src))
    ax.set_title("Energy source per step (kWh)")
    ax.set_xlabel("Step / Action"); ax.set_ylabel("kWh")
    ax.grid(True, alpha=0.3); ax.legend(ncol=2, fontsize=9)
    out = outdir / "per_source_energy_by_step.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out}")


def line_duration(per_step: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_step["step"], per_step["duration_s"], marker="o")
    ax.set_title("Time spent per step / action (s)")
    ax.set_xlabel("Step / Action"); ax.set_ylabel("Seconds")
    ax.grid(True, alpha=0.3)
    out = outdir / "per_step_duration_line.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out}")


def line_emissions(per_step: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(per_step["step"], per_step["emissions_kg"], marker="o")
    ax.set_title("Emissions per step (kg CO₂e)")
    ax.set_xlabel("Step / Action"); ax.set_ylabel("kg CO₂e")
    ax.grid(True, alpha=0.3)
    out = outdir / "per_step_emissions_line.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out}")


# ====================== Agregation multi-experiments & plots ======================

def aggregate_experiments(exp_dirs):
    """
    Agregates folders of experiments at {task, agent} level with totals.
    Returns a DataFrame with columns:
      - task
      - agent
      - emissions_kg
      - duration_s
      - cpu_energy_kwh
      - gpu_energy_kwh
      - ram_energy_kwh
      - energy_kwh
    """
    rows = []
    for p in exp_dirs:
        p = Path(p).resolve()
        try:
            per_step, _ = load_codecarbon(p)
        except Exception:
            per_step = None
        if per_step is None:
            continue

        task  = infer_task_name(p)
        agent = infer_agent_name(p)

        s = per_step[["emissions_kg", "duration_s",
                      "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh", "energy_kwh"]].sum(numeric_only=True)

        rows.append({
            "task": task,
            "agent": agent,
            "emissions_kg": float(s.get("emissions_kg", 0.0)),
            "duration_s": float(s.get("duration_s", 0.0)),
            "cpu_energy_kwh": float(s.get("cpu_energy_kwh", 0.0)),
            "gpu_energy_kwh": float(s.get("gpu_energy_kwh", 0.0)),
            "ram_energy_kwh": float(s.get("ram_energy_kwh", 0.0)),
            "energy_kwh": float(s.get("energy_kwh", 0.0)),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    
    df = df.groupby(["task", "agent"], as_index=False).sum(numeric_only=True)
    return df


def bar_emissions_by_task_agent(df: pd.DataFrame, outdir: Path):
    """
    Bar plots of the emissions per task (x), with one bar per agent within each task.
    """
    pivot = df.pivot_table(index="task", columns="agent", values="emissions_kg",
                           aggfunc="sum", fill_value=0.0).sort_index()
    ax = pivot.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Emissions per Task y Agent (kg CO₂e)")
    ax.set_xlabel("Task"); ax.set_ylabel("Emissions (kg CO₂e)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(title="Agent", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "multi_emissions_by_task_agent.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"[OK] {out}")


def stacked_hw_by_task_agent(df: pd.DataFrame, outdir: Path):
    """
    For each task and agent, draws ONE stacked bar (CPU+GPU+RAM). 
    A black dot indicates the total (kWh).
    """
    df = df.copy().sort_values(["task", "agent"])
    df["label"] = df["task"] + " | " + df["agent"]

    x = np.arange(len(df))
    cpu = df["cpu_energy_kwh"].to_numpy(float)
    gpu = df["gpu_energy_kwh"].to_numpy(float)
    ram = df["ram_energy_kwh"].to_numpy(float)
    tot = df["energy_kwh"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x, cpu, label="CPU")
    ax.bar(x, gpu, bottom=cpu, label="GPU")
    ax.bar(x, ram, bottom=cpu+gpu, label="RAM")
    ax.scatter(x, tot, s=30, c="k", marker="o", label="Total")

    ticks = [lab if len(lab) <= 28 else (lab[:25] + "…") for lab in df["label"]]
    ax.set_xticks(x); ax.set_xticklabels(ticks, rotation=30, ha="right")

    ax.set_title("Hardware Consumption per Task amd Agent (kWh)")
    ax.set_xlabel("Task | Agent"); ax.set_ylabel("Energy (kWh)")
    ax.grid(axis="y", linestyle="--", alpha=0.3); ax.legend(title="Component")

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "multi_stacked_hw_by_task_agent.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out}")


def scatter_time_vs_emissions(df: pd.DataFrame, outdir: Path):
    """
    Dispersion: X = total duration (s), Y = emissions (kg CO₂e).
    Color by agent and label with task.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    for ag in sorted(df["agent"].unique()):
        sub = df[df["agent"] == ag]
        ax.scatter(sub["duration_s"], sub["emissions_kg"], label=ag)
        for _, r in sub.iterrows():
            txt = r["task"]
            if len(txt) > 18: txt = txt[:15] + "…"
            ax.annotate(txt, (r["duration_s"], r["emissions_kg"]),
                        fontsize=8, xytext=(3, 3), textcoords="offset points", alpha=0.8)

    ax.set_title("Time vs Emissions")
    ax.set_xlabel("Total time (s)"); ax.set_ylabel("Emissions (kg CO₂e)")
    ax.grid(True, linestyle="--", alpha=0.3); ax.legend(title="Agent")

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "multi_scatter_time_vs_emissions.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[OK] {out}")


# ===================================== Main =====================================

def main():
    """
    1) Loads CSVs of CodeCarbon (per-exp).
    2) Builds step labels.
    3) Generates per-exp plots in <exp>/codecarbon_plots/.
    4) If --compare is used, aggregates multiple folders and generates multi-exp plots. 
    """
    ap = argparse.ArgumentParser(description="Plots de métricas CodeCarbon")
    ap.add_argument("--exp-dir", help="Carpeta del experimento (contiene env_log/)")
    ap.add_argument("--compare", nargs="+",
                    help="Varias carpetas de experimento para comparar por task/agent")
    ap.add_argument("--out-dir", default="codecarbon_multi_plots",
                    help="Carpeta de salida para las gráficas multi-exp")
    args = ap.parse_args()

    # --- Per-experiment ---
    if args.exp_dir:
        exp_dir = Path(args.exp_dir).resolve()
        if not (exp_dir / "env_log").exists():
            raise SystemExit(f"No se encontró env_log dentro de {exp_dir}")

        per_step, per_source_step = load_codecarbon(exp_dir)
        if per_step is None:
            raise SystemExit("No se encontraron CSVs de CodeCarbon en este experimento.")

        outdir = exp_dir / "codecarbon_plots"
        outdir.mkdir(parents=True, exist_ok=True)

        # CSVs usefull
        per_step.to_csv(outdir / "per_step.csv", index=False)
        per_source_step.to_csv(outdir / "per_source_step.csv", index=False)

        # Labels por step (Actions del env)
        step2label = build_step_labels(exp_dir, per_source_step)

        # PLots per-exp
        bars_by_step(per_step, step2label, outdir, title=f"Energy by Step: {exp_dir.name}")
        bars_by_action(per_source_step, outdir, title=f"Energy Consumption by Tool: {exp_dir.name}")
        lines_energy_by_source(per_source_step, outdir)
        line_duration(per_step, outdir)
        line_emissions(per_step, outdir)

        print(f"✅ Ready per-exp. Check {outdir}")

    # --- Multi-experiments ---
    if args.compare:
        exp_dirs = [Path(p).resolve() for p in args.compare]
        df_multi = aggregate_experiments(exp_dirs)
        if df_multi is None or df_multi.empty:
            raise SystemExit("No se pudo agregar ninguna carpeta de --compare (¿tienen env_log/ y CSVs de CodeCarbon?).")

        outdir_multi = Path(args.out_dir).resolve()
        outdir_multi.mkdir(parents=True, exist_ok=True)
        df_multi.to_csv(outdir_multi / "multi_aggregate.csv", index=False)

        bar_emissions_by_task_agent(df_multi, outdir_multi)
        stacked_hw_by_task_agent(df_multi, outdir_multi)
        scatter_time_vs_emissions(df_multi, outdir_multi)

        print(f"✅ Ready multi-exp. Check {outdir_multi}")


if __name__ == "__main__":
    main()