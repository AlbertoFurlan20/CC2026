# MLAgentBench/plot_time_vs_score.py

import os
import json
import argparse
import matplotlib.pyplot as plt


def load_run(run_dir: str, eval_file: str = "resultados_llm.json"):
    """
    Loads total_time and final_score from the evaluation JSON
    located in run_dir/eval_file.
    """
    path = os.path.join(run_dir, eval_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File eval missing: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Format not expected {path}")

    first_key = next(iter(data.keys()))
    run_info = data[first_key]

    total_time = float(run_info.get("total_time", 0.0))
    final_score = float(run_info.get("final_score", -1.0))

    return total_time, final_score


# ---------------------- Helpers for model ---------------------- #

def normalize_agent_name(name: str) -> str:
    """
    Normalizes model names to nice aliases:
      - 'llama-3.1-8b-instruct' -> 'Llama 3.1 8B'
      - 'qwen2.5-7b-instruct'   -> 'Qwen2.5 7B'
      - 'qwen2.5-14b-instruct'  -> 'Qwen2.5 14B'
      - others -> as is
    """
    if not isinstance(name, str):
        return "Unknown"

    raw = name.strip()
    low = raw.lower()

    # Llama 3.1 8B 
    if "llama" in low and "3.1" in low and "8b" in low:
        return "Llama 3.1 8B"

    # Qwen 2.5 7B
    if "qwen" in low and "2.5" in low and "7b" in low:
        return "Qwen2.5 7B"

    # Qwen 2.5 14B
    if "qwen" in low and "2.5" in low and "14b" in low:
        return "Qwen2.5 14B"

    # Otros qwen
    if "qwen" in low:
        return "Qwen (other)"

    # GPT in the case or using it again
    if "gpt" in low and "4o" in low:
        return "GPT-4o"
    if "gpt" in low and "4" in low and "mini" in low:
        return "GPT-4.1 mini"

    return raw or "Unknown"


def infer_agent_name_from_run(run_dir: str) -> str:
    """
    tries to infer the model/LLM used in a run:
      1) reads env_log/trace.json (llm_name, fast_llm_name, agent, agent_name)
      2) if no info, guesses from folder name
    """
    trace_path = os.path.join(run_dir, "env_log", "trace.json")
    raw_agent = None

    if os.path.exists(trace_path):
        try:
            with open(trace_path, "r") as f:
                tr = json.load(f)

            
            for k in ["llm_name", "fast_llm_name", "agent", "agent_name"]:
                v = tr.get(k)
                if isinstance(v, str) and v.strip():
                    raw_agent = v.strip()
                    break

            
            if raw_agent is None:
                meta = tr.get("meta", {})
                if isinstance(meta, dict):
                    for k in ["llm_name", "fast_llm_name", "agent", "agent_name"]:
                        v = meta.get(k)
                        if isinstance(v, str) and v.strip():
                            raw_agent = v.strip()
                            break
        except Exception:
            pass

    # Fallback: name per folder
    if raw_agent is None:
        low = run_dir.lower()
        if "llama" in low:
            raw_agent = "llama-3.1-8b"
        elif "qwen" in low:
            raw_agent = "qwen2.5-7b"
        else:
            raw_agent = "Unknown"

    return normalize_agent_name(raw_agent)


# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Plot of total time vs score (p. ej. test accuracy) for lots of runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Logs folder (The ones that went through as --log-folder al eval).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each run (same order as --runs).",
    )
    parser.add_argument(
        "--eval-file",
        default="resultados_llm.json",
        help="JSON name of the evaluation inside each run (Default resultados_llm.json).",
    )
    parser.add_argument(
        "--metric-name",
        default="Test accuracy",
        help="Metrics Name for Y axis (ej. 'CIFAR10 test accuracy', 'Score', etc.).",
    )
    parser.add_argument(
        "--out",
        default="time_vs_score.png",
        help="File name PNG.",
    )
    parser.add_argument(
        "--avg-lines",              # <<< NEW FLAG >>>
        action="store_true",
        help=(
            "If set, draw a horizontal line per model/agent at its mean score, "
            "from min(time) to max(time) of that model."
        ),
    )

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError("If uses --labels, should have the same amount of elements of --runs.")

    if args.labels:
        labels = args.labels
    else:
        
        labels = []
        for run_dir in args.runs:
            base = os.path.basename(os.path.normpath(run_dir))  
            if base.lower().startswith("user_") and len(base) > 5:
                
                short = "user" + base[5:]
            else:
                short = base
            labels.append(short)


    times = []
    scores = []
    agents = []  # infered model (Llama 3.1 8B, Qwen2.5 7B, etc.)

    for run_dir in args.runs:
        t, s = load_run(run_dir, eval_file=args.eval_file)
        times.append(t)
        scores.append(s)
        agents.append(infer_agent_name_from_run(run_dir))

    # ------------------ Scatter with color/marker per model ------------------ #
    plt.figure(figsize=(8, 5))

    unique_agents = sorted(set(agents))
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "*"]  # por si hay muchos

    color_map = {}
    marker_map = {}
    for i, ag in enumerate(unique_agents):
        color_map[ag] = base_colors[i % len(base_colors)]
        marker_map[ag] = markers[i % len(markers)]

    used_legend_agents = set()

    # Individual Points
    for x, y, label, ag in zip(times, scores, labels, agents):
        show_label_in_legend = ag not in used_legend_agents
        if show_label_in_legend:
            used_legend_agents.add(ag)

        plt.scatter(
            x,
            y,
            marker=marker_map[ag],
            color=color_map[ag],
            label=ag if show_label_in_legend else None,
            alpha=0.9,
        )

        plt.annotate(label, (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)

    # ------------------ Avg lines per model (optinal) ------------------ #
    if args.avg_lines:
        
        used_avg_labels = set()

        for ag in unique_agents:
            
            ag_times = [t for t, a in zip(times, agents) if a == ag]
            ag_scores = [s for s, a in zip(scores, agents) if a == ag]
            if not ag_times:
                continue

            mean_score = sum(ag_scores) / len(ag_scores)
            t_min = min(ag_times)
            t_max = max(ag_times)

            avg_label = f"{ag} (avg)"
            label_for_legend = avg_label if avg_label not in used_avg_labels else None
            if label_for_legend:
                used_avg_labels.add(avg_label)

            
            plt.hlines(
                y=mean_score,
                xmin=t_min,
                xmax=t_max,
                colors=color_map[ag],
                linestyles="dashed",
                linewidth=2,
                label=label_for_legend,
            )

    plt.xlabel("Total time (s)")
    plt.ylabel(args.metric_name)
    plt.title(f"{args.metric_name} vs time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Agent / Model")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot saved in: {args.out}")


if __name__ == "__main__":
    main()
