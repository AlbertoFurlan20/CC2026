"""
This file is the entry point for MLAgentBench.
"""

import argparse
import asyncio
import json
import os
import sys
from MLAgentBench import LLM
from MLAgentBench.environment import Environment
from MLAgentBench.agents.orchestrator_agent import OrchestratorAgent
from MLAgentBench.multi_agent.whiteboard import Whiteboard

TASK_DIFFICULTY_FILE = os.path.join(os.path.dirname(__file__), "task_difficulty.json")

def load_task_difficulty():
    """Load task difficulty classification config."""
    with open(TASK_DIFFICULTY_FILE, "r") as f:
        return json.load(f)

def get_task_difficulty(task_name, config=None):
    """Get difficulty level for a task."""
    if config is None:
        config = load_task_difficulty()
    task_name_lower = task_name.lower()
    for level, info in config.items():
        if task_name_lower in [t.lower() for t in info["tasks"]]:
            return level, info
    return "medium", config["medium"]

def select_llm_for_difficulty(level, info, args):
    """Select LLM name based on difficulty and user override."""
    if args.llm_name:
        return args.llm_name
    return info["llm"]

def run(args):
    config = load_task_difficulty()
    difficulty_level, difficulty_info = get_task_difficulty(args.task, config)

    if not args.llm_name:
        args.llm_name = select_llm_for_difficulty(difficulty_level, difficulty_info, args)
        print(f"[vLLM Router] Task '{args.task}' classified as '{difficulty_level}' → LLM: {args.llm_name} (GPU: {difficulty_info['gpu']})", file=sys.stderr)
    else:
        print(f"[vLLM Router] User override: LLM={args.llm_name} (task '{args.task}' classified as '{difficulty_level}')", file=sys.stderr)

    if not args.fast_llm_name:
        args.fast_llm_name = config["easy"]["llm"]
        print(f"[vLLM Router] Fast LLM (summarization): {args.fast_llm_name}", file=sys.stderr)

    LLM.FAST_MODEL = args.fast_llm_name

    with Environment(args) as env:

        print("=====================================")
        research_problem, benchmark_folder_name = env.get_task_description()
        print("Benchmark folder name: ", benchmark_folder_name)
        print("Research problem: ", research_problem)
        print("Lower level actions enabled: ", [action.name for action in env.low_level_actions])
        print("High level actions enabled: ", [action.name for action in env.high_level_actions])
        print("Read only files: ", env.read_only_files, file=sys.stderr)
        print(f"Difficulty level: {difficulty_level} (GPU: {difficulty_info['gpu']})")
        print(f"Primary LLM: {args.llm_name}, Fast LLM: {args.fast_llm_name}")
        print("=====================================")

        print(f"[Orchestrator] Spawning {args.num_workers} parallel workers.")
        whiteboard = Whiteboard()
        orchestrator = OrchestratorAgent(args=args, whiteboard=whiteboard)
        best_workspace = asyncio.run(orchestrator.run(env))
        print(f"[Orchestrator] Best workspace: {best_workspace}")

    env.save("final")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--log-dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="./workspace", help="work dir")
    parser.add_argument("--max-steps", type=int, default=50, help="number of steps")
    parser.add_argument("--max-time", type=int, default=5 * 60 * 60, help="max time")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    parser.add_argument("--llm-name", type=str, default="claude-v1", help="llm name")
    parser.add_argument("--fast-llm-name", type=str, default="claude-v1", help="fast llm name")
    parser.add_argument("--edit-script-llm-name", type=str, default="claude-v1", help="edit script llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=1024, help="edit script llm max tokens")
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations per agent")

    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove from prompt")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add to prompt")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context")
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    parser.add_argument("--num-workers", type=int, default=2, help="number of parallel worker agents")
    parser.add_argument("--heavy-llm-name", type=str, default="llama-3.1-8b-instruct", help="model for supervisor upgrade")

    args = parser.parse_args()
    print(args, file=sys.stderr)
    run(args)
