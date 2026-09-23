"""
This file is the entry point for MLAgentBench.
"""

import argparse
import json
import os
import sys
from MLAgentBench import LLM
from MLAgentBench.environment import Environment
from MLAgentBench.agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent
from MLAgentBench.agents.agent_research import ResearchAgent
# from MLAgentBench.agents.agent_langchain  import LangChainAgent
try:
    from MLAgentBench.agents.agent_autogpt  import AutoGPTAgent
except:
    print("Failed to import AutoGPTAgent; Make sure you have installed the autogpt dependencies if you want to use it.")
try:
    from MLAgentBench.agents.agent_langchain import LangChainAgent
except Exception:
    LangChainAgent = None

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
    return "medium", config["medium"]  # default to medium

def select_llm_for_difficulty(level, info, args):
    """Select LLM name based on difficulty and user override."""
    if args.llm_name:  # user override
        return args.llm_name
    return info["llm"]

def run(agent_cls, args):

    if args.agent_type == "LangChainAgent" and LangChainAgent is None:
        raise RuntimeError(
            "LangChainAgent no está disponible porque las dependencias de langchain/langchain_core fallan. "
            "Usa ResearchAgent o instala una combinación compatible de LangChain."
        )

    # Load task difficulty and auto-select LLM if not user-specified
    config = load_task_difficulty()
    difficulty_level, difficulty_info = get_task_difficulty(args.task, config)

    if not args.llm_name:
        args.llm_name = select_llm_for_difficulty(difficulty_level, difficulty_info, args)
        print(f"[vLLM Router] Task '{args.task}' classified as '{difficulty_level}' → LLM: {args.llm_name} (GPU: {difficulty_info['gpu']})", file=sys.stderr)
    else:
        print(f"[vLLM Router] User override: LLM={args.llm_name} (task '{args.task}' classified as '{difficulty_level}')", file=sys.stderr)

    # Set fast LLM for summarization (use lighter model)
    if not args.fast_llm_name:
        args.fast_llm_name = config["easy"]["llm"]
        print(f"[vLLM Router] Fast LLM (summarization): {args.fast_llm_name}", file=sys.stderr)

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

        agent = agent_cls(args, env)
        final_message = agent.run(env)
        print("=====================================")
        print("Final message: ", final_message)

    env.save("final")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--log-dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="./workspace", help="work dir")
    parser.add_argument("--max-steps", type=int, default=50, help="number of steps")
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # general agent configs
    parser.add_argument("--agent-type", type=str, default="ResearchAgent", help="agent type")
    parser.add_argument("--llm-name", type=str, default="claude-v1", help="llm name")
    parser.add_argument("--fast-llm-name", type=str, default="claude-v1", help="llm name")
    parser.add_argument("--edit-script-llm-name", type=str, default="claude-v1", help="llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=1024, help="llm max tokens") #default=4000
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations for agent")

    # research agent configs
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove in addition to the default ones: Read File, Write File, Append File, Retrieval from Research Log, Append Summary to Research Log, Python REPL, Edit Script Segment (AI)")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--retrieval", action="store_true", help="enable retrieval")
    parser.add_argument("--valid-format-entires", type=str, nargs='+', default=None, help="valid format entries")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context")
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    # langchain configs
    parser.add_argument("--langchain-agent", type=str, default="zero-shot-react-description", help="langchain agent")
    
    # Codecarbon configs
    # Codecarbon is an optional dependency, so we use a flag to enable it in case the user has it installed. 
    # If the user enables it but it's not installed, we'll print a warning and continue without it.
    parser.add_argument("--use-codecarbon", action="store_true",help="Active measure with CodeCarbon if its installed(default is disabled).")



    args = parser.parse_args()
    print(args, file=sys.stderr)
    if not args.retrieval or args.agent_type != "ResearchAgent":
        # should not use these actions when there is no retrieval
        args.actions_remove_from_prompt.extend(["Retrieval from Research Log", "Append Summary to Research Log", "Reflection"])
    LLM.FAST_MODEL = args.fast_llm_name
    run(getattr(sys.modules[__name__], args.agent_type), args)
    