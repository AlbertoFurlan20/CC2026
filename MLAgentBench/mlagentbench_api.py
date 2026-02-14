# MLAgentBench/mlagentbench_api.py

import os
import sys
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MLAgentBench Runner API (simple)")


class RunRequest(BaseModel):
    """
    For parameters to launch an MLAgentBench task through the API.
    """
    # Mandatory parameters
    task: str = Field(..., description="Name of the task (cifar10, imdb, house-price, feedback, etc.)")
    log_dir: str = Field(..., description="Logs Folder (gpt4o_cc, 1users_cifar10/user_1)")
    work_dir: str = Field("workspace", description="Workspace (normally 'workspace')")

    # Models
    llm_name: str = Field("llama-3.1-8B-Instruct", description="--llm-name") # Change models here
    fast_llm_name: str = Field("llama-3.1-8B-Instruct", description="--fast-llm-name") # "llama-3.1-8B-Instruct" "qwen2.5-7B-Instruct"
    edit_script_llm_name: str = Field("llama-3.1-8B-Instruct", description="--edit-script-llm-name") # "llama-3.1-8B-Instruct"

    # Device
    device: int = Field(0, description="Device ID: 0=GPU0, -1=CPU")

    # Type of agent and steps
    agent_type: str = Field("ResearchAgent", description="--agent-type en runner.py")
    max_steps: int = Field(50, description="--max-steps")

    # Optional flags to harden the ResearchAgent format
    valid_format_entires: Optional[List[str]] = Field(
        default=None,
        description="Filed list of fields that each LLM response must contain (e.g. ['Thought','Action','Action Input'])"
    )
    max_steps_in_context: Optional[int] = Field(
        default=None,
        description="--max-steps-in-context"
    )
    max_observation_steps_in_context: Optional[int] = Field(
        default=None,
        description="--max-observation-steps-in-context"
    )
    max_retries: Optional[int] = Field(
        default=None,
        description="--max-retries"
    )

    use_codecarbon: bool = Field(
        default=False,
        description="If true adds the flag --use-codecarbon to the runner, to enable codecarbon (if installed)."
    )
    # Opcional: cambiar binario de python
    python_path: Optional[str] = Field(
        default=None,
        description="Python route for --python; if its None, sys.executable is used."
    )


@app.post("/run")
def run_task(req: RunRequest):
    """
    Equivalent to running the following command in bash:

      rm -rf <log_dir> <work_dir> && mkdir -p <log_dir> <work_dir>
      python -u -m MLAgentBench.runner \
        --python <python> \
        --task <task> \
        --device <device> \
        --log-dir <log_dir> \
        --work-dir <work_dir> \
        --llm-name <llm_name> \
        --fast-llm-name <fast_llm_name> \
        --edit-script-llm-name <edit_script_llm_name> \
        --agent-type <agent_type> \
        --max-steps <max_steps> \
        [flags optianls...]
        > <log_dir>/log 2>&1
    """

    # Directory that contains MLAgentBench (the repo root)
    repo_root = Path(__file__).resolve().parents[1]

    # Paths absoluts of log_dir and work_dir
    log_dir_path = repo_root / req.log_dir
    work_dir_path = repo_root / req.work_dir

    # 1) rm -rf log_dir work_dir && mkdir -p ...
    for p in (log_dir_path, work_dir_path):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # 2) Python bin (Equivalent to "$(which python)")
    python_bin = req.python_path or sys.executable

    # 3) Built the command base of the runner
    cmd = [
        python_bin, "-u", "-m", "MLAgentBench.runner",
        "--python", python_bin,
        "--task", req.task,
        "--device", str(req.device),
        "--log-dir", req.log_dir,
        "--work-dir", req.work_dir,
        "--llm-name", req.llm_name,
        "--fast-llm-name", req.fast_llm_name,
        "--edit-script-llm-name", req.edit_script_llm_name,
        "--agent-type", req.agent_type,
        "--max-steps", str(req.max_steps),
        
    ]

    # 3.b) Optiona Flags (In case if the user take its for ResearchAgent hardening)
    if req.valid_format_entires:
        cmd += ["--valid-format-entires", *req.valid_format_entires]

    if req.max_steps_in_context is not None:
        cmd += ["--max-steps-in-context", str(req.max_steps_in_context)]

    if req.max_observation_steps_in_context is not None:
        cmd += ["--max-observation-steps-in-context", str(req.max_observation_steps_in_context)]

    if req.max_retries is not None:
        cmd += ["--max-retries", str(req.max_retries)]
    
    # if the user wants to enable CodeCarbon measurement, we add the flag 
    # (the runner will check if CodeCarbon is installed and act accordingly)
    if req.use_codecarbon:  
        cmd.append("--use-codecarbon")

    # 4) Execute runner
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error launching MLAgentBench.runner",
                "error": str(e),
                "cmd": " ".join(shlex.quote(c) for c in cmd),
            },
        )

    # 5) Saves stdout+stderr in <log_dir>/log
    log_file_path = log_dir_path / "log"
    try:
        with open(log_file_path, "w") as f:
            f.write(completed.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(completed.stderr)
    except Exception as e:
        print(f"WARNING: couldn't write on {log_file_path}: {e}", file=sys.stderr)

    # 6) If runner fails, return 500 with info
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "MLAgentBench.runner finished with non-zero return code",
                "returncode": completed.returncode,
                "cmd": " ".join(shlex.quote(c) for c in cmd),
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
                "log_file": str(log_file_path),
            },
        )

    # 7) Return OK
    return {
        "status": "ok",
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "log_dir": str(log_dir_path),
        "work_dir": str(work_dir_path),
        "log_file": str(log_file_path),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
