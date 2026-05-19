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

app = FastAPI(title="MLAgentBench Runner API")


class RunRequest(BaseModel):
    """Parameters to launch an MLAgentBench task through the API."""
    task: str = Field(..., description="Name of the task (cifar10, imdb, house-price, feedback, etc.)")
    log_dir: str = Field(..., description="Logs folder (e.g. logs/1users_cifar10/user_1)")
    work_dir: str = Field("workspace", description="Workspace directory")

    llm_name: str = Field("llama-3.1-8B-Instruct", description="--llm-name")
    fast_llm_name: str = Field("llama-3.1-8B-Instruct", description="--fast-llm-name")
    edit_script_llm_name: str = Field("llama-3.1-8B-Instruct", description="--edit-script-llm-name")
    heavy_llm_name: str = Field("llama-3.1-8B-Instruct", description="--heavy-llm-name")

    device: int = Field(0, description="Device ID: 0=GPU0, -1=CPU")
    max_steps: int = Field(50, description="--max-steps")
    num_workers: int = Field(2, description="Number of parallel worker agents")

    max_steps_in_context: Optional[int] = Field(default=None, description="--max-steps-in-context")
    max_observation_steps_in_context: Optional[int] = Field(default=None, description="--max-observation-steps-in-context")
    max_retries: Optional[int] = Field(default=None, description="--max-retries")

    use_codecarbon: bool = Field(default=False, description="Enable per-agent and system-wide CodeCarbon tracking")
    python_path: Optional[str] = Field(default=None, description="Python binary path; defaults to sys.executable")


@app.post("/run")
def run_task(req: RunRequest):
    """
    Equivalent to running:

      python -u -m MLAgentBench.runner \\
        --task <task> --device <device> --log-dir <log_dir> --work-dir <work_dir> \\
        --llm-name <llm_name> --fast-llm-name <fast_llm_name> \\
        --edit-script-llm-name <edit_script_llm_name> \\
        --heavy-llm-name <heavy_llm_name> --num-workers <num_workers> \\
        --max-steps <max_steps> [optional flags...]
        > <log_dir>/log 2>&1
    """

    repo_root = Path(__file__).resolve().parents[1]

    log_dir_path = repo_root / req.log_dir
    work_dir_path = repo_root / req.work_dir

    for p in (log_dir_path, work_dir_path):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    python_bin = req.python_path or sys.executable

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
        "--heavy-llm-name", req.heavy_llm_name,
        "--num-workers", str(req.num_workers),
        "--max-steps", str(req.max_steps),
    ]

    if req.use_codecarbon:
        cmd.append("--use-codecarbon")

    if req.max_steps_in_context is not None:
        cmd += ["--max-steps-in-context", str(req.max_steps_in_context)]

    if req.max_observation_steps_in_context is not None:
        cmd += ["--max-observation-steps-in-context", str(req.max_observation_steps_in_context)]

    if req.max_retries is not None:
        cmd += ["--max-retries", str(req.max_retries)]

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

    log_file_path = log_dir_path / "log"
    try:
        with open(log_file_path, "w") as f:
            f.write(completed.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(completed.stderr)
    except Exception as e:
        print(f"WARNING: couldn't write on {log_file_path}: {e}", file=sys.stderr)

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

    return {
        "status": "ok",
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "log_dir": str(log_dir_path),
        "work_dir": str(work_dir_path),
        "log_file": str(log_file_path),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
