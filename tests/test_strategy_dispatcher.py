from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DISPATCHER = REPO_ROOT / "scripts" / "compare_strategies.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _environment(
    tmp_path: Path, *, filesystem_type: str | None = None
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    df_script = "#!/bin/sh\n"
    if filesystem_type is not None:
        df_script += (
            "printf 'Filesystem Type 1K-blocks Used Available Use%% Mounted on\\n'\n"
            f"printf '/dev/test {filesystem_type} 1 1 1 1%% /tmp\\n'\n"
        )
    _write_executable(
        bin_dir / "df",
        df_script,
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}{os.pathsep}{environment['PATH']}"
    environment["STORAGE_DIR"] = str(tmp_path / "storage")
    return environment


def test_bayesian_method_execs_python_controller(tmp_path: Path) -> None:
    config_path = tmp_path / "bayes config.json"
    config_path.write_text(
        json.dumps({"search": {"method": "Bayesian"}}), encoding="utf-8"
    )
    capture_path = tmp_path / "dispatch.json"
    environment = _environment(tmp_path)
    python_shim = tmp_path / "bin" / "python-shim"
    _write_executable(
        python_shim,
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] == '-':\n"
        "    result = subprocess.run([sys.executable, *sys.argv[1:]], "
        "input=sys.stdin.buffer.read())\n"
        "    raise SystemExit(result.returncode)\n"
        "with open(os.environ['DISPATCH_CAPTURE'], 'w', encoding='utf-8') as f:\n"
        "    json.dump(sys.argv[1:], f)\n",
    )
    environment["PYTHON"] = str(python_shim)
    environment["DISPATCH_CAPTURE"] = str(capture_path)

    result = subprocess.run(
        ["bash", str(DISPATCHER), str(config_path)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(capture_path.read_text(encoding="utf-8")) == [
        str(REPO_ROOT / "scripts" / "run_bayesian.py"),
        str(config_path),
    ]


def test_unknown_method_is_rejected_before_execution(tmp_path: Path) -> None:
    config_path = tmp_path / "unknown.json"
    config_path.write_text(
        json.dumps({"search": {"method": "future-search"}}), encoding="utf-8"
    )
    environment = _environment(tmp_path)
    environment["PYTHON"] = sys.executable

    result = subprocess.run(
        ["bash", str(DISPATCHER), str(config_path)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unsupported search.method 'future-search'" in result.stderr


def test_overlay_storage_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "bayes.json"
    config_path.write_text(
        json.dumps({"search": {"method": "bayesian"}}), encoding="utf-8"
    )
    environment = _environment(tmp_path, filesystem_type="overlay")
    environment["PYTHON"] = sys.executable

    result = subprocess.run(
        ["bash", str(DISPATCHER), str(config_path)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "is on docker overlay FS" in result.stderr


def test_config_without_method_keeps_grid_branch(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy-grid.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment": {"name": "grid_test", "output_dir": "results"},
                "fixed": {
                    "task": "cifar10",
                    "llm_name": "test-model",
                    "max_steps": 1,
                },
                "grid": {"top_p": [0.9], "temperature": [0.2]},
                "execution": {"n_runs": 0},
            }
        ),
        encoding="utf-8",
    )
    environment = _environment(tmp_path)
    environment["PYTHON"] = sys.executable

    result = subprocess.run(
        ["bash", str(DISPATCHER), str(config_path)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "=== Grid: 1 configs × 0 runs = 0 experiments ===" in result.stdout
    summary = tmp_path / "storage" / "results" / "grid_test" / "summary.csv"
    assert summary.is_file()
    assert summary.read_text(encoding="utf-8").startswith(
        "config_idx,run,top_p,temperature"
    )
