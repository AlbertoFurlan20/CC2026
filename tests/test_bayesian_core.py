from __future__ import annotations

import copy
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import pytest

from MLAgentBench.experiments.bayesian import (
    CommandResult,
    EvaluationContractError,
    NoValidResult,
    SubprocessBackend,
    TrialExecutionError,
    aggregate_scores,
    build_eval_argv,
    build_runner_argv,
    create_run_paths,
    execute_run,
    parse_final_evaluation,
    remaining_attempts,
    terminal_trial_count,
)


def _write_eval(
    eval_path: Path,
    trace_path: Path,
    *,
    score: Any,
    evidence: list[Any] | None = None,
    extra: dict[str, Any] | None = None,
    **record_values: Any,
) -> None:
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "final_score": score,
        "score": evidence if evidence is not None else [score],
        "extra": extra or {},
        **record_values,
    }
    eval_path.write_text(json.dumps({str(trace_path): record}), encoding="utf-8")


def _flag_value(argv: Sequence[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def test_run_paths_and_argv_preserve_paths_and_fixed_flags(
    resolve_bayesian_config: Any,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["fixed"].update(
        {
            "fast_llm_name": "fast model",
            "edit_script_llm_name": "editor model",
            "agent_type": "ResearchAgent",
            "retrieval": True,
            "actions_remove_from_prompt": ["Read File", "Python REPL"],
            "actions_add_to_prompt": ["Reflection"],
            "device": 2,
        }
    )
    payload["tracking"]["codecarbon"] = True
    config = resolve_bayesian_config(payload)
    paths = create_run_paths(config, trial_number=7, run_index=2)

    runner = build_runner_argv(
        config,
        {"top_p": 0.8, "temperature": 0.25},
        paths,
        python_command=(sys.executable,),
    )
    evaluator = build_eval_argv(config, paths, python_command=(sys.executable,))

    assert paths.run_id == "bayes_test_trial_000007_run_02"
    assert paths.log_dir.parent == config.logs_root
    assert paths.work_dir.parent == config.work_root
    assert " " in str(paths.log_dir)
    assert runner[:4] == [sys.executable, "-u", "-m", "MLAgentBench.runner"]
    assert _flag_value(runner, "--log-dir") == str(paths.log_dir)
    assert _flag_value(runner, "--work-dir") == str(paths.work_dir)
    assert _flag_value(runner, "--fast-llm-name") == "fast model"
    assert _flag_value(runner, "--edit-script-llm-name") == "editor model"
    assert _flag_value(runner, "--device") == "2"
    assert _flag_value(runner, "--top-p") == "0.8"
    assert _flag_value(runner, "--temperature") == "0.25"
    assert "--retrieval" in runner
    assert "--use-codecarbon" in runner
    assert "--eval-intermediate" not in evaluator
    assert evaluator[:4] == [sys.executable, "-u", "-m", "MLAgentBench.eval"]
    assert _flag_value(evaluator, "--output-file") == str(paths.eval_json_path)


def test_runner_argv_rejects_unknown_suggested_parameter(
    resolve_bayesian_config: Any,
) -> None:
    config = resolve_bayesian_config()
    paths = create_run_paths(config, trial_number=0, run_index=1)
    with pytest.raises(ValueError, match="unsupported suggested parameter"):
        build_runner_argv(config, {"best_of": 3}, paths)


def test_parse_exact_final_score_and_metadata(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    eval_path = log_dir / "eval.json"
    trace_path = log_dir / "env_log" / "trace.json"
    relative_trace = Path("env_log") / "trace.json"
    payload = {
        str(relative_trace): {
            "final_score": 0.0,
            "score": [0.0],
            "submitted_final_answer": True,
            "total_time": 12.5,
            "extra": {
                "oom_error": False,
                "codecarbon": {
                    "totals": {
                        "emissions_kg": 0.001,
                        "energy_kwh": 0.02,
                        "duration_s": 11.0,
                    }
                },
            },
        },
        str(tmp_path / "decoy" / "trace.json"): {
            "final_score": 999.0,
            "score": [999.0],
        },
    }
    eval_path.parent.mkdir(parents=True)
    eval_path.write_text(json.dumps(payload), encoding="utf-8")

    outcome = parse_final_evaluation(eval_path, trace_path)

    assert outcome.score == 0.0
    assert outcome.submitted_final_answer is True
    assert outcome.total_time_seconds == 12.5
    assert outcome.emissions_kg == 0.001
    assert outcome.energy_kwh == 0.02
    assert outcome.codecarbon_duration_seconds == 11.0
    assert outcome.diagnostics["oom_error"] is False


def test_minus_one_is_valid_only_with_final_evaluation_evidence(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    trace_path = tmp_path / "env_log" / "trace.json"
    _write_eval(eval_path, trace_path, score=-1, evidence=[-1])
    assert parse_final_evaluation(eval_path, trace_path).score == -1.0

    _write_eval(eval_path, trace_path, score=-1, evidence=[])
    with pytest.raises(NoValidResult, match="no valid final benchmark score"):
        parse_final_evaluation(eval_path, trace_path)


@pytest.mark.parametrize("bad_score", [True, "0.5", None, float("nan"), float("inf")])
def test_invalid_final_scores_are_contract_errors(
    tmp_path: Path, bad_score: Any
) -> None:
    eval_path = tmp_path / "eval.json"
    trace_path = tmp_path / "env_log" / "trace.json"
    _write_eval(eval_path, trace_path, score=bad_score)
    with pytest.raises(EvaluationContractError, match="numeric|finite"):
        parse_final_evaluation(eval_path, trace_path)


def test_parser_never_falls_back_to_intermediate_score(tmp_path: Path) -> None:
    eval_path = tmp_path / "eval.json"
    trace_path = tmp_path / "env_log" / "trace.json"
    eval_path.write_text(
        json.dumps(
            {
                str(trace_path): {
                    "score": [0.99],
                    "score_steps": [1],
                    "extra": {},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationContractError, match="missing final_score"):
        parse_final_evaluation(eval_path, trace_path)


def test_parser_rejects_missing_ambiguous_and_failed_evaluator_records(
    tmp_path: Path,
) -> None:
    eval_path = tmp_path / "eval.json"
    trace_path = tmp_path / "env_log" / "trace.json"
    eval_path.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(EvaluationContractError, match="found 0"):
        parse_final_evaluation(eval_path, trace_path)

    eval_path.write_text(
        json.dumps(
            {
                str(trace_path): {"final_score": 0.3},
                "env_log/trace.json": {"final_score": 0.4},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationContractError, match="found 2"):
        parse_final_evaluation(eval_path, trace_path)

    _write_eval(
        eval_path,
        trace_path,
        score=-1,
        evidence=[],
        extra={"final_evaluation_error": "dataset download failed"},
    )
    with pytest.raises(EvaluationContractError, match="dataset download failed"):
        parse_final_evaluation(eval_path, trace_path)


@pytest.mark.parametrize("content", ["{", "[]"])
def test_parser_rejects_malformed_or_non_object_json(
    tmp_path: Path, content: str
) -> None:
    eval_path = tmp_path / "eval.json"
    eval_path.write_text(content, encoding="utf-8")
    with pytest.raises(
        EvaluationContractError, match="cannot read|root must be an object"
    ):
        parse_final_evaluation(eval_path, tmp_path / "trace.json")


class ArtifactBackend:
    """In-memory process backend that writes the evaluator's real JSON contract."""

    def __init__(
        self,
        *,
        final_score: float = 0.75,
        evidence: list[float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.final_score = final_score
        self.evidence = [final_score] if evidence is None else evidence
        self.extra = extra or {}
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        self.calls.append(
            {
                "argv": list(argv),
                "cwd": cwd,
                "stdout_path": stdout_path,
                "timeout_seconds": timeout_seconds,
            }
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("fake process output\n", encoding="utf-8")
        module = argv[argv.index("-m") + 1]
        if module == "MLAgentBench.eval":
            eval_path = Path(_flag_value(argv, "--output-file"))
            log_dir = Path(_flag_value(argv, "--log-folder"))
            _write_eval(
                eval_path,
                log_dir / "env_log" / "trace.json",
                score=self.final_score,
                evidence=self.evidence,
                extra=self.extra,
                submitted_final_answer=True,
                total_time=9.5,
            )
        return CommandResult(returncode=0, duration_seconds=0.25)


def test_execute_run_records_complete_result_and_commands(
    resolve_bayesian_config: Any,
) -> None:
    config = resolve_bayesian_config()
    backend = ArtifactBackend(
        final_score=0.8,
        extra={
            "codecarbon": {
                "totals": {
                    "emissions_kg": 0.01,
                    "energy_kwh": 0.2,
                    "duration_s": 8,
                }
            }
        },
    )

    record = execute_run(
        config,
        trial_number=2,
        run_index=1,
        params={"top_p": 0.8, "temperature": 0.3},
        process_backend=backend,
    )

    assert record["status"] == "complete"
    assert record["score"] == 0.8
    assert record["total_time_seconds"] == 9.5
    assert record["runner_returncode"] == 0
    assert record["eval_returncode"] == 0
    assert record["emissions_kg"] == 0.01
    assert len(backend.calls) == 2
    assert all(call["cwd"] == config.repo_root for call in backend.calls)
    assert backend.calls[0]["stdout_path"].name == "stdout.txt"
    assert backend.calls[1]["stdout_path"].name == "eval.log"


def test_execute_run_applies_behavioral_penalty(
    resolve_bayesian_config: Any,
) -> None:
    config = resolve_bayesian_config()
    backend = ArtifactBackend(final_score=-1, evidence=[])

    record = execute_run(
        config,
        trial_number=0,
        run_index=1,
        params={"top_p": 0.8, "temperature": 0.3},
        process_backend=backend,
    )

    assert record["status"] == "behavioral_penalty"
    assert record["score"] == 0.0
    assert record["failure_category"] == "behavioral_failure"


class ResultSequenceBackend:
    def __init__(self, results: Sequence[CommandResult]) -> None:
        self.results = list(results)
        self.calls = 0

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("intentional fake failure", encoding="utf-8")
        result = self.results[self.calls]
        self.calls += 1
        return result


@pytest.mark.parametrize(
    ("results", "category"),
    [
        ([CommandResult(None, 0.1, spawn_error="missing")], "runner_spawn_error"),
        ([CommandResult(-15, 0.1, timed_out=True)], "runner_timeout"),
        ([CommandResult(3, 0.1)], "runner_failure"),
        (
            [CommandResult(0, 0.1), CommandResult(None, 0.1, spawn_error="missing")],
            "evaluator_spawn_error",
        ),
        (
            [CommandResult(0, 0.1), CommandResult(-15, 0.1, timed_out=True)],
            "evaluator_timeout",
        ),
        ([CommandResult(0, 0.1), CommandResult(2, 0.1)], "evaluator_failure"),
    ],
)
def test_execute_run_classifies_process_failures(
    resolve_bayesian_config: Any,
    results: Sequence[CommandResult],
    category: str,
) -> None:
    config = resolve_bayesian_config()
    backend = ResultSequenceBackend(results)

    with pytest.raises(TrialExecutionError) as caught:
        execute_run(
            config,
            trial_number=0,
            run_index=1,
            params={"top_p": 0.8, "temperature": 0.3},
            process_backend=backend,
        )

    assert caught.value.category == category
    assert caught.value.run_record["status"] == category
    assert caught.value.run_record["finished_at"] is not None


def test_execute_run_refuses_artifact_collision(
    resolve_bayesian_config: Any,
) -> None:
    config = resolve_bayesian_config()
    paths = create_run_paths(config, trial_number=0, run_index=1)
    paths.log_dir.mkdir(parents=True)
    backend = ResultSequenceBackend([])

    with pytest.raises(TrialExecutionError) as caught:
        execute_run(
            config,
            trial_number=0,
            run_index=1,
            params={"top_p": 0.8, "temperature": 0.3},
            process_backend=backend,
        )

    assert caught.value.category == "artifact_collision"
    assert backend.calls == 0


def test_aggregate_and_resume_budget_helpers() -> None:
    assert aggregate_scores([0.1, 0.3], "mean") == pytest.approx(0.2)
    assert aggregate_scores([9, 1, 3], "median") == 3.0
    with pytest.raises(ValueError, match="empty"):
        aggregate_scores([], "mean")
    with pytest.raises(ValueError, match="unknown"):
        aggregate_scores([1], "max")

    class State:
        def __init__(self, name: str) -> None:
            self.name = name

    class Trial:
        def __init__(self, state: str) -> None:
            self.state = State(state)

    class Study:
        trials = [
            Trial("COMPLETE"),
            Trial("FAIL"),
            Trial("PRUNED"),
            Trial("WAITING"),
            Trial("RUNNING"),
        ]

    assert terminal_trial_count(Study()) == 3
    assert remaining_attempts(Study(), 5) == 2
    assert remaining_attempts(Study(), 2) == 0


def test_real_subprocess_backend_captures_output_and_timeout(tmp_path: Path) -> None:
    backend = SubprocessBackend()
    output = tmp_path / "folder with spaces" / "stdout.txt"
    success = backend.run(
        [sys.executable, "-c", "print('hello from child')"],
        cwd=tmp_path,
        stdout_path=output,
        timeout_seconds=5,
    )
    assert success.returncode == 0
    assert success.timed_out is False
    assert output.read_text(encoding="utf-8").strip() == "hello from child"

    timeout_output = tmp_path / "timeout.txt"
    timed_out = backend.run(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        stdout_path=timeout_output,
        timeout_seconds=0.05,
    )
    assert timed_out.timed_out is True
    assert "command timed out" in timeout_output.read_text(encoding="utf-8")
    assert math.isfinite(timed_out.duration_seconds)


class InterruptingBackend:
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            "interrupted while runner was active\n", encoding="utf-8"
        )
        raise KeyboardInterrupt("simulated operator interrupt")


def test_execute_run_publishes_detached_interrupted_record(
    resolve_bayesian_config: Any,
) -> None:
    config = resolve_bayesian_config()
    snapshots: list[dict[str, Any]] = []

    with pytest.raises(KeyboardInterrupt, match="simulated operator interrupt"):
        execute_run(
            config,
            trial_number=8,
            run_index=1,
            params={"top_p": 0.8, "temperature": 0.3},
            process_backend=InterruptingBackend(),
            record_updated=snapshots.append,
        )

    assert [snapshot["status"] for snapshot in snapshots] == [
        "starting",
        "runner_running",
        "interrupted",
    ]
    final = snapshots[-1]
    assert final["failure_category"] == "interrupted"
    assert final["failure_reason"] == "KeyboardInterrupt: simulated operator interrupt"
    assert final["runner_argv"] is not None
    assert final["finished_at"] is not None
    # The callback contract promises detached snapshots rather than aliases to
    # a record that will mutate later.
    assert snapshots[0]["runner_argv"] is None
    assert snapshots[0]["finished_at"] is None


def _pid_is_running(pid: int) -> bool:
    """Return false for missing or zombie processes on POSIX test hosts."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        import subprocess

        state = subprocess.check_output(
            ["ps", "-o", "stat=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return bool(state) and not state.startswith("Z")
    except (OSError, subprocess.CalledProcessError):
        return False


def _wait_not_running(pid: int, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_is_running(pid):
            return True
        time.sleep(0.02)
    return not _pid_is_running(pid)


@pytest.mark.skipif(os.name != "posix", reason="process-group semantics are POSIX-only")
@pytest.mark.parametrize(
    ("mode", "timeout_seconds", "expected_returncode", "expect_timeout"),
    [
        ("timeout", 0.3, None, True),
        ("nonzero", 5.0, 23, False),
    ],
)
def test_subprocess_backend_kills_sigterm_ignoring_descendant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    timeout_seconds: float,
    expected_returncode: int | None,
    expect_timeout: bool,
) -> None:
    pid_file = tmp_path / f"descendant-{mode}.pid"
    child_code = (
        "import os,signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "open(sys.argv[1], 'w').write(str(os.getpid()));"
        "time.sleep(60)"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time;"
        f"subprocess.Popen([sys.executable,'-c',{child_code!r},sys.argv[1]]);"
        "p=pathlib.Path(sys.argv[1]);"
        "deadline=time.monotonic()+3;"
        "\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "raise SystemExit(23) if sys.argv[2]=='nonzero' else time.sleep(60)"
    )
    backend = SubprocessBackend()
    monkeypatch.setattr(SubprocessBackend, "termination_grace_seconds", 0.2)
    child_pid: int | None = None
    try:
        result = backend.run(
            [sys.executable, "-c", parent_code, str(pid_file), mode],
            cwd=tmp_path,
            stdout_path=tmp_path / f"{mode}.log",
            timeout_seconds=timeout_seconds,
        )
        assert pid_file.exists(), (tmp_path / f"{mode}.log").read_text(encoding="utf-8")
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        assert result.timed_out is expect_timeout
        if expected_returncode is not None:
            assert result.returncode == expected_returncode
        assert _wait_not_running(child_pid), (
            f"descendant {child_pid} survived backend cleanup in {mode} mode"
        )
    finally:
        if child_pid is not None and _pid_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)
