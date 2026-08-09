"""Persistent Optuna orchestration for MLAgentBench comparison studies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

from .config import (
    ConfigError,
    ResolvedBayesianConfig,
    RUNNER_FIXED_ARGUMENTS,
    SEARCH_PARAMETER_FLAGS,
    load_and_validate_config,
)

try:
    import optuna
except ImportError:  # pragma: no cover - exercised by the CLI error path.
    optuna = None  # type: ignore[assignment]


class EvaluationContractError(RuntimeError):
    """The evaluator process did not produce its documented artifact shape."""


class NoValidResult(RuntimeError):
    """The agent completed but did not produce a scoreable final artifact."""


class TrialExecutionError(RuntimeError):
    """A complete Optuna trial cannot continue.

    Instances of this exception are explicitly caught by Optuna when the config
    says to continue, causing the trial to be recorded as FAIL without stopping
    the rest of the study.
    """

    def __init__(self, message: str, *, category: str, run_record: dict[str, Any]):
        super().__init__(message)
        self.category = category
        self.run_record = run_record


@dataclass(frozen=True)
class CommandResult:
    returncode: int | None
    duration_seconds: float
    timed_out: bool = False
    spawn_error: str | None = None


class ProcessBackend(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        """Run a command without a shell and redirect combined output."""


class SubprocessBackend:
    """Production subprocess backend with process-group timeout cleanup."""

    termination_grace_seconds = 10.0

    @staticmethod
    def _group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover - defensive on shared hosts.
            return True
        return True

    @classmethod
    def _terminate(cls, process: subprocess.Popen[str]) -> None:
        if os.name != "posix":  # pragma: no cover - Brev/dev hosts are POSIX.
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=cls.termination_grace_seconds)
            except subprocess.TimeoutExpired:
                if process.poll() is None:
                    process.kill()
                process.wait()
            return

        # start_new_session=True makes the child PID its process-group ID. The
        # leader may already have exited while a training grandchild survives,
        # so group cleanup must not be conditional on process.poll().
        process_group_id = process.pid
        if cls._group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass

        deadline = time.monotonic() + cls.termination_grace_seconds
        while cls._group_exists(process_group_id) and time.monotonic() < deadline:
            if process.poll() is None:
                try:
                    process.wait(
                        timeout=min(0.1, max(0.0, deadline - time.monotonic()))
                    )
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(0.05)

        if cls._group_exists(process_group_id):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive.
                pass

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            with stdout_path.open("w", encoding="utf-8", buffering=1) as output:
                process = subprocess.Popen(
                    list(argv),
                    cwd=str(cwd),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=timeout_seconds)
                    # A runner can exit while a script it spawned remains alive.
                    # Always reap the isolated group before starting another
                    # trial, including after an apparently successful exit.
                    self._terminate(process)
                    return CommandResult(
                        returncode=returncode,
                        duration_seconds=time.monotonic() - started,
                    )
                except subprocess.TimeoutExpired:
                    self._terminate(process)
                    output.write(
                        f"\n[orchestrator] command timed out after {timeout_seconds:.3f}s\n"
                    )
                    return CommandResult(
                        returncode=process.returncode,
                        duration_seconds=time.monotonic() - started,
                        timed_out=True,
                    )
                except BaseException:
                    self._terminate(process)
                    raise
        except OSError as exc:
            return CommandResult(
                returncode=None,
                duration_seconds=time.monotonic() - started,
                spawn_error=f"{type(exc).__name__}: {exc}",
            )


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    log_dir: Path
    work_dir: Path
    stdout_path: Path
    eval_log_path: Path
    eval_json_path: Path
    expected_trace_path: Path


@dataclass(frozen=True)
class EvaluationOutcome:
    score: float
    submitted_final_answer: bool | None
    total_time_seconds: float | None
    emissions_kg: float | None
    energy_kwh: float | None
    codecarbon_duration_seconds: float | None
    diagnostics: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sigterm_to_interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt("received SIGTERM")


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as source:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(max(0, size - limit))
            return source.read()
    except OSError:
        return ""


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Any) -> None:
    content = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_bytes(path, content.encode("utf-8"))


def _atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


@contextmanager
def exclusive_study_lock(path: Path) -> Iterable[None]:
    """Prevent two local optimizer processes from mutating the same study."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ConfigError(f"another optimizer already holds {path}") from exc
        except ImportError:  # pragma: no cover - fallback for non-POSIX hosts.
            pass
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"pid={os.getpid()} started={_utc_now()}\n")
        lock_file.flush()
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):  # pragma: no cover
            pass
        lock_file.close()


def suggest_parameters(trial: Any, space: Mapping[str, Any]) -> dict[str, Any]:
    # The configuration fingerprint deliberately ignores JSON object ordering.
    # Keep the sampler call order canonical as well so reformatting a config
    # cannot alter seeded TPE suggestions, including across a resume.
    return {name: space[name].suggest(trial) for name in sorted(space)}


def create_run_paths(
    config: ResolvedBayesianConfig,
    *,
    trial_number: int,
    run_index: int,
) -> RunPaths:
    run_id = f"{config.experiment_name}_trial_{trial_number:06d}_run_{run_index:02d}"
    log_dir = config.logs_root / run_id
    work_dir = config.work_root / run_id
    return RunPaths(
        run_id=run_id,
        log_dir=log_dir,
        work_dir=work_dir,
        stdout_path=log_dir / "stdout.txt",
        eval_log_path=log_dir / "eval.log",
        eval_json_path=log_dir / "eval.json",
        expected_trace_path=log_dir / "env_log" / "trace.json",
    )


def _append_runner_arg(argv: list[str], name: str, value: Any) -> None:
    flag, kind = RUNNER_FIXED_ARGUMENTS[name]
    if kind == "bool_flag":
        if value:
            argv.append(flag)
    elif kind == "str_list":
        if value:
            argv.extend([flag, *[str(item) for item in value]])
    else:
        argv.extend([flag, str(value)])


def build_runner_argv(
    config: ResolvedBayesianConfig,
    params: Mapping[str, Any],
    paths: RunPaths,
    *,
    python_command: Sequence[str] = (sys.executable,),
) -> list[str]:
    if not python_command:
        raise ValueError("python_command must not be empty")
    argv = [*python_command, "-u", "-m", "MLAgentBench.runner"]
    argv.extend(["--python", str(python_command[0])])
    argv.extend(["--log-dir", str(paths.log_dir), "--work-dir", str(paths.work_dir)])
    for name in RUNNER_FIXED_ARGUMENTS:
        if name in config.fixed:
            _append_runner_arg(argv, name, config.fixed[name])
    for name in sorted(params):
        value = params[name]
        if name not in SEARCH_PARAMETER_FLAGS:
            raise ValueError(f"unsupported suggested parameter {name}")
        argv.extend([SEARCH_PARAMETER_FLAGS[name], str(value)])
    if config.use_codecarbon:
        argv.append("--use-codecarbon")
    return argv


def build_eval_argv(
    config: ResolvedBayesianConfig,
    paths: RunPaths,
    *,
    python_command: Sequence[str] = (sys.executable,),
) -> list[str]:
    return [
        *python_command,
        "-u",
        "-m",
        "MLAgentBench.eval",
        "--log-folder",
        str(paths.log_dir),
        "--task",
        str(config.fixed["task"]),
        "--output-file",
        str(paths.eval_json_path),
    ]


def _canonical_path(raw: str, *, base: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _optional_finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def parse_final_evaluation(
    eval_json_path: str | Path,
    expected_trace_path: str | Path,
) -> EvaluationOutcome:
    """Parse exactly one expected trace and never use intermediate-score fallback."""
    eval_path = Path(eval_json_path).resolve()
    expected = Path(expected_trace_path).resolve()
    try:
        with eval_path.open(encoding="utf-8") as result_file:
            payload = json.load(result_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"cannot read evaluator JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationContractError("evaluator JSON root must be an object")

    matches: list[dict[str, Any]] = []
    for raw_path, record in payload.items():
        if not isinstance(raw_path, str):
            continue
        if _canonical_path(raw_path, base=eval_path.parent) == expected:
            if not isinstance(record, dict):
                raise EvaluationContractError(
                    "expected evaluator record must be an object"
                )
            matches.append(record)
    if len(matches) != 1:
        raise EvaluationContractError(
            f"expected exactly one result for {expected}, found {len(matches)}"
        )

    record = matches[0]
    extra = record.get("extra")
    diagnostics = dict(extra) if isinstance(extra, dict) else {}
    final_evaluation_error = diagnostics.get("final_evaluation_error")
    if final_evaluation_error:
        raise EvaluationContractError(
            f"benchmark evaluator failed: {final_evaluation_error}"
        )

    if "final_score" not in record:
        raise EvaluationContractError("evaluator record is missing final_score")
    raw_score = record["final_score"]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise EvaluationContractError("final_score must be numeric")
    score = float(raw_score)
    if not math.isfinite(score):
        raise EvaluationContractError("final_score must be finite")

    # eval.py initializes final_score to -1.  A benchmark may legitimately
    # return -1, in which case that same value is also appended to score.
    if score == -1.0:
        evidence = record.get("score")
        valid_minus_one = isinstance(evidence, list) and any(
            _optional_finite(item) == -1.0 for item in evidence
        )
        if not valid_minus_one:
            raise NoValidResult("agent produced no valid final benchmark score")

    cc_totals: Mapping[str, Any] = {}
    codecarbon = diagnostics.get("codecarbon")
    if isinstance(codecarbon, dict) and isinstance(codecarbon.get("totals"), dict):
        cc_totals = codecarbon["totals"]

    submitted = record.get("submitted_final_answer")
    return EvaluationOutcome(
        score=score,
        submitted_final_answer=submitted if isinstance(submitted, bool) else None,
        total_time_seconds=_optional_finite(record.get("total_time")),
        emissions_kg=_optional_finite(cc_totals.get("emissions_kg")),
        energy_kwh=_optional_finite(cc_totals.get("energy_kwh")),
        codecarbon_duration_seconds=_optional_finite(cc_totals.get("duration_s")),
        diagnostics=diagnostics,
    )


def _base_run_record(
    paths: RunPaths,
    *,
    trial_number: int,
    run_index: int,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "trial_number": trial_number,
        "run_index": run_index,
        "run_id": paths.run_id,
        "params": dict(params),
        "status": "starting",
        "score": None,
        "failure_category": None,
        "failure_reason": None,
        "log_dir": str(paths.log_dir),
        "work_dir": str(paths.work_dir),
        "stdout_path": str(paths.stdout_path),
        "eval_log_path": str(paths.eval_log_path),
        "eval_json_path": str(paths.eval_json_path),
        "runner_argv": None,
        "eval_argv": None,
        "runner_returncode": None,
        "eval_returncode": None,
        "runner_time_seconds": None,
        "eval_time_seconds": None,
        "total_time_seconds": None,
        "submitted_final_answer": None,
        "emissions_kg": None,
        "energy_kwh": None,
        "codecarbon_duration_seconds": None,
        "started_at": _utc_now(),
        "finished_at": None,
    }


def _failure(
    record: dict[str, Any],
    message: str,
    *,
    category: str,
) -> TrialExecutionError:
    record["status"] = category
    record["failure_category"] = category
    record["failure_reason"] = message
    record["finished_at"] = _utc_now()
    return TrialExecutionError(message, category=category, run_record=record)


def execute_run(
    config: ResolvedBayesianConfig,
    *,
    trial_number: int,
    run_index: int,
    params: Mapping[str, Any],
    process_backend: ProcessBackend,
    python_command: Sequence[str] = (sys.executable,),
    record_updated: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    paths = create_run_paths(config, trial_number=trial_number, run_index=run_index)
    record = _base_run_record(
        paths,
        trial_number=trial_number,
        run_index=run_index,
        params=params,
    )

    def publish() -> None:
        if record_updated is not None:
            # Publish a detached copy because Optuna serializes user attrs and
            # later in-memory mutations must not be mistaken for persistence.
            record_updated(dict(record))

    publish()

    try:
        if paths.log_dir.exists() or paths.work_dir.exists():
            raise _failure(
                record,
                f"refusing to overwrite existing run paths for {paths.run_id}",
                category="artifact_collision",
            )
        paths.log_dir.mkdir(parents=True)
        paths.work_dir.mkdir(parents=True)

        runner_argv = build_runner_argv(
            config, params, paths, python_command=python_command
        )
        record["runner_argv"] = runner_argv
        record["status"] = "runner_running"
        publish()
        runner_result = process_backend.run(
            runner_argv,
            cwd=config.repo_root,
            stdout_path=paths.stdout_path,
            timeout_seconds=config.execution.subprocess_timeout_seconds,
        )
        record["runner_returncode"] = runner_result.returncode
        record["runner_time_seconds"] = runner_result.duration_seconds
        if runner_result.spawn_error:
            raise _failure(
                record,
                f"runner could not start: {runner_result.spawn_error}",
                category="runner_spawn_error",
            )
        if runner_result.timed_out:
            raise _failure(record, "runner timed out", category="runner_timeout")
        if runner_result.returncode != 0:
            tail = _tail(paths.stdout_path)
            message = f"runner exited with code {runner_result.returncode}"
            if tail:
                message += f"; log tail: {tail}"
            raise _failure(record, message, category="runner_failure")

        eval_argv = build_eval_argv(config, paths, python_command=python_command)
        record["eval_argv"] = eval_argv
        record["status"] = "evaluator_running"
        publish()
        eval_result = process_backend.run(
            eval_argv,
            cwd=config.repo_root,
            stdout_path=paths.eval_log_path,
            timeout_seconds=config.execution.subprocess_timeout_seconds,
        )
        record["eval_returncode"] = eval_result.returncode
        record["eval_time_seconds"] = eval_result.duration_seconds
        if eval_result.spawn_error:
            raise _failure(
                record,
                f"evaluator could not start: {eval_result.spawn_error}",
                category="evaluator_spawn_error",
            )
        if eval_result.timed_out:
            raise _failure(record, "evaluator timed out", category="evaluator_timeout")
        if eval_result.returncode != 0:
            tail = _tail(paths.eval_log_path)
            message = f"evaluator exited with code {eval_result.returncode}"
            if tail:
                message += f"; log tail: {tail}"
            raise _failure(record, message, category="evaluator_failure")

        try:
            outcome = parse_final_evaluation(
                paths.eval_json_path, paths.expected_trace_path
            )
        except NoValidResult as exc:
            if config.objective.behavioral_failure_policy == "penalty":
                record["status"] = "behavioral_penalty"
                record["failure_category"] = "behavioral_failure"
                record["failure_reason"] = str(exc)
                record["score"] = config.objective.behavioral_failure_value
                record["total_time_seconds"] = (
                    runner_result.duration_seconds + eval_result.duration_seconds
                )
                record["finished_at"] = _utc_now()
                publish()
                return record
            raise _failure(record, str(exc), category="behavioral_failure") from exc
        except EvaluationContractError as exc:
            raise _failure(
                record, str(exc), category="evaluation_contract_error"
            ) from exc

        record.update(
            {
                "status": "complete",
                "score": outcome.score,
                "total_time_seconds": outcome.total_time_seconds
                if outcome.total_time_seconds is not None
                else runner_result.duration_seconds + eval_result.duration_seconds,
                "submitted_final_answer": outcome.submitted_final_answer,
                "emissions_kg": outcome.emissions_kg,
                "energy_kwh": outcome.energy_kwh,
                "codecarbon_duration_seconds": outcome.codecarbon_duration_seconds,
                "finished_at": _utc_now(),
            }
        )
        publish()
        return record
    except TrialExecutionError:
        publish()
        raise
    except BaseException as exc:
        record["status"] = "interrupted"
        record["failure_category"] = "interrupted"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        record["finished_at"] = _utc_now()
        publish()
        raise


def aggregate_scores(scores: Sequence[float], method: str) -> float:
    if not scores:
        raise ValueError("cannot aggregate an empty score list")
    if method == "mean":
        return float(statistics.fmean(scores))
    if method == "median":
        return float(statistics.median(scores))
    raise ValueError(f"unknown aggregation method {method}")


def terminal_trial_count(study: Any) -> int:
    terminal = {"COMPLETE", "FAIL", "PRUNED"}
    return sum(
        getattr(trial.state, "name", str(trial.state)) in terminal
        for trial in study.trials
    )


def remaining_attempts(study: Any, configured_total: int) -> int:
    return max(0, configured_total - terminal_trial_count(study))


def _save_sampler(config: ResolvedBayesianConfig, sampler: Any) -> None:
    _atomic_bytes(
        config.sampler_path, pickle.dumps(sampler, protocol=pickle.HIGHEST_PROTOCOL)
    )


def _load_sampler(config: ResolvedBayesianConfig) -> Any | None:
    if not config.sampler_path.exists():
        return None
    try:
        with config.sampler_path.open("rb") as sampler_file:
            return pickle.load(sampler_file)
    except Exception as exc:
        if config.persistence.require_sampler_state_on_resume:
            raise ConfigError(f"cannot restore sampler state: {exc}") from exc
        print(
            f"WARNING: cannot restore sampler state; starting a fresh sampler: {exc}",
            file=sys.stderr,
        )
        return None


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_revision(repo_root: Path) -> str | None:
    """Identify the Git commit plus relevant tracked/untracked source edits."""
    commit = _git_commit(repo_root)
    if commit is None:
        return None
    try:
        revision_paths = [
            "MLAgentBench",
            "scripts",
            "requirements.txt",
            "requirements_main.txt",
        ]
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD", "--", *revision_paths],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        untracked_output = subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                *revision_paths,
            ],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return commit

    digest = hashlib.sha256()
    digest.update(diff)
    has_changes = bool(diff)
    for relative in sorted(filter(None, untracked_output.splitlines())):
        path = repo_root / relative
        if not path.is_file():
            continue
        has_changes = True
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        try:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            digest.update(b"<unreadable>")
        digest.update(b"\0")
    if not has_changes:
        return commit
    return f"{commit}+dirty.{digest.hexdigest()[:16]}"


def _trial_rows(
    study: Any, config: ResolvedBayesianConfig
) -> tuple[list[str], list[dict[str, Any]]]:
    parameter_names = sorted(config.space)
    parameter_columns = [f"param_{name}" for name in parameter_names]
    fields = [
        "trial_number",
        "state",
        "objective_value",
        "direction",
        *parameter_columns,
        "datetime_start",
        "datetime_complete",
        "duration_seconds",
        "run_ids",
        "failure_category",
        "failure_reason",
    ]
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        attrs = trial.user_attrs
        records = (
            attrs.get("run_records")
            if isinstance(attrs.get("run_records"), list)
            else []
        )
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "state": getattr(trial.state, "name", str(trial.state)),
            "objective_value": trial.value,
            "direction": config.objective.direction,
            "datetime_start": trial.datetime_start.isoformat()
            if trial.datetime_start
            else None,
            "datetime_complete": trial.datetime_complete.isoformat()
            if trial.datetime_complete
            else None,
            "duration_seconds": trial.duration.total_seconds()
            if trial.duration
            else None,
            "run_ids": ";".join(
                str(record.get("run_id")) for record in records if record.get("run_id")
            ),
            "failure_category": attrs.get("failure_category"),
            "failure_reason": attrs.get("failure_reason"),
        }
        for name in parameter_names:
            row[f"param_{name}"] = trial.params.get(name)
        rows.append(row)
    return fields, rows


def _run_rows(
    study: Any, config: ResolvedBayesianConfig
) -> tuple[list[str], list[dict[str, Any]]]:
    parameter_names = sorted(config.space)
    fields = [
        "trial_number",
        "run_index",
        "run_id",
        "status",
        "score",
        *[f"param_{name}" for name in parameter_names],
        "log_dir",
        "work_dir",
        "runner_returncode",
        "eval_returncode",
        "runner_time_seconds",
        "eval_time_seconds",
        "total_time_seconds",
        "submitted_final_answer",
        "emissions_kg",
        "energy_kwh",
        "codecarbon_duration_seconds",
        "failure_category",
        "failure_reason",
        "started_at",
        "finished_at",
    ]
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        records = trial.user_attrs.get("run_records")
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            row = dict(record)
            params = (
                record.get("params") if isinstance(record.get("params"), dict) else {}
            )
            for name in parameter_names:
                row[f"param_{name}"] = params.get(name)
            rows.append(row)
    rows.sort(key=lambda row: (int(row["trial_number"]), int(row["run_index"])))
    return fields, rows


def export_study_artifacts(study: Any, config: ResolvedBayesianConfig) -> None:
    """Atomically regenerate human-readable exports from Optuna's source of truth."""
    trial_fields, trial_rows = _trial_rows(study, config)
    run_fields, run_rows = _run_rows(study, config)
    _atomic_csv(config.results_dir / "trials.csv", trial_fields, trial_rows)
    _atomic_csv(config.results_dir / "runs.csv", run_fields, run_rows)

    completed = [
        trial
        for trial in study.trials
        if getattr(trial.state, "name", str(trial.state)) == "COMPLETE"
        and trial.value is not None
    ]
    if completed:
        best = study.best_trial
        records = best.user_attrs.get("run_records")
        best_payload: dict[str, Any] = {
            "status": "complete",
            "trial_number": best.number,
            "objective_value": best.value,
            "direction": config.objective.direction,
            "params": best.params,
            "fixed": config.fixed,
            "run_ids": [
                record.get("run_id")
                for record in records
                if isinstance(record, dict) and record.get("run_id")
            ]
            if isinstance(records, list)
            else [],
            "updated_at": _utc_now(),
        }
    else:
        best_payload = {
            "status": "no_completed_trials",
            "direction": config.objective.direction,
            "updated_at": _utc_now(),
        }
    _atomic_json(config.results_dir / "best.json", best_payload)


def _write_manifest(
    config: ResolvedBayesianConfig,
    *,
    status: str,
    study: Any | None = None,
    error: str | None = None,
) -> None:
    path = config.results_dir / "manifest.json"
    created_at = _utc_now()
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(previous.get("created_at"), str):
                created_at = previous["created_at"]
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    manifest = {
        "experiment": config.experiment_name,
        "status": status,
        "created_at": created_at,
        "updated_at": _utc_now(),
        "config_path": str(config.config_path),
        "config_fingerprint": config.fingerprint,
        "git_commit": _git_commit(config.repo_root),
        "creation_code_revision": (
            study.user_attrs.get("code_revision") if study is not None else None
        ),
        "current_code_revision": _code_revision(config.repo_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "optuna_version": getattr(optuna, "__version__", None),
        "configured_trials": config.total_trials,
        "terminal_trials": terminal_trial_count(study) if study is not None else 0,
        "error": error,
    }
    _atomic_json(path, manifest)


def _validate_or_initialize_study(study: Any, config: ResolvedBayesianConfig) -> None:
    current_revision = _code_revision(config.repo_root)
    existing = study.user_attrs.get("config_fingerprint")
    if existing is None:
        if study.trials:
            raise ConfigError(
                "existing study has trials but no configuration fingerprint"
            )
        study.set_user_attr("config_fingerprint", config.fingerprint)
        study.set_user_attr("immutable_config", config.immutable_payload())
        study.set_user_attr("code_revision", current_revision)
    elif existing != config.fingerprint:
        raise ConfigError(
            "refusing to resume: fixed arguments, search space, sampler, objective, "
            "execution, or tracking settings changed"
        )
    else:
        if "immutable_config" not in study.user_attrs:
            if study.trials:
                raise ConfigError(
                    "existing study has trials but no immutable config; "
                    "refusing an unverifiable resume"
                )
            study.set_user_attr("immutable_config", config.immutable_payload())
        if "code_revision" not in study.user_attrs:
            if study.trials:
                raise ConfigError(
                    "existing study has trials but no code revision; "
                    "refusing an unverifiable resume"
                )
            study.set_user_attr("code_revision", current_revision)
        elif study.user_attrs.get("code_revision") != current_revision:
            raise ConfigError(
                "refusing to resume because the MLAgentBench/scripts code revision "
                "changed; use the original revision or a new experiment name"
            )
    if (
        getattr(study.direction, "name", str(study.direction)).lower()
        != config.objective.direction
    ):
        raise ConfigError(
            "existing study objective direction does not match the config"
        )


def _recover_running_trials(study: Any) -> int:
    assert optuna is not None
    running = [
        trial
        for trial in study.trials
        if getattr(trial.state, "name", str(trial.state)) == "RUNNING"
    ]
    for trial in running:
        records = trial.user_attrs.get("run_records")
        if isinstance(records, list) and records:
            recovered_records = [
                dict(record) if isinstance(record, dict) else record
                for record in records
            ]
            last = recovered_records[-1]
            if isinstance(last, dict) and last.get("status") in {
                "starting",
                "runner_running",
                "evaluator_running",
            }:
                last["status"] = "abandoned_after_restart"
                last["failure_category"] = "abandoned_after_restart"
                last["failure_reason"] = (
                    "optimizer stopped while this run was in progress"
                )
                last["finished_at"] = _utc_now()
            study._storage.set_trial_user_attr(  # noqa: SLF001 - no public resume API
                trial._trial_id, "run_records", recovered_records
            )
        study._storage.set_trial_user_attr(  # noqa: SLF001 - no public resume API
            trial._trial_id, "failure_category", "abandoned_after_restart"
        )
        study._storage.set_trial_user_attr(  # noqa: SLF001 - no public resume API
            trial._trial_id,
            "failure_reason",
            "optimizer stopped while this trial was in progress",
        )
        study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
    return len(running)


class BayesianObjective:
    def __init__(
        self,
        config: ResolvedBayesianConfig,
        *,
        process_backend: ProcessBackend,
        python_command: Sequence[str],
        save_sampler: Callable[[], None],
    ) -> None:
        self.config = config
        self.process_backend = process_backend
        self.python_command = tuple(python_command)
        self.save_sampler = save_sampler

    def __call__(self, trial: Any) -> float:
        params = suggest_parameters(trial, self.config.space)
        # Persist RNG state immediately after suggestions, before a long-running
        # subprocess can be interrupted.
        self.save_sampler()
        records: list[dict[str, Any]] = []
        trial.set_user_attr("run_records", records)
        try:
            for run_index in range(1, self.config.execution.runs_per_trial + 1):

                def persist_in_progress(record: dict[str, Any]) -> None:
                    trial.set_user_attr("run_records", [*records, record])

                try:
                    record = execute_run(
                        self.config,
                        trial_number=trial.number,
                        run_index=run_index,
                        params=params,
                        process_backend=self.process_backend,
                        python_command=self.python_command,
                        record_updated=persist_in_progress,
                    )
                except TrialExecutionError as exc:
                    records.append(exc.run_record)
                    trial.set_user_attr("run_records", records)
                    trial.set_user_attr("failure_category", exc.category)
                    trial.set_user_attr("failure_reason", str(exc))
                    raise
                records.append(record)
                trial.set_user_attr("run_records", records)

            scores = [float(record["score"]) for record in records]
            value = aggregate_scores(scores, self.config.objective.run_aggregation)
            # Record the final objective as an Optuna intermediate value too.
            # We intentionally use a NopPruner: every expensive benchmark run
            # completes, while Optuna still exposes the reported score.
            trial.report(value, step=0)
            trial.set_user_attr("aggregate_score", value)
            trial.set_user_attr("failure_category", None)
            trial.set_user_attr("failure_reason", None)
            return value
        except (KeyboardInterrupt, SystemExit):
            raise


def run_bayesian_study(
    config: ResolvedBayesianConfig,
    *,
    process_backend: ProcessBackend | None = None,
    python_command: Sequence[str] = (sys.executable,),
) -> Any:
    """Create or resume a sequential TPE study and execute its remaining budget."""
    if optuna is None:
        raise ConfigError(
            "Optuna is not installed; install requirements_main.txt in the benchmark environment"
        )
    if not python_command:
        raise ConfigError("python_command must not be empty")

    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.logs_root.mkdir(parents=True, exist_ok=True)
    config.work_root.mkdir(parents=True, exist_ok=True)
    backend = process_backend or SubprocessBackend()

    with exclusive_study_lock(config.lock_path):
        database_existed = config.sqlite_path.exists()
        if database_existed and not config.persistence.resume:
            raise ConfigError(
                f"study already exists at {config.sqlite_path}; enable resume or use a new experiment name"
            )

        if not database_existed:
            stale_results = sorted(
                path
                for path in config.results_dir.iterdir()
                if path != config.lock_path
            )
            stale_runs = sorted(
                [*config.logs_root.glob(f"{config.experiment_name}_trial_*")]
                + [*config.work_root.glob(f"{config.experiment_name}_trial_*")]
            )
            stale_artifacts = [*stale_results, *stale_runs]
            if stale_artifacts:
                preview = ", ".join(str(path) for path in stale_artifacts[:3])
                suffix = " ..." if len(stale_artifacts) > 3 else ""
                raise ConfigError(
                    f"study database is missing but prior artifacts exist: "
                    f"{preview}{suffix}; restore study.sqlite3 or use a new "
                    "experiment name"
                )

        sampler = _load_sampler(config)
        if sampler is None:
            sampler = optuna.samplers.TPESampler(
                seed=config.sampler_seed,
                n_startup_trials=config.n_initial,
            )

        storage = optuna.storages.RDBStorage(url=f"sqlite:///{config.sqlite_path}")
        existing_studies = {
            summary.study_name for summary in optuna.get_all_study_summaries(storage)
        }
        if existing_studies and config.persistence.study_name not in existing_studies:
            raise ConfigError(
                "study database already belongs to: "
                + ", ".join(sorted(existing_studies))
            )

        study = optuna.create_study(
            study_name=config.persistence.study_name,
            storage=storage,
            sampler=sampler,
            direction=config.objective.direction,
            load_if_exists=config.persistence.resume,
            pruner=optuna.pruners.NopPruner(),
        )

        _validate_or_initialize_study(study, config)
        trials_requiring_sampler_state = [
            trial
            for trial in study.trials
            if getattr(trial.state, "name", str(trial.state)) != "WAITING"
        ]
        if (
            database_existed
            and trials_requiring_sampler_state
            and not config.sampler_path.exists()
        ):
            if config.persistence.require_sampler_state_on_resume:
                raise ConfigError(
                    "existing study has trials but sampler.pkl is missing; refusing a non-reproducible resume"
                )

        recovered = _recover_running_trials(study)
        if recovered:
            print(f"Recovered {recovered} abandoned RUNNING trial(s) as FAIL.")

        attempted = terminal_trial_count(study)
        if config.total_trials < attempted:
            raise ConfigError(
                f"configured trial budget ({config.total_trials}) is below the "
                f"existing terminal trial count ({attempted}); a resumed budget "
                "may only stay the same or increase"
            )

        # Idempotent on every resume. If the process stopped part-way through
        # enqueueing, skip_if_exists preserves prior points and adds the rest.
        for candidate in config.enqueue:
            study.enqueue_trial(candidate, skip_if_exists=True)

        _atomic_json(config.results_dir / "config.resolved.json", config.to_snapshot())
        _save_sampler(config, sampler)
        export_study_artifacts(study, config)
        _write_manifest(config, status="running", study=study)

        def after_trial(current_study: Any, _trial: Any) -> None:
            _save_sampler(config, current_study.sampler)
            export_study_artifacts(current_study, config)
            _write_manifest(config, status="running", study=current_study)

        objective = BayesianObjective(
            config,
            process_backend=backend,
            python_command=python_command,
            save_sampler=lambda: _save_sampler(config, study.sampler),
        )
        remaining = remaining_attempts(study, config.total_trials)
        print(
            f"Study {study.study_name}: {terminal_trial_count(study)}/{config.total_trials} "
            f"attempted, {remaining} remaining."
        )
        try:
            if remaining:
                caught: tuple[type[Exception], ...] = (
                    (TrialExecutionError,)
                    if config.execution.continue_on_trial_failure
                    else ()
                )
                study.optimize(
                    objective,
                    n_trials=remaining,
                    n_jobs=1,
                    catch=caught,
                    callbacks=[after_trial],
                    gc_after_trial=True,
                )
        except BaseException as exc:
            export_study_artifacts(study, config)
            _save_sampler(config, study.sampler)
            _write_manifest(
                config,
                status="interrupted",
                study=study,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        export_study_artifacts(study, config)
        _save_sampler(config, study.sampler)
        _write_manifest(config, status="complete", study=study)
        return study


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run or resume an Optuna Bayesian MLAgentBench comparison study."
    )
    parser.add_argument("config", help="Path to comparison_bayes.json")
    parser.add_argument(
        "--storage-dir",
        default=os.environ.get("STORAGE_DIR"),
        help="Persistent storage root (default: STORAGE_DIR or repository root)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and print the resolved configuration without creating a study",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    storage_dir = Path(args.storage_dir).expanduser() if args.storage_dir else repo_root
    previous_sigterm: Any = None
    if hasattr(signal, "SIGTERM"):
        try:
            previous_sigterm = signal.signal(
                signal.SIGTERM,
                _sigterm_to_interrupt,
            )
        except ValueError:  # pragma: no cover - main() called off main thread.
            previous_sigterm = None
    try:
        config = load_and_validate_config(
            config_path,
            storage_dir=storage_dir,
            repo_root=repo_root,
        )
        if args.validate_only:
            print(json.dumps(config.to_snapshot(), indent=2, sort_keys=True))
            return 0
        study = run_bayesian_study(config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except TrialExecutionError as exc:
        print(
            f"trial execution error ({exc.category}): {exc}",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print(
            "Optimization interrupted; persistent study state was saved.",
            file=sys.stderr,
        )
        return 130
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)

    completed = sum(
        getattr(trial.state, "name", str(trial.state)) == "COMPLETE"
        for trial in study.trials
    )
    print(
        f"Study complete: {terminal_trial_count(study)} attempted, "
        f"{completed} completed. Results: {config.results_dir}"
    )
    if completed:
        print(f"Best value: {study.best_value}; params: {study.best_params}")
        return 0
    print("Study finished without a completed trial.", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
