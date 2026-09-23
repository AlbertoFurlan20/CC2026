from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path
from typing import Any, Sequence

import pytest

optuna = pytest.importorskip("optuna")

from MLAgentBench.experiments.bayesian import (  # noqa: E402
    BayesianObjective,
    CommandResult,
    _recover_running_trials,
    _validate_or_initialize_study,
    export_study_artifacts,
    run_bayesian_study,
)
from MLAgentBench.experiments.config import ConfigError  # noqa: E402


def _arg(argv: Sequence[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


class MixedOutcomeBackend:
    """GPU-free backend exercising fail, penalty, and success in one study."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _trial_number(log_dir: Path) -> int:
        match = re.search(r"_trial_(\d+)_run_", log_dir.name)
        assert match is not None
        return int(match.group(1))

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        module = argv[argv.index("-m") + 1]
        if module == "MLAgentBench.runner":
            log_dir = Path(_arg(argv, "--log-dir"))
        else:
            log_dir = Path(_arg(argv, "--log-folder"))
        trial_number = self._trial_number(log_dir)
        self.calls.append(
            {
                "module": module,
                "trial_number": trial_number,
                "argv": list(argv),
                "cwd": str(cwd),
                "timeout_seconds": timeout_seconds,
            }
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            f"fake {module} for trial {trial_number}\n", encoding="utf-8"
        )

        # The first proposal simulates an infrastructure/runner failure.
        if module == "MLAgentBench.runner" and trial_number == 0:
            return CommandResult(returncode=17, duration_seconds=0.01)

        if module == "MLAgentBench.eval":
            eval_path = Path(_arg(argv, "--output-file"))
            expected_trace = log_dir / "env_log" / "trace.json"
            if trial_number == 1:
                score = -1
                evidence: list[float] = []
            else:
                score = 0.75
                evidence = [score]
            payload = {
                str(expected_trace): {
                    "final_score": score,
                    "score": evidence,
                    "submitted_final_answer": trial_number == 2,
                    "total_time": 1.25,
                    "extra": {},
                }
            }
            eval_path.write_text(json.dumps(payload), encoding="utf-8")

        return CommandResult(returncode=0, duration_seconds=0.01)


class SuccessfulBackend(MixedOutcomeBackend):
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path,
        timeout_seconds: float,
    ) -> CommandResult:
        module = argv[argv.index("-m") + 1]
        if module == "MLAgentBench.runner":
            log_dir = Path(_arg(argv, "--log-dir"))
        else:
            log_dir = Path(_arg(argv, "--log-folder"))
        trial_number = self._trial_number(log_dir)
        self.calls.append(
            {"module": module, "trial_number": trial_number, "argv": list(argv)}
        )
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("success\n", encoding="utf-8")
        if module == "MLAgentBench.eval":
            score = 0.5 + trial_number / 100
            Path(_arg(argv, "--output-file")).write_text(
                json.dumps(
                    {
                        str(log_dir / "env_log" / "trace.json"): {
                            "final_score": score,
                            "score": [score],
                            "extra": {},
                        }
                    }
                ),
                encoding="utf-8",
            )
        return CommandResult(returncode=0, duration_seconds=0.01)


def test_gpu_free_persistent_study_continues_and_resume_is_noop(
    resolve_bayesian_config: Any,
) -> None:
    config = resolve_bayesian_config()
    backend = MixedOutcomeBackend()

    study = run_bayesian_study(
        config,
        process_backend=backend,
        python_command=("fake-python",),
    )

    assert [trial.state.name for trial in study.trials] == [
        "FAIL",
        "COMPLETE",
        "COMPLETE",
    ]
    assert study.trials[0].user_attrs["failure_category"] == "runner_failure"
    assert study.trials[1].value == 0.0
    assert study.trials[1].user_attrs["run_records"][0]["status"] == (
        "behavioral_penalty"
    )
    assert study.trials[2].value == 0.75
    assert study.best_trial.number == 2
    assert [call["trial_number"] for call in backend.calls] == [0, 1, 1, 2, 2]
    assert all(call["cwd"] == str(config.repo_root) for call in backend.calls)

    expected_artifacts = {
        "study.sqlite3",
        "sampler.pkl",
        "config.resolved.json",
        "manifest.json",
        "trials.csv",
        "runs.csv",
        "best.json",
        "study.lock",
    }
    assert expected_artifacts <= {path.name for path in config.results_dir.iterdir()}
    assert not list(config.results_dir.glob(".*.tmp.*"))

    with (config.results_dir / "trials.csv").open(
        newline="", encoding="utf-8"
    ) as source:
        trial_rows = list(csv.DictReader(source))
    assert [row["state"] for row in trial_rows] == ["FAIL", "COMPLETE", "COMPLETE"]
    assert trial_rows[0]["failure_category"] == "runner_failure"
    assert trial_rows[2]["objective_value"] == "0.75"

    with (config.results_dir / "runs.csv").open(newline="", encoding="utf-8") as source:
        run_rows = list(csv.DictReader(source))
    assert [row["status"] for row in run_rows] == [
        "runner_failure",
        "behavioral_penalty",
        "complete",
    ]
    assert [row["run_index"] for row in run_rows] == ["1", "1", "1"]

    best = json.loads((config.results_dir / "best.json").read_text(encoding="utf-8"))
    assert best["status"] == "complete"
    assert best["trial_number"] == 2
    assert best["objective_value"] == 0.75
    assert best["run_ids"] == ["bayes_test_trial_000002_run_01"]

    manifest = json.loads(
        (config.results_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "complete"
    assert manifest["configured_trials"] == 3
    assert manifest["terminal_trials"] == 3
    snapshot = json.loads(
        (config.results_dir / "config.resolved.json").read_text(encoding="utf-8")
    )
    assert snapshot["config_fingerprint"] == config.fingerprint

    call_count = len(backend.calls)
    resumed = run_bayesian_study(
        config,
        process_backend=backend,
        python_command=("fake-python",),
    )
    assert len(resumed.trials) == 3
    assert len(backend.calls) == call_count


def test_resume_rejects_incompatible_immutable_config(
    resolve_bayesian_config: Any,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["search"]["n_initial"] = 0
    payload["search"]["n_iter"] = 1
    payload["search"]["enqueue"] = []
    config = resolve_bayesian_config(payload)
    backend = MixedOutcomeBackend()
    run_bayesian_study(
        config,
        process_backend=backend,
        python_command=("fake-python",),
    )

    changed = copy.deepcopy(payload)
    changed["search"]["space"]["temperature"]["high"] = 0.8
    changed_config = resolve_bayesian_config(
        changed,
        storage_dir=config.storage_dir,
        repo_root=config.repo_root,
    )
    with pytest.raises(ConfigError, match="refusing to resume"):
        run_bayesian_study(
            changed_config,
            process_backend=backend,
            python_command=("fake-python",),
        )


def test_existing_study_requires_resume_enabled(
    resolve_bayesian_config: Any,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["search"]["n_initial"] = 0
    payload["search"]["n_iter"] = 1
    payload["search"]["enqueue"] = []
    config = resolve_bayesian_config(payload)
    run_bayesian_study(
        config,
        process_backend=MixedOutcomeBackend(),
        python_command=("fake-python",),
    )

    no_resume = copy.deepcopy(payload)
    no_resume["persistence"]["resume"] = False
    no_resume_config = resolve_bayesian_config(
        no_resume,
        storage_dir=config.storage_dir,
        repo_root=config.repo_root,
    )
    with pytest.raises(ConfigError, match="study already exists"):
        run_bayesian_study(
            no_resume_config,
            process_backend=MixedOutcomeBackend(),
            python_command=("fake-python",),
        )


def test_bayesian_objective_persists_interrupted_in_flight_record(
    resolve_bayesian_config: Any,
) -> None:
    config = resolve_bayesian_config()

    class InterruptBackend:
        def run(
            self,
            argv: Sequence[str],
            *,
            cwd: Path,
            stdout_path: Path,
            timeout_seconds: float,
        ) -> CommandResult:
            raise KeyboardInterrupt("stop the study")

    objective = BayesianObjective(
        config,
        process_backend=InterruptBackend(),
        python_command=("fake-python",),
        save_sampler=lambda: None,
    )
    study = optuna.create_study(direction="maximize")
    with pytest.raises(KeyboardInterrupt, match="stop the study"):
        study.optimize(objective, n_trials=1)

    frozen = study.trials[0]
    assert frozen.state.name == "FAIL"
    records = frozen.user_attrs["run_records"]
    assert len(records) == 1
    assert records[0]["status"] == "interrupted"
    assert records[0]["failure_category"] == "interrupted"
    assert records[0]["failure_reason"] == "KeyboardInterrupt: stop the study"
    assert records[0]["finished_at"] is not None

    export_study_artifacts(study, config)
    with (config.results_dir / "runs.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 1
    assert rows[0]["status"] == "interrupted"


def test_recover_running_trial_marks_record_and_trial_as_abandoned() -> None:
    study = optuna.create_study(direction="maximize")
    live_trial = study.ask()
    live_trial.set_user_attr(
        "run_records",
        [
            {
                "trial_number": live_trial.number,
                "run_index": 1,
                "run_id": "stale_run",
                "status": "evaluator_running",
                "failure_category": None,
                "failure_reason": None,
                "finished_at": None,
            }
        ],
    )

    assert _recover_running_trials(study) == 1

    recovered = study.trials[0]
    assert recovered.state.name == "FAIL"
    assert recovered.user_attrs["failure_category"] == "abandoned_after_restart"
    assert "stopped while this trial" in recovered.user_attrs["failure_reason"]
    record = recovered.user_attrs["run_records"][0]
    assert record["status"] == "abandoned_after_restart"
    assert record["failure_category"] == "abandoned_after_restart"
    assert "run was in progress" in record["failure_reason"]
    assert record["finished_at"] is not None
    assert _recover_running_trials(study) == 0


def test_partial_enqueue_is_reconciled_idempotently_on_resume(
    resolve_bayesian_config: Any,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["search"]["n_initial"] = 3
    payload["search"]["n_iter"] = 0
    payload["search"]["enqueue"] = [
        {"top_p": 0.8, "temperature": 0.2},
        {"top_p": 0.9, "temperature": 0.4},
        {"top_p": 1.0, "temperature": 0.6},
    ]
    config = resolve_bayesian_config(payload)
    config.results_dir.mkdir(parents=True)
    storage = optuna.storages.RDBStorage(url=f"sqlite:///{config.sqlite_path}")
    partial = optuna.create_study(
        study_name=config.persistence.study_name,
        storage=storage,
        direction=config.objective.direction,
    )
    _validate_or_initialize_study(partial, config)
    partial.enqueue_trial(dict(config.enqueue[0]))
    assert [trial.state.name for trial in partial.trials] == ["WAITING"]

    backend = SuccessfulBackend()
    resumed = run_bayesian_study(
        config,
        process_backend=backend,
        python_command=("fake-python",),
    )

    assert [trial.params for trial in resumed.trials] == list(config.enqueue)
    assert [trial.state.name for trial in resumed.trials] == [
        "COMPLETE",
        "COMPLETE",
        "COMPLETE",
    ]
    calls_after_completion = len(backend.calls)
    run_bayesian_study(
        config,
        process_backend=backend,
        python_command=("fake-python",),
    )
    assert len(backend.calls) == calls_after_completion


@pytest.mark.parametrize("stale_kind", ["result", "log", "workspace"])
def test_new_study_refuses_orphaned_deterministic_artifacts(
    resolve_bayesian_config: Any,
    stale_kind: str,
) -> None:
    config = resolve_bayesian_config()
    assert not config.sqlite_path.exists()
    if stale_kind == "result":
        config.results_dir.mkdir(parents=True, exist_ok=True)
        (config.results_dir / "manifest.json").write_text("{}", encoding="utf-8")
    elif stale_kind == "log":
        (config.logs_root / "bayes_test_trial_000000_run_01").mkdir(
            parents=True, exist_ok=True
        )
    else:
        (config.work_root / "bayes_test_trial_000000_run_01").mkdir(
            parents=True, exist_ok=True
        )
    backend = SuccessfulBackend()

    with pytest.raises(
        ConfigError, match="database is missing but prior artifacts exist"
    ):
        run_bayesian_study(
            config,
            process_backend=backend,
            python_command=("fake-python",),
        )

    assert not config.sqlite_path.exists()
    assert backend.calls == []


def test_resume_rejects_reducing_trial_budget(
    resolve_bayesian_config: Any,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["search"]["n_initial"] = 0
    payload["search"]["n_iter"] = 2
    payload["search"]["enqueue"] = []
    config = resolve_bayesian_config(payload)
    run_bayesian_study(
        config,
        process_backend=MixedOutcomeBackend(),
        python_command=("fake-python",),
    )

    reduced = copy.deepcopy(payload)
    reduced["search"]["n_iter"] = 1
    reduced_config = resolve_bayesian_config(
        reduced,
        storage_dir=config.storage_dir,
        repo_root=config.repo_root,
    )
    with pytest.raises(ConfigError, match="may only stay the same or increase"):
        run_bayesian_study(
            reduced_config,
            process_backend=MixedOutcomeBackend(),
            python_command=("fake-python",),
        )
