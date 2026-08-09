from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from MLAgentBench.experiments.config import (
    ConfigError,
    ParameterSpec,
    load_and_validate_config,
)
from MLAgentBench.experiments.bayesian import suggest_parameters


class RecordingTrial:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def suggest_float(self, name: str, *args: Any, **kwargs: Any) -> float:
        self.calls.append(("float", (name, *args), kwargs))
        return 0.75

    def suggest_int(self, name: str, *args: Any, **kwargs: Any) -> int:
        self.calls.append(("int", (name, *args), kwargs))
        return 4

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.calls.append(("categorical", (name, choices), {}))
        return choices[-1]


def _set_path(payload: dict[str, Any], dotted: str, value: Any) -> None:
    target: dict[str, Any] = payload
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def test_load_config_applies_defaults_and_stable_fingerprint(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["fixed"] = {"task": "cifar10", "llm_name": "model"}
    payload.pop("execution")
    payload.pop("tracking")
    payload["persistence"] = {"resume": True}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    first = load_and_validate_config(
        config_path, storage_dir=tmp_path / "data", repo_root=tmp_path
    )
    second = load_and_validate_config(
        config_path, storage_dir=tmp_path / "other-data", repo_root=tmp_path
    )

    assert first.fixed == {
        "task": "cifar10",
        "llm_name": "model",
        "fast_llm_name": "model",
        "edit_script_llm_name": "model",
        "agent_type": "ResearchAgent",
        "max_steps": 30,
        "max_time": 18000,
        "device": 0,
    }
    assert first.execution.runs_per_trial == 1
    assert first.execution.subprocess_timeout_seconds == 18300
    assert first.use_codecarbon is False
    assert first.total_trials == 3
    assert first.fingerprint == second.fingerprint
    assert first.to_snapshot()["resolved_paths"]["storage_dir"] == str(
        (tmp_path / "data").resolve()
    )


def test_fingerprint_allows_budget_extension_but_not_timeout_changes(
    resolve_bayesian_config: Any,
    valid_bayesian_config: dict[str, Any],
) -> None:
    baseline = resolve_bayesian_config(valid_bayesian_config)

    extended = copy.deepcopy(valid_bayesian_config)
    extended["search"]["n_iter"] += 5
    assert resolve_bayesian_config(extended).fingerprint == baseline.fingerprint

    changed_timeout = copy.deepcopy(valid_bayesian_config)
    changed_timeout["execution"]["subprocess_timeout_seconds"] += 1
    assert resolve_bayesian_config(changed_timeout).fingerprint != baseline.fingerprint


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("schema_version", 2, "schema_version must be 1"),
        ("experiment.name", "../escape", "experiment.name may contain"),
        ("experiment.output_dir", "../escape", "safe relative path"),
        ("fixed.max_steps", True, "must be an integer"),
        ("fixed.max_time", 0, "must be >= 1"),
        ("search.method", "grid", "must be 'bayesian'"),
        ("search.sampler.name", "random", "must be 'tpe'"),
        ("search.sampler.seed", True, "must be an integer"),
        ("search.n_initial", -1, "must be >= 0"),
        ("objective.metric", "accuracy", "only 'final_score'"),
        ("objective.direction", "sideways", "maximize.*minimize"),
        ("objective.run_aggregation", "max", "mean.*median"),
        ("execution.runs_per_trial", 0, "must be >= 1"),
        ("execution.subprocess_timeout_seconds", float("inf"), "must be finite"),
        ("execution.continue_on_trial_failure", 1, "must be a boolean"),
        ("persistence.resume", "yes", "must be a boolean"),
        ("tracking.codecarbon", 1, "must be a boolean"),
    ],
)
def test_invalid_config_values_are_rejected(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
    path: str,
    value: Any,
    message: str,
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    _set_path(payload, path, value)
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_and_validate_config(config_path, storage_dir=tmp_path, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"type": "float", "low": 0.9, "high": 0.7}, "less than high"),
        ({"type": "float", "low": 0.7, "high": 0.9, "step": 0}, "positive"),
        ({"type": "float", "low": 0, "high": 0.9, "log": True}, "> 0"),
        (
            {"type": "float", "low": 0.7, "high": 0.9, "log": True, "step": 0.1},
            "cannot combine step and log",
        ),
        ({"type": "int", "low": 1, "high": 3}, "integer distribution"),
        ({"type": "categorical", "choices": []}, "non-empty list"),
        ({"type": "categorical", "choices": [0.7, 0.7]}, "must be unique"),
        ({"type": "categorical", "choices": [0.7, "bad"]}, "numeric values"),
        ({"type": "categorical", "choices": [0.0, 0.7]}, "top_p values"),
        ({"type": "mystery", "low": 0.7, "high": 0.9}, "float, int, or categorical"),
    ],
)
def test_invalid_parameter_distributions_are_rejected(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
    spec: dict[str, Any],
    message: str,
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["search"]["space"] = {"top_p": spec}
    payload["search"]["enqueue"] = []
    config_path = tmp_path / "invalid_space.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_and_validate_config(config_path, storage_dir=tmp_path, repo_root=tmp_path)


def test_unknown_keys_and_search_parameters_fail_fast(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["surprise"] = True
    path = tmp_path / "unknown_root.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported keys: surprise"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)

    payload = copy.deepcopy(valid_bayesian_config)
    payload["search"]["space"] = {"best_of": {"type": "int", "low": 1, "high": 3}}
    payload["search"]["enqueue"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported search parameter best_of"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)


@pytest.mark.parametrize(
    "candidate",
    [
        {"top_p": 0.9},
        {"top_p": 1.1, "temperature": 0.5},
        {"top_p": 0.9, "temperature": 1.0},
    ],
)
def test_enqueue_must_exactly_match_and_fit_space(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["search"]["enqueue"] = [candidate]
    path = tmp_path / "bad_enqueue.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ConfigError, match="must provide exactly|outside its distribution"
    ):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)


def test_enqueue_rejects_duplicate_candidates_and_budget_overflow(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    candidate = {"top_p": 0.8, "temperature": 0.3}
    # Reverse key order to prove duplicate detection is canonical, not based on
    # the insertion order of the JSON object.
    duplicate = {"temperature": 0.3, "top_p": 0.8}
    payload["search"]["enqueue"] = [candidate, duplicate]
    path = tmp_path / "duplicate_enqueue.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicates an earlier candidate"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)

    payload["search"]["enqueue"] = [
        {"top_p": 1, "temperature": 0.3},
        {"top_p": 1.0, "temperature": 0.3},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicates an earlier candidate"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)

    payload["search"]["space"]["temperature"]["low"] = 0.0
    payload["search"]["enqueue"] = [
        {"top_p": 0.8, "temperature": -0.0},
        {"top_p": 0.8, "temperature": 0.0},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicates an earlier candidate"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)

    payload["search"]["n_initial"] = 0
    payload["search"]["n_iter"] = 2
    payload["search"]["enqueue"] = [
        {"top_p": 0.8, "temperature": 0.2},
        {"top_p": 0.9, "temperature": 0.4},
        {"top_p": 1.0, "temperature": 0.6},
    ]
    path = tmp_path / "over_budget_enqueue.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ConfigError, match="more candidates than the total trial budget"
    ):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)


def test_behavioral_penalty_contract_is_strict(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
) -> None:
    payload = copy.deepcopy(valid_bayesian_config)
    payload["objective"]["behavioral_failure"] = {"policy": "penalty"}
    path = tmp_path / "missing_penalty.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="value is required"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)

    payload["objective"]["behavioral_failure"] = {"policy": "fail", "value": 0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="only valid for penalty"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)


def test_invalid_json_and_non_object_root_are_config_errors(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigError, match="failed to read config"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ConfigError, match="config root must be an object"):
        load_and_validate_config(path, storage_dir=tmp_path, repo_root=tmp_path)


def test_parameter_specs_dispatch_typed_optuna_suggestions() -> None:
    trial = RecordingTrial()
    space = {
        "float_param": ParameterSpec(
            "float_param", "float", low=0.1, high=1.0, step=0.1
        ),
        "int_param": ParameterSpec("int_param", "int", low=2, high=8, step=2),
        "choice_param": ParameterSpec(
            "choice_param", "categorical", choices=("a", "b")
        ),
    }

    values = suggest_parameters(trial, space)

    assert values == {"float_param": 0.75, "int_param": 4, "choice_param": "b"}
    assert trial.calls == [
        ("categorical", ("choice_param", ["a", "b"]), {}),
        ("float", ("float_param", 0.1, 1.0), {"log": False, "step": 0.1}),
        ("int", ("int_param", 2, 8), {"step": 2, "log": False}),
    ]


def test_suggestion_order_is_canonical() -> None:
    trial = RecordingTrial()
    space = {
        "z_choice": ParameterSpec("z_choice", "categorical", choices=("a", "b")),
        "a_float": ParameterSpec("a_float", "float", low=0.1, high=1.0),
    }

    assert list(suggest_parameters(trial, space)) == ["a_float", "z_choice"]
    assert [call[1][0] for call in trial.calls] == ["a_float", "z_choice"]
