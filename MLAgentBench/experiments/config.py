"""Configuration contract for persistent Bayesian comparison studies.

The optimizer deliberately accepts only a small, explicit subset of runner
arguments.  A misspelled or unsupported setting must fail before an expensive
GPU trial starts rather than being silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the schema."""


_EXPERIMENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PARAMETER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# argparse destination -> (CLI flag, value kind)
RUNNER_FIXED_ARGUMENTS: dict[str, tuple[str, str]] = {
    "task": ("--task", "str"),
    "llm_name": ("--llm-name", "str"),
    "fast_llm_name": ("--fast-llm-name", "str"),
    "edit_script_llm_name": ("--edit-script-llm-name", "str"),
    "agent_type": ("--agent-type", "str"),
    "max_steps": ("--max-steps", "positive_int"),
    "max_time": ("--max-time", "positive_int"),
    "device": ("--device", "int"),
    "agent_max_steps": ("--agent-max-steps", "positive_int"),
    "edit_script_llm_max_tokens": ("--edit-script-llm-max-tokens", "positive_int"),
    "max_steps_in_context": ("--max-steps-in-context", "nonnegative_int"),
    "max_observation_steps_in_context": (
        "--max-observation-steps-in-context",
        "nonnegative_int",
    ),
    "max_retries": ("--max-retries", "positive_int"),
    "retrieval": ("--retrieval", "bool_flag"),
    "actions_remove_from_prompt": ("--actions-remove-from-prompt", "str_list"),
    "actions_add_to_prompt": ("--actions-add-to-prompt", "str_list"),
    # Keep the runner's historical argparse spelling for compatibility.
    "valid_format_entires": ("--valid-format-entires", "str_list"),
}

SEARCH_PARAMETER_FLAGS: dict[str, str] = {
    "top_p": "--top-p",
    "temperature": "--temperature",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalized_float(value: int | float) -> float:
    number = float(value)
    return 0.0 if number == 0.0 else number


def _finite_number(value: Any, field: str) -> float:
    if not _is_number(value):
        raise ConfigError(f"{field} must be a number")
    number = _normalized_float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field} must be finite")
    return number


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{field} must be >= {minimum}")
    return value


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{field} must be an object")
    return value


def _only_keys(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"{field} contains unsupported keys: {', '.join(unknown)}")


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def _safe_relative_path(value: Any, field: str) -> str:
    raw = _nonempty_string(value, field)
    path = Path(raw)
    if raw == "." or path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{field} must be a safe relative path")
    if any(part in {"", "."} for part in path.parts):
        raise ConfigError(f"{field} must not contain empty or '.' path segments")
    return path.as_posix()


def _json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str
    low: int | float | None = None
    high: int | float | None = None
    step: int | float | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()

    def suggest(self, trial: Any) -> Any:
        """Ask an Optuna-like trial for a value using this distribution."""
        if self.kind == "float":
            kwargs: dict[str, Any] = {"log": self.log}
            if self.step is not None:
                kwargs["step"] = self.step
            return trial.suggest_float(self.name, self.low, self.high, **kwargs)
        if self.kind == "int":
            return trial.suggest_int(
                self.name,
                self.low,
                self.high,
                step=self.step if self.step is not None else 1,
                log=self.log,
            )
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        raise AssertionError(f"unsupported parameter kind {self.kind}")

    def contains(self, value: Any) -> bool:
        if self.kind == "categorical":
            return value in self.choices
        if not _is_number(value):
            return False
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        assert self.low is not None and self.high is not None
        if numeric < float(self.low) or numeric > float(self.high):
            return False
        if self.kind == "int" and not isinstance(value, int):
            return False
        if self.step is not None:
            quotient = (numeric - float(self.low)) / float(self.step)
            if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        if self.kind == "categorical":
            return {"type": self.kind, "choices": list(self.choices)}
        result: dict[str, Any] = {
            "type": self.kind,
            "low": self.low,
            "high": self.high,
        }
        if self.step is not None:
            result["step"] = self.step
        if self.log:
            result["log"] = True
        return result


@dataclass(frozen=True)
class ObjectiveConfig:
    metric: str
    direction: str
    run_aggregation: str
    behavioral_failure_policy: str
    behavioral_failure_value: float | None


@dataclass(frozen=True)
class ExecutionConfig:
    runs_per_trial: int
    subprocess_timeout_seconds: float
    continue_on_trial_failure: bool


@dataclass(frozen=True)
class PersistenceConfig:
    resume: bool
    require_sampler_state_on_resume: bool
    study_name: str


@dataclass(frozen=True)
class ResolvedBayesianConfig:
    schema_version: int
    config_path: Path
    repo_root: Path
    storage_dir: Path
    experiment_name: str
    output_dir: str
    fixed: dict[str, Any]
    sampler_name: str
    sampler_seed: int
    n_initial: int
    n_iter: int
    space: dict[str, ParameterSpec]
    enqueue: tuple[dict[str, Any], ...]
    objective: ObjectiveConfig
    execution: ExecutionConfig
    persistence: PersistenceConfig
    use_codecarbon: bool

    @property
    def total_trials(self) -> int:
        return self.n_initial + self.n_iter

    @property
    def results_dir(self) -> Path:
        return self.storage_dir / self.output_dir / self.experiment_name

    @property
    def logs_root(self) -> Path:
        return self.storage_dir / "logs"

    @property
    def work_root(self) -> Path:
        return self.storage_dir / "workspace"

    @property
    def sqlite_path(self) -> Path:
        return self.results_dir / "study.sqlite3"

    @property
    def sampler_path(self) -> Path:
        return self.results_dir / "sampler.pkl"

    @property
    def lock_path(self) -> Path:
        return self.results_dir / "study.lock"

    def immutable_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "study_name": self.persistence.study_name,
            "fixed": self.fixed,
            "sampler": {
                "name": self.sampler_name,
                "seed": self.sampler_seed,
                "n_initial": self.n_initial,
            },
            "space": {
                name: spec.to_dict() for name, spec in sorted(self.space.items())
            },
            "enqueue": list(self.enqueue),
            "objective": {
                "metric": self.objective.metric,
                "direction": self.objective.direction,
                "run_aggregation": self.objective.run_aggregation,
                "behavioral_failure_policy": self.objective.behavioral_failure_policy,
                "behavioral_failure_value": self.objective.behavioral_failure_value,
            },
            "runs_per_trial": self.execution.runs_per_trial,
            "subprocess_timeout_seconds": (self.execution.subprocess_timeout_seconds),
            "use_codecarbon": self.use_codecarbon,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.immutable_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": {
                "name": self.experiment_name,
                "output_dir": self.output_dir,
            },
            "fixed": self.fixed,
            "search": {
                "method": "bayesian",
                "sampler": {"name": self.sampler_name, "seed": self.sampler_seed},
                "n_initial": self.n_initial,
                "n_iter": self.n_iter,
                "total_trials": self.total_trials,
                "space": {
                    name: self.space[name].to_dict() for name in sorted(self.space)
                },
                "enqueue": list(self.enqueue),
            },
            "objective": {
                "metric": self.objective.metric,
                "direction": self.objective.direction,
                "run_aggregation": self.objective.run_aggregation,
                "behavioral_failure": {
                    "policy": self.objective.behavioral_failure_policy,
                    "value": self.objective.behavioral_failure_value,
                },
            },
            "execution": {
                "runs_per_trial": self.execution.runs_per_trial,
                "subprocess_timeout_seconds": self.execution.subprocess_timeout_seconds,
                "continue_on_trial_failure": self.execution.continue_on_trial_failure,
            },
            "persistence": {
                "resume": self.persistence.resume,
                "require_sampler_state_on_resume": (
                    self.persistence.require_sampler_state_on_resume
                ),
                "study_name": self.persistence.study_name,
            },
            "tracking": {"codecarbon": self.use_codecarbon},
            "resolved_paths": {
                "storage_dir": str(self.storage_dir),
                "results_dir": str(self.results_dir),
                "logs_root": str(self.logs_root),
                "work_root": str(self.work_root),
            },
            "config_fingerprint": self.fingerprint,
        }


def _parse_fixed(raw: Any) -> dict[str, Any]:
    fixed = _mapping(raw, "fixed")
    _only_keys(fixed, set(RUNNER_FIXED_ARGUMENTS), "fixed")

    if "task" not in fixed:
        raise ConfigError("fixed.task is required")
    if "llm_name" not in fixed:
        raise ConfigError("fixed.llm_name is required")

    resolved = dict(fixed)
    resolved.setdefault("fast_llm_name", resolved["llm_name"])
    resolved.setdefault("edit_script_llm_name", resolved["llm_name"])
    resolved.setdefault("agent_type", "ResearchAgent")
    resolved.setdefault("max_steps", 30)
    resolved.setdefault("max_time", 18000)
    resolved.setdefault("device", 0)

    for name, value in resolved.items():
        kind = RUNNER_FIXED_ARGUMENTS[name][1]
        field = f"fixed.{name}"
        if kind == "str":
            resolved[name] = _nonempty_string(value, field)
        elif kind == "positive_int":
            resolved[name] = _integer(value, field, minimum=1)
        elif kind == "nonnegative_int":
            resolved[name] = _integer(value, field, minimum=0)
        elif kind == "int":
            resolved[name] = _integer(value, field)
        elif kind == "bool_flag":
            resolved[name] = _boolean(value, field)
        elif kind == "str_list":
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item for item in value
            ):
                raise ConfigError(f"{field} must be a list of non-empty strings")
        else:
            raise AssertionError(kind)
    return resolved


def _parse_parameter(name: str, raw: Any) -> ParameterSpec:
    if not _PARAMETER_NAME_RE.match(name):
        raise ConfigError(f"invalid search parameter name: {name}")
    if name not in SEARCH_PARAMETER_FLAGS:
        allowed = ", ".join(sorted(SEARCH_PARAMETER_FLAGS))
        raise ConfigError(f"unsupported search parameter {name}; allowed: {allowed}")

    data = _mapping(raw, f"search.space.{name}")
    kind = _nonempty_string(data.get("type"), f"search.space.{name}.type").lower()

    if kind in {"float", "int"}:
        _only_keys(data, {"type", "low", "high", "step", "log"}, f"search.space.{name}")
        if "low" not in data or "high" not in data:
            raise ConfigError(f"search.space.{name} requires low and high")
        if kind == "float":
            low: int | float = _finite_number(data["low"], f"search.space.{name}.low")
            high: int | float = _finite_number(
                data["high"], f"search.space.{name}.high"
            )
            step = (
                _finite_number(data["step"], f"search.space.{name}.step")
                if "step" in data
                else None
            )
        else:
            low = _integer(data["low"], f"search.space.{name}.low")
            high = _integer(data["high"], f"search.space.{name}.high")
            step = (
                _integer(data["step"], f"search.space.{name}.step", minimum=1)
                if "step" in data
                else 1
            )
        if low >= high:
            raise ConfigError(f"search.space.{name}.low must be less than high")
        if step is not None and step <= 0:
            raise ConfigError(f"search.space.{name}.step must be positive")
        if step is not None:
            intervals = (float(high) - float(low)) / float(step)
            if not math.isclose(intervals, round(intervals), abs_tol=1e-9):
                raise ConfigError(
                    f"search.space.{name}.step must divide the low-to-high interval"
                )
        log = _boolean(data.get("log", False), f"search.space.{name}.log")
        if log and low <= 0:
            raise ConfigError(f"search.space.{name}.low must be > 0 when log=true")
        if log and "step" in data:
            raise ConfigError(f"search.space.{name} cannot combine step and log")
        spec = ParameterSpec(name, kind, low=low, high=high, step=step, log=log)
    elif kind == "categorical":
        _only_keys(data, {"type", "choices"}, f"search.space.{name}")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ConfigError(f"search.space.{name}.choices must be a non-empty list")
        if not all(_json_scalar(choice) for choice in choices):
            raise ConfigError(f"search.space.{name}.choices must contain JSON scalars")
        canonical = [
            f"number:{_normalized_float(choice)!r}"
            if _is_number(choice)
            else json.dumps(choice, sort_keys=True)
            for choice in choices
        ]
        if len(set(canonical)) != len(canonical):
            raise ConfigError(f"search.space.{name}.choices must be unique")
        spec = ParameterSpec(name, kind, choices=tuple(choices))
    else:
        raise ConfigError(
            f"search.space.{name}.type must be float, int, or categorical"
        )

    # The current runner accepts these values as continuous floats.  Supporting
    # categorical numeric choices is useful for finite-grid comparisons, but an
    # integer distribution would change their semantics.
    if name in {"top_p", "temperature"} and spec.kind == "int":
        raise ConfigError(f"search parameter {name} cannot use an integer distribution")

    values: Sequence[Any]
    if spec.kind == "categorical":
        values = spec.choices
    else:
        values = [spec.low, spec.high]
    for value in values:
        if not _is_number(value):
            raise ConfigError(f"search parameter {name} must contain numeric values")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ConfigError(f"search parameter {name} must contain finite values")
        if name == "top_p" and not (0.0 < numeric <= 1.0):
            raise ConfigError("top_p values must be in (0, 1]")
        if name == "temperature" and numeric < 0.0:
            raise ConfigError("temperature values must be >= 0")
    if spec.kind == "categorical":
        # All currently supported parameters are numeric. Canonicalize them so
        # JSON 1 and 1.0 represent one Optuna categorical point.
        spec = ParameterSpec(
            name,
            spec.kind,
            choices=tuple(_normalized_float(value) for value in spec.choices),
        )
    return spec


def _parse_enqueue(
    raw: Any, space: Mapping[str, ParameterSpec]
) -> tuple[dict[str, Any], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ConfigError("search.enqueue must be a list")
    result: list[dict[str, Any]] = []
    canonical_candidates: set[str] = set()
    for index, candidate_raw in enumerate(raw):
        candidate = _mapping(candidate_raw, f"search.enqueue[{index}]")
        if set(candidate) != set(space):
            raise ConfigError(
                f"search.enqueue[{index}] must provide exactly: {', '.join(sorted(space))}"
            )
        for name, value in candidate.items():
            if not space[name].contains(value):
                raise ConfigError(
                    f"search.enqueue[{index}].{name} is outside its distribution"
                )
        normalized = {
            name: _normalized_float(candidate[name])
            if space[name].kind in {"float", "categorical"}
            else candidate[name]
            for name in sorted(candidate)
        }
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        if canonical in canonical_candidates:
            raise ConfigError(
                f"search.enqueue[{index}] duplicates an earlier candidate"
            )
        canonical_candidates.add(canonical)
        result.append(normalized)
    return tuple(result)


def load_and_validate_config(
    path: str | Path,
    *,
    storage_dir: str | Path,
    repo_root: str | Path,
) -> ResolvedBayesianConfig:
    """Load a JSON config, apply defaults, and reject unsafe/ambiguous input."""
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open(encoding="utf-8") as config_file:
            root = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"failed to read config {config_path}: {exc}") from exc

    root = _mapping(root, "config root")
    _only_keys(
        root,
        {
            "schema_version",
            "experiment",
            "fixed",
            "search",
            "objective",
            "execution",
            "persistence",
            "tracking",
        },
        "config root",
    )

    schema_version = _integer(root.get("schema_version", 1), "schema_version")
    if schema_version != 1:
        raise ConfigError("schema_version must be 1")

    experiment = _mapping(root.get("experiment"), "experiment")
    _only_keys(experiment, {"name", "output_dir"}, "experiment")
    experiment_name = _nonempty_string(experiment.get("name"), "experiment.name")
    if not _EXPERIMENT_NAME_RE.match(experiment_name):
        raise ConfigError(
            "experiment.name may contain only letters, digits, '.', '_' and '-'"
        )
    output_dir = _safe_relative_path(
        experiment.get("output_dir", "results"), "experiment.output_dir"
    )

    fixed = _parse_fixed(root.get("fixed"))

    search = _mapping(root.get("search"), "search")
    _only_keys(
        search,
        {"method", "sampler", "n_initial", "n_iter", "space", "enqueue"},
        "search",
    )
    method = _nonempty_string(search.get("method"), "search.method").lower()
    if method != "bayesian":
        raise ConfigError("search.method must be 'bayesian'")
    sampler = _mapping(
        search.get("sampler", {"name": "tpe", "seed": 42}), "search.sampler"
    )
    _only_keys(sampler, {"name", "seed"}, "search.sampler")
    sampler_name = _nonempty_string(
        sampler.get("name", "tpe"), "search.sampler.name"
    ).lower()
    if sampler_name != "tpe":
        raise ConfigError("search.sampler.name must be 'tpe'")
    sampler_seed = _integer(sampler.get("seed", 42), "search.sampler.seed", minimum=0)
    if sampler_seed > 2**32 - 1:
        raise ConfigError("search.sampler.seed must be <= 4294967295")
    n_initial = _integer(search.get("n_initial", 6), "search.n_initial", minimum=0)
    n_iter = _integer(search.get("n_iter", 12), "search.n_iter", minimum=0)
    if n_initial + n_iter <= 0:
        raise ConfigError("search.n_initial + search.n_iter must be positive")
    raw_space = _mapping(search.get("space"), "search.space")
    if not raw_space:
        raise ConfigError("search.space must not be empty")
    space = {name: _parse_parameter(name, spec) for name, spec in raw_space.items()}
    enqueue = _parse_enqueue(search.get("enqueue"), space)
    if len(enqueue) > n_initial + n_iter:
        raise ConfigError(
            "search.enqueue cannot contain more candidates than the total trial budget"
        )

    objective_raw = _mapping(root.get("objective"), "objective")
    _only_keys(
        objective_raw,
        {"metric", "direction", "run_aggregation", "behavioral_failure"},
        "objective",
    )
    metric = _nonempty_string(
        objective_raw.get("metric", "final_score"), "objective.metric"
    )
    if metric != "final_score":
        raise ConfigError("objective.metric currently supports only 'final_score'")
    direction = _nonempty_string(
        objective_raw.get("direction"), "objective.direction"
    ).lower()
    if direction not in {"maximize", "minimize"}:
        raise ConfigError("objective.direction must be 'maximize' or 'minimize'")
    aggregation = _nonempty_string(
        objective_raw.get("run_aggregation", "mean"), "objective.run_aggregation"
    ).lower()
    if aggregation not in {"mean", "median"}:
        raise ConfigError("objective.run_aggregation must be 'mean' or 'median'")
    behavior = _mapping(
        objective_raw.get("behavioral_failure", {"policy": "fail"}),
        "objective.behavioral_failure",
    )
    _only_keys(behavior, {"policy", "value"}, "objective.behavioral_failure")
    behavior_policy = _nonempty_string(
        behavior.get("policy", "fail"), "objective.behavioral_failure.policy"
    ).lower()
    if behavior_policy not in {"fail", "penalty"}:
        raise ConfigError(
            "objective.behavioral_failure.policy must be 'fail' or 'penalty'"
        )
    behavior_value: float | None = None
    if behavior_policy == "penalty":
        if "value" not in behavior:
            raise ConfigError(
                "objective.behavioral_failure.value is required for penalty"
            )
        behavior_value = _finite_number(
            behavior["value"], "objective.behavioral_failure.value"
        )
    elif "value" in behavior and behavior["value"] is not None:
        raise ConfigError(
            "objective.behavioral_failure.value is only valid for penalty"
        )

    execution_raw = _mapping(root.get("execution", {}), "execution")
    _only_keys(
        execution_raw,
        {"runs_per_trial", "subprocess_timeout_seconds", "continue_on_trial_failure"},
        "execution",
    )
    runs_per_trial = _integer(
        execution_raw.get("runs_per_trial", 1), "execution.runs_per_trial", minimum=1
    )
    timeout = _finite_number(
        execution_raw.get("subprocess_timeout_seconds", fixed["max_time"] + 300),
        "execution.subprocess_timeout_seconds",
    )
    if timeout <= 0:
        raise ConfigError("execution.subprocess_timeout_seconds must be positive")
    continue_on_failure = _boolean(
        execution_raw.get("continue_on_trial_failure", True),
        "execution.continue_on_trial_failure",
    )

    persistence_raw = _mapping(root.get("persistence", {}), "persistence")
    _only_keys(
        persistence_raw,
        {"resume", "require_sampler_state_on_resume", "study_name"},
        "persistence",
    )
    resume = _boolean(persistence_raw.get("resume", True), "persistence.resume")
    require_sampler = _boolean(
        persistence_raw.get("require_sampler_state_on_resume", True),
        "persistence.require_sampler_state_on_resume",
    )
    study_name = _nonempty_string(
        persistence_raw.get("study_name", experiment_name), "persistence.study_name"
    )

    tracking_raw = _mapping(root.get("tracking", {}), "tracking")
    _only_keys(tracking_raw, {"codecarbon"}, "tracking")
    use_codecarbon = _boolean(
        tracking_raw.get("codecarbon", False), "tracking.codecarbon"
    )

    return ResolvedBayesianConfig(
        schema_version=schema_version,
        config_path=config_path,
        repo_root=Path(repo_root).expanduser().resolve(),
        storage_dir=Path(storage_dir).expanduser().resolve(),
        experiment_name=experiment_name,
        output_dir=output_dir,
        fixed=fixed,
        sampler_name=sampler_name,
        sampler_seed=sampler_seed,
        n_initial=n_initial,
        n_iter=n_iter,
        space=space,
        enqueue=enqueue,
        objective=ObjectiveConfig(
            metric=metric,
            direction=direction,
            run_aggregation=aggregation,
            behavioral_failure_policy=behavior_policy,
            behavioral_failure_value=behavior_value,
        ),
        execution=ExecutionConfig(
            runs_per_trial=runs_per_trial,
            subprocess_timeout_seconds=timeout,
            continue_on_trial_failure=continue_on_failure,
        ),
        persistence=PersistenceConfig(
            resume=resume,
            require_sampler_state_on_resume=require_sampler,
            study_name=study_name,
        ),
        use_codecarbon=use_codecarbon,
    )
