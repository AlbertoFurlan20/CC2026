from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from MLAgentBench.experiments.config import (
    ResolvedBayesianConfig,
    load_and_validate_config,
)


@pytest.fixture
def valid_bayesian_config() -> dict[str, Any]:
    """Small, complete config suitable for unit and integration tests."""
    return {
        "schema_version": 1,
        "experiment": {"name": "bayes_test", "output_dir": "results"},
        "fixed": {
            "task": "cifar10",
            "llm_name": "test-model",
            "max_steps": 4,
            "max_time": 30,
        },
        "search": {
            "method": "bayesian",
            "sampler": {"name": "tpe", "seed": 7},
            "n_initial": 1,
            "n_iter": 2,
            "space": {
                "top_p": {"type": "float", "low": 0.7, "high": 1.0},
                "temperature": {
                    "type": "float",
                    "low": 0.1,
                    "high": 0.9,
                },
            },
            "enqueue": [{"top_p": 0.9, "temperature": 0.5}],
        },
        "objective": {
            "metric": "final_score",
            "direction": "maximize",
            "run_aggregation": "mean",
            "behavioral_failure": {"policy": "penalty", "value": 0.0},
        },
        "execution": {
            "runs_per_trial": 1,
            "subprocess_timeout_seconds": 60,
            "continue_on_trial_failure": True,
        },
        "persistence": {
            "resume": True,
            "require_sampler_state_on_resume": True,
            "study_name": "bayes_test_study",
        },
        "tracking": {"codecarbon": False},
    }


@pytest.fixture
def resolve_bayesian_config(
    tmp_path: Path,
    valid_bayesian_config: dict[str, Any],
) -> Callable[..., ResolvedBayesianConfig]:
    counter = 0

    def resolve(
        payload: dict[str, Any] | None = None,
        *,
        storage_dir: Path | None = None,
        repo_root: Path | None = None,
    ) -> ResolvedBayesianConfig:
        nonlocal counter
        counter += 1
        config_path = tmp_path / f"config_{counter}.json"
        config_path.write_text(
            json.dumps(copy.deepcopy(payload or valid_bayesian_config)),
            encoding="utf-8",
        )
        return load_and_validate_config(
            config_path,
            storage_dir=storage_dir or (tmp_path / "storage root"),
            repo_root=repo_root or tmp_path,
        )

    return resolve
