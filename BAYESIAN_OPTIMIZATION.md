# Bayesian optimization

The comparison entry point now supports both the existing grid search and a
persistent Optuna TPE study. The dispatcher reads `search.method`:

```text
scripts/compare_strategies.sh
├── grid      -> scripts/emit_grid_combos.py + existing shell loop
└── bayesian  -> scripts/run_bayesian.py -> MLAgentBench runner -> evaluator
```

The Bayesian path is sequential (`n_jobs=1`) because every trial owns a GPU
benchmark workspace. It searches only parameters that affect the current vLLM
agent path: `top_p` and `temperature`. `best_of` and `n_samples` are excluded
because the current agent consumes only the first returned choice and some vLLM
chat-server versions ignore `best_of`.

## Run and validate

Install the updated requirements, then validate the config without launching a
benchmark:

```bash
python scripts/run_bayesian.py configs/comparison_bayes.json --validate-only
```

Run through the shared dispatcher:

```bash
bash scripts/compare_strategies.sh configs/comparison_bayes.json
```

Or call the controller directly and choose a persistent storage root:

```bash
python scripts/run_bayesian.py \
  configs/comparison_bayes.json \
  --storage-dir /data
```

The sample configuration performs 18 trial attempts: six TPE startup trials
plus twelve subsequent TPE trials. Its enqueued point is evaluated first. Set
`execution.runs_per_trial` above one to repeat a parameter set and aggregate its
scores with the configured mean or median.

## Objective and failures

Only the benchmark's final score is optimized. Intermediate evaluator scores
are never substituted for a missing final result, and a `NopPruner` ensures an
expensive benchmark run is not stopped based on incomplete evidence. The final
aggregate is also recorded through Optuna's `trial.report` API for inspection.

Failure handling separates scientific outcomes from infrastructure problems:

- A completed agent run with no valid final submission follows
  `objective.behavioral_failure`: the sample assigns a score of `0.0`.
- Runner, evaluator, timeout, malformed-artifact, and evaluator-exception
  failures make the Optuna trial `FAIL`; they are never converted into scores.
- `execution.continue_on_trial_failure` controls whether one failed trial stops
  the study or the optimizer proceeds to the next attempt.

## Persistence and artifacts

For experiment `cifar10_bayesian_sampling`, artifacts are written below
`<storage>/results/cifar10_bayesian_sampling/`:

| Artifact | Purpose |
| --- | --- |
| `study.sqlite3` | Authoritative Optuna study and trial states |
| `sampler.pkl` | TPE RNG/model state needed for deterministic resume |
| `config.resolved.json` | Validated config, defaults, paths, and fingerprint |
| `manifest.json` | Run status, trial counts, versions, and Git revision |
| `trials.csv` | One row per Optuna trial |
| `runs.csv` | One row per benchmark repetition, including failures/energy |
| `best.json` | Best completed trial and fixed/suggested parameters |
| `study.lock` | Prevents two local controllers using one study concurrently |

Per-run logs and workspaces use deterministic, non-overwriting names such as
`cifar10_bayesian_sampling_trial_000003_run_01` under `<storage>/logs` and
`<storage>/workspace`.

With `persistence.resume=true`, rerunning the same command restores SQLite and
the sampler and executes only the missing attempts. The immutable settings are
fingerprinted: changing fixed runner arguments, search distributions, sampler,
objective, repetitions, subprocess timeout, or tracking is rejected. The
MLAgentBench/scripts Git revision (including relevant dirty source files) is
also checked. You may increase `n_iter` to extend a finished study; lowering the
total budget below its existing terminal trial count is rejected. Use a new
experiment name for a different study.

For an NVIDIA Brev deployment, see [BREV_BAYESIAN_GUIDE.md](BREV_BAYESIAN_GUIDE.md).
