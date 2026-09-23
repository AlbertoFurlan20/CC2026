# Running Bayesian optimization on NVIDIA Brev

This deployment runs two services by default:

- `vllm` serves the configured model on the internal URL `http://vllm:8002/v1`.
- `optimizer` waits for the vLLM health check and then runs
  `scripts/run_bayesian.py` with the selected Bayesian configuration.

An optional `bench` service provides an interactive container for diagnostics.
The Brev Compose file is standalone; do not combine it with the repository's
existing `docker-compose.yml`.

Two host-side environment presets are included:

- `.env.brev.example` is the conservative shared-GPU/smoke-test preset.
- `.env.brev.dedicated.example` is tuned for a 2x L40S instance. It assigns
  vLLM and benchmark training to different GPUs and uses
  `0.85 / 32 / 16384` serving limits.

The dedicated preset increases serving capacity, not model intelligence. Both
presets serve the same Llama 3.1 8B model with an 8192-token context window.

The configuration follows NVIDIA's documented Compose GPU reservation format.
Brev instances include Docker and the NVIDIA Container Toolkit. See the
[NVIDIA Brev container guide](https://docs.nvidia.com/brev/guides/development-tools/custom-containers).

## 1. Prepare the instance

Choose a GPU that has enough VRAM for both the model and the benchmark workload,
or choose two GPUs and isolate the services as described below. After connecting
to the instance, keep the checkout beneath Brev's persistent workspace:

```bash
cd /home/ubuntu/workspace
git clone <repository-url> CC2026
cd CC2026

nvidia-smi
docker compose version
```

Create the persistent data directories. The Compose stack mounts this entire
directory at `/data`, so Optuna storage, exported results, logs, workspaces, and
the Hugging Face model cache survive container recreation:

```bash
mkdir -p /home/ubuntu/workspace/cc2026-data/{results,logs,workspace,hf-cache}
mkdir -p /home/ubuntu/workspace/.secrets
```

Brev preserves `/home/ubuntu/workspace` when an instance is stopped, but removes
it when the instance is deleted. Keep a second copy of important results. See
[Brev's data-persistence table](https://docs.nvidia.com/brev/concepts/gpu-instances#data-persistence).

## 2. Configure secrets and deployment settings

Choose one preset and store the resulting environment file outside the Git
checkout. For a shared GPU or the first smoke test:

```bash
install -m 600 .env.brev.example /home/ubuntu/workspace/.secrets/cc2026-brev.env
```

For a 2x L40S instance with a dedicated inference GPU and a separate benchmark
GPU:

```bash
install -m 600 .env.brev.dedicated.example /home/ubuntu/workspace/.secrets/cc2026-brev.env
```

Then create the token file and review the selected settings:

```bash
install -m 600 /dev/null /home/ubuntu/workspace/.secrets/huggingface-token
nano /home/ubuntu/workspace/.secrets/huggingface-token
nano /home/ubuntu/workspace/.secrets/cc2026-brev.env
```

Put only the Hugging Face token in `huggingface-token`, with no `HF_TOKEN=`
prefix. The token file is mounted as a Compose secret; the token is not embedded
in the image, Compose file, or container environment. `huggingface_hub` reads it
through `HF_TOKEN_PATH`, as documented in its
[environment-variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hf-token-path).
The account must have access to the configured gated model.

Verify that the token file exists, is non-empty, and is readable before building:

```bash
test -r /home/ubuntu/workspace/.secrets/huggingface-token
test -s /home/ubuntu/workspace/.secrets/huggingface-token
stat -c '%a %U:%G %n' /home/ubuntu/workspace/.secrets/huggingface-token
```

For the commands below, set a convenience variable:

```bash
export CC2026_BREV_ENV=/home/ubuntu/workspace/.secrets/cc2026-brev.env
```

The most important environment settings are:

| Variable | Purpose |
| --- | --- |
| `BREV_STORAGE_ROOT` | Persistent host directory mounted at `/data` |
| `BAYES_CONFIG` | Bayesian JSON config passed to `run_bayesian.py` |
| `VLLM_MODEL_ID` | Hugging Face model ID loaded by vLLM |
| `VLLM_SERVED_MODEL_NAME` | Model name expected by the benchmark config |
| `VLLM_GPU_ID` | Host GPU assigned to vLLM |
| `OPTIMIZER_GPU_ID` | Host GPU assigned to benchmark runs |
| `VLLM_GPU_MEMORY_UTILIZATION` | Fraction of the vLLM GPU reserved by vLLM |
| `VLLM_MAX_MODEL_LEN` | Maximum context length for one request |
| `VLLM_MAX_NUM_SEQS` | Maximum number of active sequences in a scheduler iteration |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | Maximum tokens processed in one scheduler iteration |
| `CONTAINER_SHM_SIZE` | Shared-memory size available inside each container |
| `OPTIMIZER_RESTART_POLICY` | Compose restart policy for the controller |

`VLLM_SERVED_MODEL_NAME` must match the logical model name in the Bayesian
configuration. The default values match the repository's Llama 3.1 8B setup.

### GPU layouts and serving limits

| Preset | vLLM GPU | Optimizer GPU | VRAM fraction | Context | Sequences | Batched tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Shared/smoke | 0 | 0 | 0.60 | 8192 | 8 | 8192 |
| Dedicated 2x L40S | 0 | 1 | 0.85 | 8192 | 32 | 16384 |

For the intended Brev deployment, prefer 2x L40S. Each L40S has 48 GB of GPU
memory, so the 0.85 vLLM limit gives the inference process roughly 40.8 GB while
leaving the second GPU entirely available to benchmark training. The L40S does
not provide NVLink; Llama 3.1 8B fits on one card, so tensor parallelism remains
one and there is no reason to split this model across both cards. See NVIDIA's
[L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/).

On a single sufficiently large-memory GPU, both services use device `0`:

```dotenv
VLLM_GPU_ID=0
OPTIMIZER_GPU_ID=0
VLLM_GPU_MEMORY_UTILIZATION=0.60
```

Start conservatively because vLLM and benchmark training share VRAM. The
unquantized 8B model may not fit inside a 60% reservation on a 24 GB card; in
that case use separate GPUs, a larger inference GPU, or a quantized checkpoint.
Do not copy the dedicated preset's 0.85 reservation onto a shared GPU.

On a 2x L40S instance, isolate inference and training:

```dotenv
VLLM_GPU_ID=0
OPTIMIZER_GPU_ID=1
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=8192
VLLM_MAX_NUM_SEQS=32
VLLM_MAX_NUM_BATCHED_TOKENS=16384
```

`max-num-seqs=32` and `max-num-batched-tokens=16384` are concurrency ceilings;
they do not make an individual model response smarter. The current Bayesian
controller submits trials sequentially, so the larger values mainly provide
headroom for future concurrent clients. If vLLM reports preemption or CUDA
out-of-memory, lower the batched-token budget to 8192. If you later add many
concurrent clients, load-test 32768 rather than assuming it is faster:

```dotenv
VLLM_MAX_NUM_BATCHED_TOKENS=32768
```

The Compose command enables prefix caching for repeated agent prefixes and
chunked prefill for fairer scheduling of long prompts alongside decode work.
These controls and their tradeoffs are described in the pinned
[vLLM 0.13 optimization guide](https://docs.vllm.ai/en/v0.13.0/configuration/optimization/)
and [serve-argument reference](https://docs.vllm.ai/en/v0.13.0/cli/serve/).

This Compose profile intentionally assigns one GPU to each service and fixes
vLLM tensor parallelism to one. A model requiring multi-GPU tensor parallelism
needs a separate Compose variant with multiple reserved inference devices and
another GPU for benchmark training.

## 3. Validate and build

Render and validate the configuration before using GPU time:

```bash
docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  config --quiet

docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  build
```

The image build installs both the benchmark environment and the separate vLLM
environment. It can take a while on the first run.

## 4. Start vLLM and verify it

Start only the inference service first:

```bash
docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  up -d vllm

docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  logs -f vllm
```

The first start downloads the model into
`/home/ubuntu/workspace/cc2026-data/hf-cache`. In another terminal, wait for the
service to become healthy and query it through the loopback-only host port:

```bash
docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  ps

curl --fail http://127.0.0.1:8002/v1/models
```

If vLLM becomes unhealthy, inspect its logs before starting the optimizer.
Authentication errors usually mean the token is invalid or the account lacks
access to the configured model; CUDA out-of-memory errors require a smaller
memory utilization, shorter model context, or a larger/separate GPU.

## 5. Run the Bayesian study

Start the optimizer in the foreground for the first run:

```bash
docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  up --abort-on-container-exit --exit-code-from optimizer optimizer
```

This exits with the optimizer's status and stops the attached vLLM service when
the study finishes, so the foreground command does not remain attached to an
idle inference container.

For an unattended run, use detached mode and follow the logs:

```bash
docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  up -d optimizer

docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  logs -f optimizer
```

The optimizer uses `STORAGE_DIR=/data`. Its SQLite study and trial exports are
therefore on the Brev host beneath `BREV_STORAGE_ROOT`. The container has an
`on-failure` restart policy; if a transient process failure restarts it, the
Bayesian runner reopens the same persistent study rather than starting over.
Compose gates optimizer startup on vLLM health, but does not pause a running
study if vLLM later becomes unavailable. Keep the vLLM logs visible during the
first full run; stop the optimizer if the inference service enters a persistent
restart loop, fix vLLM, and resume with the same command. Infrastructure
failures are recorded as failed trials rather than fabricated objective scores.

The sample Bayesian config continues after a failed trial, so the default
`OPTIMIZER_RESTART_POLICY=on-failure:3` is intended for controller/container
crashes. If you set `execution.continue_on_trial_failure=false` to require a
strict stop at the first infrastructure failure, also set this in the Brev env
file so Compose does not immediately resume the next trial:

```dotenv
OPTIMIZER_RESTART_POLICY=no
```

To resume manually after stopping the stack or rebooting the instance, run the
same detached `up -d optimizer` command. Do not change the study's fixed
arguments, search space, objective direction, or failure policy when resuming.
Use a new experiment name for an incompatible configuration.

Before stopping or deleting the Brev instance, verify the host artifacts:

```bash
find /home/ubuntu/workspace/cc2026-data/results -maxdepth 3 -type f -print
```

## 6. Optional diagnostic shell

Start an ephemeral debug container with the same code, persistent storage, vLLM
endpoint, and optimizer GPU:

```bash
docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  --profile debug \
  run --rm bench bash
```

Inside it, use the benchmark environment explicitly when needed:

```bash
conda run --no-capture-output -n autogpt python -V
python -c "import urllib.request; print(urllib.request.urlopen('http://vllm:8002/v1/models').status)"
```

## 7. Stop services

Stop containers without removing persistent host data:

```bash
docker compose \
  --env-file "$CC2026_BREV_ENV" \
  -f docker-compose.brev.yml \
  down
```

The bind-mounted data remains under `BREV_STORAGE_ROOT`. Deleting the Brev
instance still deletes that host storage, so archive trial results elsewhere
before deletion.
