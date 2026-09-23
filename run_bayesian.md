# Run Bayesian optimization on CIFAR-10

This guide shows how to run the existing CIFAR-10 experiment on an NVIDIA GPU
server. It uses Docker Compose, so the setup is the same each time.

## What the experiment does

The Llama 3.1 8B agent tries to improve a CIFAR-10 training program. Optuna
searches for good values of:

| Parameter | Range |
| --- | --- |
| `temperature` | 0.2–0.7 |
| `top_p` | 0.7–1.0 |

The model, dataset, and agent settings do not change between trials.

The study has 18 attempts:

- The first 6 trials collect starting results.
- The next 12 trials use TPE to choose promising values.

Each trial does this:

```text
Choose temperature and top_p
              ↓
Run the CIFAR-10 agent
              ↓
Evaluate the final submission
              ↓
Save the score
              ↓
Use the results to choose the next values
```

Trials run one at a time. A failed trial still counts as an attempt.

## What the server needs

- Ubuntu or another Linux distribution
- An NVIDIA GPU, preferably an L40S with 48 GB VRAM
- Docker with Docker Compose
- NVIDIA Container Toolkit
- Persistent storage
- A Hugging Face read token with access to Llama 3.1 8B

Check the server:

```bash
nvidia-smi
docker compose version
```

## 1. Get the project

On Brev:

```bash
cd /home/ubuntu/workspace
git clone https://github.com/AlbertoFurlan20/CC2026.git CC2026
cd CC2026
git switch dev/bayesian
git pull --ff-only
```

To use the exact project version from this guide:

```bash
git checkout 76ee598801cdd0805f9def799e89333f5c71b423
```

Skip the clone if the project is already on the server.

## 2. Create the storage folders

```bash
mkdir -p /home/ubuntu/workspace/cc2026-data/{results,logs,workspace,hf-cache}
mkdir -p /home/ubuntu/workspace/.secrets
```

Results, logs, trial workspaces, and the downloaded model will remain inside
`/home/ubuntu/workspace/cc2026-data`.

For a server other than Brev, replace `/home/ubuntu/workspace` with a persistent
directory that you can write to.

## 3. Add the Hugging Face token

```bash
install -m 600 /dev/null /home/ubuntu/workspace/.secrets/huggingface-token
nano /home/ubuntu/workspace/.secrets/huggingface-token
```

Paste only the token. Do not write `HF_TOKEN=` before it. Save and close the
file.

Check that the file is not empty:

```bash
test -s /home/ubuntu/workspace/.secrets/huggingface-token && echo "Token is ready"
```

Never commit or share the token.

## 4. Choose the GPU setup

For one L40S, use:

```bash
install -m 600 .env.brev.example /home/ubuntu/workspace/.secrets/cc2026-bayes.env
```

This makes vLLM and CIFAR-10 training share GPU 0. The preset limits vLLM to
60% of the GPU memory so training has room.

For a server with two GPUs, use this instead:

```bash
install -m 600 .env.brev.dedicated.example /home/ubuntu/workspace/.secrets/cc2026-bayes.env
```

The two-GPU preset uses GPU 0 for vLLM and GPU 1 for CIFAR-10 training.

## 5. Prepare the Docker command

Run this from the `CC2026` directory:

```bash
export CC2026_BAYES_ENV=/home/ubuntu/workspace/.secrets/cc2026-bayes.env

bayes() {
  docker compose \
    --env-file "$CC2026_BAYES_ENV" \
    -f docker-compose.brev.yml "$@"
}
```

This creates a short `bayes` command for the current terminal. Run this section
again after reconnecting with SSH.

## 6. Build the project

```bash
bayes config --quiet
bayes build
```

The first build may take a while. It does not start the experiment.

Optionally validate the CIFAR-10 configuration:

```bash
bayes run --rm --no-deps optimizer \
  conda run --no-capture-output -n autogpt \
  python scripts/run_bayesian.py configs/comparison_bayes.json --validate-only
```

This only checks the configuration; it does not create a trial.

## 7. Start vLLM

```bash
bayes up -d vllm
bayes logs -f --tail=100 vllm
```

The first start downloads Llama 3.1 8B. Wait until vLLM is healthy. Press
`Ctrl-C` to leave the logs; the container keeps running.

Check the service:

```bash
bayes ps
curl --fail http://127.0.0.1:8002/v1/models
```

Continue when the `curl` command returns the model information.

## 8. Start the experiment

```bash
bayes up -d optimizer
```

The experiment now runs in the background. You can disconnect from SSH and
turn off your laptop. The GPU server continues running.

The 18 trials usually take several hours. The exact duration depends on the
training programs created by the agent.

## 9. Watch the experiment

```bash
bayes logs -f --tail=100 optimizer
```

The log will show messages such as:

```text
Trial 0 finished with value: ...
Trial 1 finished with value: ...
Best is trial ...
```

Press `Ctrl-C` to stop watching. The experiment continues.

Other useful commands:

```bash
bayes ps -a
nvidia-smi
```

See the latest saved trials:

```bash
tail -n 10 \
  /home/ubuntu/workspace/cc2026-data/results/cifar10_bayesian_sampling/trials.csv
```

## 10. Check the final result

The experiment is complete when all 18 attempts are finished and `optimizer`
exits with code 0.

```bash
bayes ps -a
```

Show the best result:

```bash
python3 -m json.tool \
  /home/ubuntu/workspace/cc2026-data/results/cifar10_bayesian_sampling/best.json
```

The result folder contains:

| File | Meaning |
| --- | --- |
| `best.json` | Best score, `temperature`, and `top_p` |
| `trials.csv` | All Optuna trials |
| `runs.csv` | Details for every benchmark run |
| `study.sqlite3` | Optuna study database |
| `sampler.pkl` | Saved TPE state |
| `manifest.json` | Study status and trial counts |

Full path:

```text
/home/ubuntu/workspace/cc2026-data/results/cifar10_bayesian_sampling/
```

Detailed agent logs are in `/home/ubuntu/workspace/cc2026-data/logs/`.

## Stop without deleting the results

```bash
bayes down
```

This stops the containers. The files in `cc2026-data` remain. Do not use
`bayes down -v`.

## Resume an interrupted study

Reconnect to the server, enter the project, and recreate the short command:

```bash
cd /home/ubuntu/workspace/CC2026
export CC2026_BAYES_ENV=/home/ubuntu/workspace/.secrets/cc2026-bayes.env

bayes() {
  docker compose \
    --env-file "$CC2026_BAYES_ENV" \
    -f docker-compose.brev.yml "$@"
}
```

Then run:

```bash
bayes up -d optimizer
bayes logs -f --tail=100 optimizer
```

Optuna loads the existing database and runs only the remaining attempts. Keep
the same config and Git revision when resuming.

## Start a completely new study

Running the same command after completion does not create another 18 trials. It
opens the completed study and finds that no attempts remain.

To start a separate study:

1. Copy `configs/comparison_bayes.json`.
2. Change `experiment.name` in the copy.
3. Change `BAYES_CONFIG` in `cc2026-bayes.env` to the copied config.

For example, change:

```json
"name": "cifar10_bayesian_sampling"
```

to:

```json
"name": "cifar10_bayesian_sampling_02"
```

This keeps the original results and creates a new result folder.

## Before deleting the server

Download the result folder to your local computer:

```bash
rsync -avz \
  ubuntu@<SERVER_HOST>:/home/ubuntu/workspace/cc2026-data/results/cifar10_bayesian_sampling/ \
  ./cifar10_bayesian_sampling/
```

Stopping a Brev instance keeps its workspace. Deleting the instance removes
it, so download the results and any logs you need first.

## If something fails

Check both container logs:

```bash
bayes ps -a
bayes logs --tail=200 vllm optimizer
```

- An authentication error usually means the Hugging Face token is expired or
  the account does not have model access.
- A CUDA out-of-memory error on one L40S usually means vLLM is using too much
  memory. Keep `VLLM_GPU_MEMORY_UTILIZATION=0.60` in the one-GPU preset.
- The optimizer waits while vLLM is unhealthy. Fix vLLM first.
