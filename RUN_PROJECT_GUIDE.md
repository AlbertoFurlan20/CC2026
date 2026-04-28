# Project Run Guide (New Branch)

This guide provides step-by-step instructions to run the MLAgentBench framework on your current branch. It covers building the Docker container, starting the local inference server (vLLM), and running an autonomous agent task.

## Prerequisites

Ensure you have the following ready before starting:
- **Docker** installed with NVIDIA GPU support (`nvidia-container-toolkit`).
- **NVIDIA GPU** with sufficient VRAM to run the chosen model.
- **Hugging Face Token**: Required if you plan to use gated models like Meta's Llama 3.1.
- **Kaggle API Key**: Ensure your `kaggle.json` is placed in a `.kaggle/` directory so the agent can download task datasets.

---

## 1. Start the Docker Container (via WSL)

Open your WSL terminal and **first navigate to the project directory** on your Windows host:
```bash
cd /mnt/c/Users/LENOVO/CC2026
```

The project runs entirely inside a pre-configured Docker container. Because the run command mounts your current directory (`-v ${PWD}:/MLAgentBench`), **any code changes you make or branches you switch to on your host machine are instantly reflected inside the container.** 

*Note: If this is your first time running on a new machine, or if you changed dependencies like `requirements.txt` or `Dockerfile`, you must build the image first:*
```bash
# docker build -t mlagentbench-thesis .
```

**Run the container:**
Launch the container with GPU support and volume mounting (ensure you are still in `/mnt/c/Users/LENOVO/CC2026`):

*(Note: If you get a "Conflict" error saying `The container name "/mlagentbench-thesis-ctr" is already in use`, run `docker rm -f mlagentbench-thesis-ctr` first to remove the old container).*

```bash
docker rm -f mlagentbench-thesis-ctr
```

```bash
docker run --gpus all --user root -w /MLAgentBench \
  --name mlagentbench-thesis-ctr \
  -p 8001:8000 -p 8002:8002 \
  -v ${PWD}:/MLAgentBench -it mlagentbench-thesis
```

---

## 2. Start the Inference Server (Terminal 1)

Once inside the container's interactive shell, you need to spin up the local LLM server using `vLLM`. 

**Important - Hugging Face Authentication:**
Models like `meta-llama/Llama-3.2-3B-Instruct` are gated limits. Before starting the server:
1. Ensure you have accepted Meta's terms on the model's Hugging Face page.
2. Create an access token in your HF Settings.
3. Run `huggingface-cli login` in the terminal and paste your token.
*(Alternatively, you can switch the `--model` to an open, non-gated model like `Qwen/Qwen2.5-3B-Instruct-AWQ` to skip authentication completely).*

```bash
# 1. Activate the dedicated vLLM conda environment
conda activate vllm_srv

# 2. Start the vLLM server
# To avoid Out-of-Memory (OOM) errors on GPUs with limited VRAM (like 6GB), 
# use a quantized 3B model (e.g., AWQ INT4) and enforce strict memory limits.
python -m vllm.entrypoints.openai.api_server \
  --model casperhansen/llama-3.2-3b-instruct-awq \
  --served-model-name llama-3.2-3b-instruct-awq \
  --port 8002 \
  --tensor-parallel-size 1 \
  --quantization awq \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 \
  --enforce-eager
```
*Leave this terminal running. It hosts the LLM that the agent will communicate with.*

---

## 3. Run the Agent Task (Terminal 2)

Open a **Brand New Terminal Window** on your host machine, and connect to the running container:

```bash
# Connect to the running container from your host
docker exec -it mlagentbench-thesis-ctr /bin/bash
```

Inside this new terminal session, prepare and run your desired benchmark task (e.g., `cifar10`):

```bash
# 1. Prepare the task environment
python -m MLAgentBench.prepare_task cifar10 $(which python)

# 2. Run the agent
# Make sure to override the fast and edit LLMs to use your local model instead of Claude APIs
python -m MLAgentBench.runner \
  --task vectorization \
  --llm-name llama-3.2-3b-instruct-awq \
  --fast-llm-name llama-3.2-3b-instruct-awq \
  --edit-script-llm-name llama-3.2-3b-instruct-awq \
  --device 0
```

### Important Notes:
- **Prompt Strategies**: This branch incorporates "Rigid Prompts" designed to minimize hallucinations and API loops for local 3B models. These are used by default to increase success rates.
- **Energy Footprint**: The framework integrates CodeCarbon to track energy usage (GPU, CPU, RAM) heavily tied to the `Execute Script` and `Edit Script` actions.
- **Troubleshooting VRAM**: If you encounter Out-Of-Memory (OOM) errors on the API server, consider using INT4 AWQ quantized models and adjusting `--gpu-memory-utilization` and `--max-model-len` parameters when starting vLLM.