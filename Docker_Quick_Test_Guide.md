# Docker Quick Test Guide (Updated)

This guide provides the corrected, step-by-step instructions to run a quick test using the `house-price` task (a simpler dataset) inside your Docker container via WSL.

## Step 1: Clean Up Previous Containers (If Needed)
If you encountered a "Conflict" error or mounted the wrong directory previously, stop and remove the old container first. Run this in your normal WSL terminal:

```bash
docker rm -f mlagentbench-thesis-ctr
```

## Step 2: Navigate to Project Directory
Before starting the container, you MUST be in the correct directory. Your WSL terminal likely defaults to your Linux home directory (`~/`), which caused the previous mount issue.

In your WSL terminal, navigate to your mounted Windows project folder:
```bash
cd /mnt/c/Users/LENOVO/CC2026
```
*(Verify you are in the right folder by running `ls`, you should see files like `README.md` and the `MLAgentBench` folder).*

## Step 3: Run the Docker Container
Launch the container with GPU support from inside your project folder. Run this in your WSL terminal:

```bash
docker run --gpus all --user root -w /MLAgentBench \
  --name mlagentbench-thesis-ctr \
  -p 8001:8000 -p 8002:8002 \
  -v ${PWD}:/MLAgentBench -it cc2026-app
```

## Step 4: Start the Inference Server (Terminal 1)
The previous command will drop you into the interactive shell of the container. Start the `vLLM` server to serve the LLM model. 

*(Note: We are using `meta-llama/Llama-3.2-1B-Instruct` because it fits in limited VRAM constraints. Because LLaMA is a gated model, you must log in to Hugging Face first.)*

```bash
# 1. Activate the vLLM conda environment
conda activate vllm_srv

# 2. Login to Hugging Face (You will need an Access Token from huggingface.co/settings/tokens)
huggingface-cli login

# 3. Start the vLLM server
# (Fine-tuned memory limits to fit within ~4.9GB free VRAM with eager mode to save overhead)
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --served-model-name llama \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4096 \
  --enforce-eager \
  --port 8002 \
  --tensor-parallel-size 1
```


## Step 5: Run the Agent Task (Terminal 2)
Leave Terminal 1 running. Open a **Brand New WSL Terminal Window** (Do not run this inside the container!) and connect to the running container:

```bash
# Connect to the running container from your host WSL
docker exec -it mlagentbench-thesis-ctr /bin/bash
```

Once inside this second terminal session (your prompt should look like `root@...:/MLAgentBench#`), prepare and execute the task. 

*(Note: We are using the `cifar10` task. We fetched the actual `train.py` from the official repository so the agent has real code to modify.)*:

```bash
# 1. Prepare the environment
python -m MLAgentBench.runner --task cifar10 --llm-name llama --fast-llm-name llama --edit-script-llm-name llama --device 0
```