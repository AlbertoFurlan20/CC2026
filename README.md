# CC2026

This repository contains the framework developed for the Master’s Thesis: "Implementation and benchmarking of local LLM agents: evaluation of functional performance and computational efficiency" at Politecnico di Milano.

The project extends the MLAgentBench framework: (https://github.com/snap-stanford/MLAgentBench/) to enable autonomous agents in local environments, focusing on open-source models, energy sustainability, and performance optimization for smaller LLMs.

## Recomendations
By performing the clone of the repository, be aware that it already contains all the results obtained during the project analysis, saved on all folders named (user1_taskname_model, userN_taskname_model, mis_multi_plots, mis_multi_plotsN) following the nomenclature of the tests that were carried out by the original (normal prompt) with the names of "userN" while the other tests by the name of "user1" were made by using the edited (rigid prompt) ones, all this folders of the resutls are 10 GB in size.

For a complete guide step by step please read the "Guide -MLAgentBench user manual.pdf" file

## Key Features
Local Model Support: Seamless integration with Llama 3.1-8B and Qwen-2.5-7B using the vLLM inference engine.

Green AI Monitoring: Integrated CodeCarbon to track real-time energy consumption (GPU, CPU, RAM) and CO2 emissions per agent action.

Rigid Prompting Strategy: Implementation of a "Rigid" structured prompt format that significantly reduces hallucinations and loops, increasing success rates for 7B/8B models.

Variability Analysis: Stress testing using Apache JMeter to evaluate system stability and response consistency under concurrent loads.

Containerized Environment: A fully pre-configured Docker environment (~45GB) with all dependencies, Conda environments, and tools pre-installed.

## Installation & Setup
1. Prerequisites
  Docker installed on a machine with NVIDIA GPU support (nvidia-container-toolkit).

  Kaggle API Key: Required for downloading task datasets. Place your kaggle.json in the .kaggle/ directory.

  Hugging Face Token: For accessing gated models like Llama 3.1.

1. First things first, setup the tests you want to run by changing the entries in the [tasks file](MLAgentBench/benchmarks/tasks.json)
```json
{
    "cifar10": {
        "research_problem": "Given a training script on a dataset train.py, improve upon the current performance of the model with a simple change.",
        "benchmark_folder_name": "cifar10"
    },
    "vector": {
        "research_problem": "Given a training script on a dataset train.py, improve upon the current performance of the model with a simple change.",
        "benchmark_folder_name": "vector"
    },
  ...
}
```

2. Docker Deployment
### Build and run
```bash
docker build -t mlagentbench-thesis .
docker run --gpus all --user root -w /MLAgentBench \
  --name mlagentbench-thesis-ctr \
  -p 8001:8000 -p 8002:8002 \
  -v ${PWD}:/MLAgentBench -it mlagentbench-thesis
```

## Quick Start Guide
(_From within the container_)
```bash
docker exec -it mlagentbench-thesis-ctr /bin/bash
```

Step 1: Start the Inference Server (vLLM) - Quantized llama version
```bash
conda activate vllm_srv
python -m vllm.entrypoints.openai.api_server \
  --model RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 \
  --served-model-name llama-3.1-8B-Instruct \
  --port 8002 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048
```

Brief note: vLLM performs routing to 2+ models, so you shall serve 2 models, as below

Complete setup:
```bash
## Terminal 1 — Llama on 8002                                                                                                             
python -m vllm.entrypoints.openai.api_server \                                                                                             
--model RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a16 \                                                                            
--port 8002 --tensor-parallel-size 1 --gpu-memory-utilization 0.85                                                                       
                                                                                                                                             
## Terminal 2 — Qwen on 8003                                                                                                                
python -m vllm.entrypoints.openai.api_server \                                                                                             
--model Qwen/Qwen2.5-7B-Instruct \                                                                                                       
--port 8003 --tensor-parallel-size 1 --gpu-memory-utilization 0.85    
```

Step 2: Run an Agent Task
In another terminal (inside the container), prepare and run a benchmark task (e.g., cifar10):

### 1. Prepare the task environment
```bash
python -m MLAgentBench.prepare_task cifar10 $(which python)
```

Note: if you get already prepared, you can `rm -rf /MLAgentBench/MLAgentBench/benchmarks/cifar10/scripts/prepared`.

### 2. Run the agent
```bash
python -m MLAgentBench.runner \
  --task cifar10 --llm-name llama-3.1-8B-Instruct --device 0
```

For the quantized version:
```bash
python -m MLAgentBench.runner \
  --task cifar10 \
  --llm-name llama-3.1-8B-Instruct \
  --edit-script-llm-name llama-3.1-8B-Instruct \
  --fast-llm-name llama-3.1-8B-Instruct \
  --device 0 \
  --use-codecarbon \
  --valid-format-entires Thought Action "Action Input"
```

## Notes on run
Halting conditions:

## Research Highlights

Prompt Engineering: Moving from standard prompts to Rigid Prompts improved Qwen-2.5-7B's success rate from 19.4% to 55.6%.

Energy Footprint: Research showed that 89% of the total energy consumption is concentrated in "Edit Script" and "Execute Script" actions.

Operational Efficiency: Local deployment offers a sustainable and private alternative to cloud-based LLM APIs for automated software engineering.
"\n Test" 
