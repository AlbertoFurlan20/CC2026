git# Overview of the `CC2026` Directory

The `CC2026` folder contains a framework developed for a Master's Thesis titled *"Implementation and benchmarking of local LLM agents: evaluation of functional performance and computational efficiency"* at Politecnico di Milano. 

This repository is an extension of Stanford's [MLAgentBench](https://github.com/snap-stanford/MLAgentBench/), but adapted to run autonomous AI agents in local computational environments rather than relying on external API calls. It primarily focuses on evaluating local, open-source Large Language Models (LLMs) like Llama 3.1-8B and Qwen-2.5-7B, tracking their energy consumption, and evaluating their responses under pressure.

Here is a step-by-step breakdown of everything happening in the `CC2026` folder and its core components:

## 1. Local LLM Environment Configuration
Instead of using cloud-based OpenAI models via APIs, this environment attempts to run local models efficiently:
* **`requirements_main.txt`, `requirements_vllm_srv.txt`, `requirements.txt`**: These files declare the python package and dependency structures required for setting up different components. Specifically, `requirements_vllm_srv.txt` sets up the `vLLM` inference engine environment, which is highly optimized for local LLM inference and serving.

## 2. Containerized Deployment (`Dockerfile`s)
* **`Dockerfile`, `Dockerfile.extra`, `Dockerfile_original_old`**: To prevent dependency hell and configuration mismatch (given the complexity of running local inference and ML benchmarking tools), the project uses Docker. It builds an environment (which ends up around 45GB) pre-equipped with Anaconda, NVIDIA graphical optimizations, CodeCarbon, and vLLM. 

## 3. Stress Testing and Variability Analysis (`JMETER/`)
* **The `JMETER` directory**: This folder contains numerous `.jmx` (Apache JMeter plan files) and `.jtl` (test result log files). JMeter triggers simulated loads against the running inference server across various data tasks (e.g., *babylm, cifar10, fathomnet, imdb, parkinson*). This stress testing evaluates how the chosen Language Models queue requests, behave, and whether their API consistency drops when running concurrent loads.

## 4. The Agent Benchmarking Logic (`MLAgentBench/`)
* This is the core logic ported and extended from Stanford's original codebase.
* **`runner.py` & `prepare_task.py`**: These scripts are used respectively to bootstrap an isolated test environment for an AI agent on a specific dataset, and then to trigger the agent to solve tasks.
* **`LLM.py`, `environment.py`, `low_level_actions.py`**: These describe how the autonomous agent operates. It executes Python code, queries data, checks logs, and writes software.
* **Prompt Engineering adjustments**: The thesis implemented a "Rigid" prompting structure. The agent isn't just taking standard AI prompts, but heavily structured instructions aimed at decreasing hallucinations for smaller local-friendly models.
* **Green AI & Plotting Scripts (`plot_codecarbon.py`, `plot.py`, `plot_accuracy_time.py`)**: Integrated throughout this benchmark is `CodeCarbon`, which dynamically measures how much RAM, CPU, and GPU power is heavily consumed by the agents. Operations like "Edit Script" and "Execute Script" are measured and translated into CO2 emissions data.

## 5. Execution Scripts (`scripts/`)
* **Shell Scripts**: The `/scripts/` folder (`run_experiments.sh`, `eval.sh`, `multi_run_experiment.sh`) is meant to automate the long-running execution of benchmark scenarios. They sequentially loop through tasks (like training a cifar10 model), spin up the agent, record its runtime parameters, run tests, and output evaluation logs.

## 6. The `agents/` Module inside MLAgentBench
Within the core framework (`MLAgentBench/agents/`), there are several different classes of AI agents that can be instantiated to solve the machine learning tasks. This structure provides a modular way to compare different agent architectures on the exact same benchmarks:
* **`agent.py`**: The foundational file that defines the base `Agent` class interface. Other specific agent architectures extend this base class to inherit environment hooks and action states.
* **`agent_research.py`**: The custom "AI Research Assistant" designed closely for this specific thesis and the original MLAgentBench. It injects the "Rigid" prompts, establishes tool access paths, and integrates directly with `CodeCarbon` to track realtime energy and emission statistics during operations.
* **`agent_langchain.py`**: A wrapper file that implements models driven by the standard `LangChain` architecture (such as ReAct agents). It translates LangChain's internal actions to the standard protocol MLAgentBench requires.
* **`agent_autogpt.py` & `Auto-GPT/` directory**: A wrapper and accompanying source repository for running the open-source Auto-GPT algorithm within the MLAgentBench ecosystem, providing another baseline to compare task completion capabilities.

## In Summary
When you work in `CC2026`:
1. You are running a local backend Server (`vLLM`) that acts as the "Ghost API" serving your downloaded Llama/Qwen models over local ports.
2. The core framework (`MLAgentBench`) runs autonomous sub-agents that attempt to complete complex machine learning tasks contextually.
3. Tests evaluate these tasks for energy consumption footprint (CodeCarbon) and consistency using JMeter frameworks as the system evaluates prompt constraints under heavy load.