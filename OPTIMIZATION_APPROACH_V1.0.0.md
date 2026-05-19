# Optimization approach v1.0.0
This file contains the first idea for the first optimization of the system.

## Brief schema
This schema represents the actual flow of operations the system does AS OF NOW:
1. We might start with some code (not always pre-existing code). 
   - Benchmarks provide a workspace with starter files (sometimes nothing, sometimes partial code + data prep).
   - Agent often writes training code from scratch.
2. Task → difficulty, not dataset → complexity. task_difficulty.json maps task names (e.g. "cifar10") to levels. Same dataset could have multiple tasks at different difficulties.                                                                                                  
3. vLLM loads 2 models, routing is difficulty-based:
    - easy → qwen2.5-7b
    - medium/hard → llama-3.1-8b
    - User can override via --llm-name
4. After model pick → agent loop, not just "optimization". Agent iterates:
    - LLM generates code/action
    - Code executes in workspace
    - Observes output/errors
    - Repeats up to max_steps


## Task evaluation approach
This section explains the possible new approach macro idea, the flow is divided into 2 phases.

Terminology clarification: I'll refer to "lite" llm when talking about the model for the easy task, instead the "heavy" llm is the one for the difficult tasks.
Important notes about the actual llms in use: right now the models that are used are llama-3.1 8B (plain or quantized) and qwen 2.5 7B:
- both of them are very lightweight models, therefore we will take care to pick best ones.


### Architecture description
The system is designed to incorporate various agentic patterns.
Agent that will be used by our system:
- Orchestrator agent.
- Worker agent.
- Supervisor agent.
Memory / Shared state: whiteboard memory.

How they work and cooperate will be covered in phase 2.

### Phase 1 - active task difficulty evaluation
During this initial phase the task is evaluated.
Goal: determine the task's difficulty.
How: various methodologies are possible:
- by dataset evaluation: "easier"
- by task simulation and evaluation: this could be done by running a brief run of the original "agent loop" with the "lite" model.


### Phase 2 - boosted multi-agent loop
Once phase 1 has terminated and the difficulty is established, we'll proceed with the real multi-agent.
- The orchestrator is the first agent that receives the task:
  It decides how many worker agents to spawn: they are spawned according to the parallel pattern, each agent tries a different code/ action approach.
- Each worker agent follows the actual approach:
   1. LLM generates code/action
   2. Code executes in workspace
   3. Observes output/errors
   4. Repeats up to max_steps
- Every worker agent shares is state in the whiteboard memory that acts as a shared state.
  The worker shares the state: 1) when it has done writing the code 2) when it observes outputs errors (and so on for every loop)
  So we say the worker "writes" to shared state.
- The worker may eventually output the code if it is confident it's good - we'll cover output section later.
- The supervisor agent is the one that reads the shared state, it follows the supervisor pattern on the worker agents.
  It reads the shared state, if every agent is doing fine -> no action is taken.
  If the supervisor detects behaviors such as:
   1. The LLM of the worker agents is struggling (meaning repetitive thinking, always the same approach is taken, no improvement, unstable training, low training scores, no plateau, etc...) -> route to a more powerful model via vLLM.
      In this case the supervisor will communicate to the orchestrator to stop spawning workers with the current model and to start spawning workers with a heavier model (all of this of course via vLLM routing)
      The orchestrator will wait for a threshold time for the active workers to finish, and then it will shut them down, proceed with another batch of boosted workers.
   2. Performance degradation: like if the shared state highlights that the scores of the trained code are low or getting worse, same rationale as above.

    
#### Output section
We have to take the best output of the ones we got by all the agents, therefore the supervisor / orchestrator will be asked to score and retrieve the top-1 result