# Interesting Multi-Agent patterns
- Supervisor pattern: \
  A supervisor agent oversees the work of multiple worker agents, providing guidance, support, and coordination to ensure that tasks are completed efficiently and effectively.\
  - It knows the goal, decides which agents to call and when.
  - It loops until the goal is achieved, and it can also adjust the plan if needed.
  - Might be useful for complex tasks that require coordination among multiple agents, such as project management or customer service.
  - Might be good paired with a strong plan.
  Supervisor might also decide to create a strict pipeline run, this is called the pipeline pattern (just chain agents together, each one doing a specific task, and passing the result to the next one).
- Parallel pattern: 
  - Option a: multiple agents work on different parts of a task simultaneously, allowing for faster completion and increased efficiency.
  - Option b: multiple agents work on the same task - using different approaches and different methods - and then compare results to find the best solution.\
    This can be a good solution when it comes to comparing results.
- Feedback loop pattern: \
  A "generator" agent produces an output, that is then scored by a "scorer" agent, that decides whether to pass the output back to the generator for improvement or to accept it as is
  Like for code generation.
- Hierarchical pattern: \
  Agents are organized in a hierarchical structure, with higher-level agents overseeing and coordinating the work of lower-level agents. \
  - Supervisors manage supervisors, who manage workers, who manage other workers, and so on.
  - A top-level orchestrator breaks the goal into domains, mid-level managers handle their domain, workers execute the tasks, and so on.
- Router pattern: \
  An agent acts as a router, directing tasks and information to the appropriate agents based on their capabilities and expertise. \
  This can help to improve efficiency and ensure that tasks are completed by the most qualified agents.
- Shared memory pattern: \
  Multiple agents read-from and write-to a shared memory space, allowing them to share information and coordinate their actions. \
  No direct communication between agents, but they can still collaborate by sharing information in the shared memory.
- Plan-then-execute pattern: \ 
  An agent first creates a plan for how to complete a task, and then executes that plan. This can help to improve efficiency and reduce errors by ensuring that the agent has a clear understanding of the task before it begins working on it.
