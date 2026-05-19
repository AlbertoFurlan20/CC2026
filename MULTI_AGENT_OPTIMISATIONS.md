# MultiAgent Optimisations

## Role-Based Model Assignment (RBMA)

The three roles in the system have fundamentally different cognitive demands and should be served by different models.

| Role | Cognitive demand | Model profile |
|------|-----------------|---------------|
| **Worker** | Deep reasoning, code generation, long context | Large, high-quality — `llm_name` from difficulty routing |
| **Orchestrator** | Structured decisions, short I/O, low latency | Small, fast — `fast_llm_name` class |
| **Supervisor** | Pattern recognition over structured data, binary signals | Tiny or pure Python — same as fast or smaller |

```python
ROLE_MODEL_MAP = {
    "worker":       args.llm_name,       # e.g. qwen2.5-72b-instruct
    "orchestrator": args.fast_llm_name,  # e.g. qwen2.5-7b-instruct
    "supervisor":   args.fast_llm_name,  # same or dedicated --supervisor-llm-name
}
```
Notes:
- **Key insight:** the Orchestrator and Supervisor are doing control flow, not research. Using the heavy model for them wastes GPU batch slots and causes head-of-line blocking behind Worker generations.
- **Supervisor note:** stagnation *detection* (loss delta threshold + action diversity check) should be pure Python over `WhiteboardEntry` structs — no LLM call at all. Reserve the LLM call only for generating the upgrade-hint injected into the replacement worker's context. This drops Supervisor LLM calls by ~80%.

### RBMA - Optimizatons

 Tier | Technique | Effort | Gain |
|------|-----------|--------|------|
| 0 | Static role→model map | **Zero** | Eliminates waste on Orchestrator/Supervisor |
| 1 | Worker model heterogeneity | Low | Avoids local optima, more diverse exploration |
| 2 | Cascade routing within Worker role | Medium | 45–85% cost reduction, 95% quality retained |
| 3 | LoRA swapping (vLLM multi-LoRA) | Medium | Eliminates model-switch latency, saves VRAM |
| 4 | Adaptive role promotion | Medium-High | Self-correcting control plane |
| 5 | MALBO offline Bayesian search | High | Principled Pareto-optimal assignment |

#### RBMA - Optimization notes
##### Tier 0 — Static heterogeneous assignment (baseline, what's described above)
Fix the model per role at startup. Zero runtime overhead. Already an improvement over a single-model system.

##### Tier 1 — Worker heterogeneity (diversity-driven)
Don't make all workers identical. Spawn workers with intentionally different models to maximize solution diversity and avoid converging to the same local optimum:

```python
# Example with 3 workers
worker_models = [
    "qwen2.5-72b-instruct",   # high quality, slow
    "qwen2.5-7b-instruct",    # fast, more steps per wall-clock
    "llama-3.1-8b-instruct",  # different reasoning style
]
```

Heterogeneous Swarms (Google, NeurIPS 2025, arxiv 2502.04510) formalizes this: it jointly optimizes model roles and weights as a DAG using particle swarm optimization, and shows that **systems benefit directly from the diversity of the initial model pool** — homogeneous workers converge to the same local optimum.

##### Tier 2 — Cascade routing within a role (quality-cost adaptive)
Instead of a fixed model per role, use **cascading**: start each Worker call with the fast model; if the output confidence is below a threshold, escalate to the heavy model. This is especially useful early in a task when steps are simple (file reads, env setup) vs. later when they require deep reasoning (model architecture changes).
The unified cascade routing framework (arxiv 2410.10347) proves optimality conditions for this pattern. In practice: use **perplexity or token entropy** as the escalation signal — probe-based methods are more reliable than self-reported model confidence.
- **Watch out for routing collapse** (EquiRouter, arxiv 2602.03478): naive routers trained on scalar performance scores tend to always escalate to the largest model as budget increases, undermining the cost savings. Use ranking-based routers rather than score-based ones.

##### Tier 3 — LoRA swapping instead of model switching
Rather than switching between full model checkpoints (Orchestrator=7B, Worker=72B), serve one base model and swap LoRA adapters per role. vLLM's multi-LoRA batching serves hundreds of adapters in the same batch with near-zero overhead. This eliminates model-switch latency entirely and reduces GPU memory footprint — the base weights are shared, only the adapter tensors differ.

##### Tier 4 — Adaptive role promotion (dynamic re-assignment)
Start the Orchestrator on the fast model. If it makes a sequence of poor spawn/cancel decisions (workers it promotes stagnate quickly, workers it cancels would have converged), promote the Orchestrator itself to the heavy model for the remainder of the run. Same logic applies in reverse: a Supervisor that consistently emits false-positive stagnation signals gets its LLM call downgraded or disabled.
This is inspired by HieraMAS (arxiv 2602.20229), which proposes hierarchical collaboration where internal model mixtures within each role node are optimized alongside inter-role topology.

##### Tier 5 — MALBO: Bayesian optimization of the full assignment (offline, long-term)
MALBO (arxiv 2511.11788) formalizes RBMA as a multi-objective Bayesian optimization problem over the combinatorial space of (role → model) assignments, optimizing the Pareto front of task performance vs. inference cost. This is an **offline meta-optimization**: run it once over a set of benchmark tasks to find the Pareto-optimal assignment for your specific model pool, then hardcode the result into `task_difficulty.json`.


## Optimizations from plan analysis

| Area | Your idea | Assessment | Priority |
|------|-----------|------------|----------|
| TOON/compact encoding | Partial value | Low — observation summarization beats it | Low |
| gRPC | Not applicable to current arch | Skip unless multi-host | — |
| Context window mgmt | Not yet in plan | Critical gap | **High** |
| vLLM prefix caching | Not mentioned | Free win, just enable the flag | **High** |
| LLM request batching | Not mentioned | Medium complexity, high throughput gain | Medium |
| Speculative decoding | Not mentioned | High gain for ReAct output patterns | Medium |
| Overlay FS for workspaces | Not mentioned | Clean solution for large tasks | Medium |
| Per-worker Whiteboard locks | Not mentioned | Matters at N≥8 | Medium |
| Progressive model routing | Not mentioned | Best ROI on GPU budget | **High** |
| Cross-worker knowledge transfer | Not mentioned | Quality improvement on respawn | Medium |

## From most recent papers

| Trick | Paper | Effort | Expected gain |
|-------|-------|--------|---------------|
| RelayCaching — KV reuse on respawn | arxiv 2603.13289 | Medium | 4x TTFT on worker upgrade |
| ACON typed context compression | arxiv 2510.00615 | Medium | Prevents context bloat, improves reasoning |
| vLLM prefix caching | vLLM docs | **Zero** — one flag | 30-60% TTFT on all workers |
| AgentDropout — graduated worker throttling | arxiv 2503.18891 | Low | Better GPU utilization at N≥4 |
| LoRA swapping instead of model switch | vLLM multi-LoRA | High | Eliminates model-switch latency |
| Optima-style DPO on whiteboard entries | arxiv 2410.08115 | Very high | 90% token reduction long-term |
| Tiered memory (MemGPT pattern) | MemGPT | High | Unlimited worker horizon |