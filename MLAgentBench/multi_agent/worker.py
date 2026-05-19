import asyncio
import os
import time
from argparse import Namespace
from typing import Optional

from MLAgentBench.LLM import async_complete_text, complete_text_fast
from MLAgentBench.multi_agent.whiteboard import Whiteboard
from MLAgentBench.multi_agent.scoring import extract_eval_loss
from MLAgentBench.multi_agent.carbon_tracker import AgentCarbonTracker
from MLAgentBench.schema import Action, WorkerState, WorkerStatus, WhiteboardEntry
from MLAgentBench.agents.agent import Agent


WORKER_SYSTEM_PROMPT = """You are a machine learning research worker agent.
Your goal is to optimize training code to achieve the lowest possible validation loss.
Think step by step, execute code, observe results, and improve iteratively.
Always report validation/eval loss values in your observations."""


class WorkerAgent(Agent):
    """Async ReAct agent that writes per-step state to a shared Whiteboard."""

    def __init__(self, worker_id: str, args: Namespace, env, whiteboard: Whiteboard):
        super().__init__(args, env)
        self.worker_id = worker_id
        self.whiteboard = whiteboard
        self.primary_model = args.llm_name
        self.fast_model = getattr(args, "fast_llm_name", args.llm_name)
        self._state = WorkerState(worker_id=worker_id, model=self.primary_model)
        self._carbon = AgentCarbonTracker(
            worker_id=worker_id,
            log_dir=self.log_dir,
            enabled=getattr(args, "use_codecarbon", False),
            device_index=getattr(args, "device", 0),
        )

    async def _llm_call(self, prompt: str, log_file: str) -> str:
        return await async_complete_text(
            prompt,
            log_file=log_file,
            model=self.primary_model,
            system_prompt=WORKER_SYSTEM_PROMPT,
        )

    async def _report(self, step: int, action: str, observation: str) -> None:
        loss = extract_eval_loss(observation)
        if loss is not None:
            if self._state.best_eval_loss is None or loss < self._state.best_eval_loss:
                self._state = WorkerState(
                    worker_id=self.worker_id,
                    model=self._state.model,
                    status=self._state.status,
                    current_step=step,
                    best_eval_loss=loss,
                    last_actions=self._state.last_actions[-5:] + [action],
                    history=self._state.history,
                )

        entry = WhiteboardEntry(
            worker_id=self.worker_id,
            step=step,
            action=action,
            observation=observation,
            eval_loss=loss,
            timestamp=time.time(),
        )
        await self.whiteboard.append_entry(self.worker_id, entry)
        await self.whiteboard.update_worker_state(self.worker_id, self._state)

    async def run(self, env) -> str:
        self._carbon.start()
        await self.whiteboard.update_worker_state(self.worker_id, self._state)

        while not env.is_final() and len(self.history_steps) < self.args.agent_max_steps:
            curr_step = len(self.history_steps)
            log_file = os.path.join(self.log_dir, f"step_{curr_step}_log.log")

            prompt = self._build_prompt(curr_step)

            entries = None
            valid_response = False
            completion = ""
            for _ in range(self.args.max_retries):
                try:
                    completion = await self._llm_call(prompt, log_file)
                    entries = self.parse_entries(completion, self.valid_format_entires)
                    assert entries["Action"].strip() in self.all_tool_names
                    valid_response = True
                    break
                except Exception:
                    pass

            if not valid_response:
                return f"[Worker {self.worker_id}] No valid response after max_retries"

            action = entries["Action"].strip()
            raw_action_input = entries["Action Input"]

            try:
                action_input = self.parse_action_input(raw_action_input, self.action_infos[action])
            except Exception:
                action_input = raw_action_input

            if isinstance(action_input, dict):
                observation = env.execute(Action(action, action_input))
            else:
                observation = f"ActionInputParsingError: could not parse input for {action}"

            if len(observation) > 5000:
                summarize_log = os.path.join(self.log_dir, f"step_{curr_step}_summary.log")
                observation = self.summarize_observation(
                    self._print_entries(entries), observation, summarize_log
                )

            self.history_steps.append({
                "step_idx": curr_step,
                "action": entries,
                "observation": observation,
            })

            await self._report(curr_step, action, observation)

        emissions, utilization = self._carbon.stop()
        done_state = WorkerState(
            worker_id=self.worker_id,
            model=self._state.model,
            status=WorkerStatus.DONE,
            current_step=len(self.history_steps),
            best_eval_loss=self._state.best_eval_loss,
            last_actions=self._state.last_actions,
            history=self._state.history,
            emissions=emissions,
            utilization=utilization,
        )
        await self.whiteboard.update_worker_state(self.worker_id, done_state)
        return f"[Worker {self.worker_id}] Finished. Best eval loss: {self._state.best_eval_loss}"

    def _build_prompt(self, curr_step: int) -> str:
        prompt = self.initial_prompt
        prompt += "\nNow let's start!\n\n"
        last_steps = self.args.max_steps_in_context
        for idx in range(max(curr_step - last_steps, 0), curr_step):
            action_str = self.print_action(self.history_steps[idx]["action"], self.valid_format_entires)
            prompt += f"\nAssistant:\n{action_str}\nObservation:"
            if curr_step - idx > self.args.max_observation_steps_in_context:
                prompt += "<Done>\n\n"
            else:
                prompt += f"\n```\n{self.history_steps[idx]['observation']}\n```\n\n"
        return prompt

    def _print_entries(self, entries: dict) -> str:
        return self.print_action(entries, self.valid_format_entires)
