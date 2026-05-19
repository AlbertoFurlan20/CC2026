import asyncio
import copy
import os
from argparse import Namespace
from typing import List, Optional

from MLAgentBench.multi_agent.worker import WorkerAgent
from MLAgentBench.multi_agent.supervisor import SupervisorAgent
from MLAgentBench.multi_agent.whiteboard import Whiteboard
from MLAgentBench.multi_agent.workspace_manager import WorkspaceManager
from MLAgentBench.multi_agent.scoring import select_best_worker


class OrchestratorAgent:
    """Spawns N WorkerAgents in parallel, supervises them, returns best result."""

    def __init__(self, args: Namespace, whiteboard: Optional[Whiteboard] = None):
        self.args = args
        self.whiteboard = whiteboard or Whiteboard()
        self._workspace_mgr = WorkspaceManager(base_dir=args.work_dir)
        self._num_workers = getattr(args, "num_workers", 2)
        self._heavy_model = getattr(args, "heavy_llm_name", args.llm_name)
        self._current_model = args.llm_name

    async def run(self, env) -> Optional[str]:
        template_dir = env.work_dir
        upgrade_event = asyncio.Event()

        worker_ids = [f"w{i}" for i in range(self._num_workers)]
        tasks = await self._spawn_workers(worker_ids, template_dir, env, upgrade_event)

        supervisor_task = asyncio.create_task(
            SupervisorAgent(
                whiteboard=self.whiteboard,
                upgrade_event=upgrade_event,
                poll_interval=10.0,
            ).run()
        )

        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )

        if upgrade_event.is_set() and self._current_model != self._heavy_model:
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            self._workspace_mgr.cleanup_all()

            self._current_model = self._heavy_model
            heavy_args = copy.copy(self.args)
            heavy_args.llm_name = self._heavy_model
            heavy_args.fast_llm_name = self._heavy_model

            new_ids = [f"w{i}_heavy" for i in range(self._num_workers)]
            heavy_tasks = await self._spawn_workers(new_ids, template_dir, env, asyncio.Event(), args=heavy_args)
            await asyncio.gather(*heavy_tasks, return_exceptions=True)
        else:
            await asyncio.gather(*pending, return_exceptions=True)

        supervisor_task.cancel()
        try:
            await supervisor_task
        except asyncio.CancelledError:
            pass

        best_id = select_best_worker(self.whiteboard)
        if best_id is None:
            return None
        try:
            return self._workspace_mgr.get_path(best_id)
        except KeyError:
            return None

    async def _spawn_workers(
        self,
        worker_ids: List[str],
        template_dir: str,
        env,
        upgrade_event: asyncio.Event,
        args: Optional[Namespace] = None,
    ) -> List[asyncio.Task]:
        worker_args = args or self.args
        tasks = []
        for wid in worker_ids:
            ws_path = self._workspace_mgr.create(worker_id=wid, template_dir=template_dir)
            per_worker_args = copy.copy(worker_args)
            per_worker_args.work_dir = ws_path
            per_worker_args.log_dir = os.path.join(worker_args.log_dir, wid)

            worker = WorkerAgent(
                worker_id=wid,
                args=per_worker_args,
                env=env,
                whiteboard=self.whiteboard,
            )
            tasks.append(asyncio.create_task(worker.run(env)))
        return tasks
