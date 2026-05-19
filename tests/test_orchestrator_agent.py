import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_orchestrator_returns_best_worker_path():
    from MLAgentBench.multi_agent.orchestrator import OrchestratorAgent
    from MLAgentBench.multi_agent.whiteboard import Whiteboard
    from MLAgentBench.schema import WorkerState

    wb = Whiteboard()

    mock_args = MagicMock()
    mock_args.llm_name = "qwen2.5-7b-instruct"
    mock_args.fast_llm_name = "qwen2.5-7b-instruct"
    mock_args.num_workers = 2
    mock_args.heavy_llm_name = "llama-3.1-8b-instruct"
    mock_args.log_dir = "/tmp/test_orch_log"
    mock_args.work_dir = "/tmp/test_orch_work"
    mock_args.task = "cifar10"
    mock_args.agent_max_steps = 1
    mock_args.max_steps_in_context = 3
    mock_args.max_observation_steps_in_context = 3
    mock_args.max_retries = 1
    mock_args.actions_remove_from_prompt = []
    mock_args.actions_add_to_prompt = []
    mock_args.edit_script_llm_name = "qwen2.5-7b-instruct"
    mock_args.edit_script_llm_max_tokens = 1024

    mock_env = MagicMock()
    mock_env.work_dir = "/tmp/template_work"
    mock_env.research_problem = "Train cifar10 classifier"
    mock_env.action_infos = {}
    mock_env.low_level_actions = []
    mock_env.high_level_actions = []
    mock_env.is_final.return_value = True

    async def fake_worker_run(env):
        state = WorkerState(
            worker_id="w0", model="qwen2.5-7b-instruct", best_eval_loss=0.4
        )
        await wb.update_worker_state("w0", state)
        return "[Worker w0] Finished"

    with patch("MLAgentBench.multi_agent.orchestrator.WorkerAgent") as MockWorker, \
         patch("MLAgentBench.multi_agent.orchestrator.WorkspaceManager") as MockMgr:

        mock_worker_inst = MagicMock()
        mock_worker_inst.run = AsyncMock(side_effect=fake_worker_run)
        MockWorker.return_value = mock_worker_inst
        MockMgr.return_value.create.return_value = "/tmp/worker_ws"

        orch = OrchestratorAgent(args=mock_args, whiteboard=wb)
        result = await orch.run(mock_env)

    assert result is not None
