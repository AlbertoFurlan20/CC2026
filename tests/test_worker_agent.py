import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from MLAgentBench.multi_agent.whiteboard import Whiteboard
from MLAgentBench.schema import WorkerState


@pytest.mark.asyncio
async def test_worker_agent_writes_to_whiteboard():
    wb = Whiteboard()

    mock_env = MagicMock()
    mock_env.is_final.side_effect = [False, True]
    mock_env.execute.return_value = "loss=0.55\nDone."
    mock_env.research_problem = "Train a classifier"
    mock_env.action_infos = {}
    mock_env.low_level_actions = []
    mock_env.high_level_actions = []

    mock_args = MagicMock()
    mock_args.agent_max_steps = 2
    mock_args.max_steps_in_context = 3
    mock_args.max_observation_steps_in_context = 3
    mock_args.max_retries = 1
    mock_args.llm_name = "qwen2.5-7b-instruct"
    mock_args.fast_llm_name = "qwen2.5-7b-instruct"
    mock_args.log_dir = "/tmp/test_worker_log"
    mock_args.actions_remove_from_prompt = []
    mock_args.actions_add_to_prompt = []
    mock_args.edit_script_llm_name = "qwen2.5-7b-instruct"
    mock_args.edit_script_llm_max_tokens = 1024

    with patch("MLAgentBench.agents.worker_agent.async_complete_text", new=AsyncMock(
        return_value="Thought: test\nAction: Final Answer\nAction Input: {\"final_answer\": \"done\"}"
    )):
        from MLAgentBench.agents.worker_agent import WorkerAgent
        agent = WorkerAgent(worker_id="w0", args=mock_args, env=mock_env, whiteboard=wb)
        await agent.run(mock_env)

    states = wb.read_all()
    assert "w0" in states
