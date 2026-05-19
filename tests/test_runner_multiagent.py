from unittest.mock import patch, MagicMock
import sys


def test_runner_routes_to_orchestrator_when_num_workers_gt_1():
    import argparse
    with patch("MLAgentBench.runner.OrchestratorAgent") as MockOrch, \
         patch("MLAgentBench.runner.Environment") as MockEnv, \
         patch("MLAgentBench.runner.asyncio") as mock_asyncio:

        mock_env_inst = MagicMock()
        mock_env_inst.__enter__ = MagicMock(return_value=mock_env_inst)
        mock_env_inst.__exit__ = MagicMock(return_value=False)
        mock_env_inst.get_task_description.return_value = ("problem", "cifar10")
        mock_env_inst.low_level_actions = []
        mock_env_inst.high_level_actions = []
        mock_env_inst.read_only_files = []
        MockEnv.return_value = mock_env_inst

        args = argparse.Namespace(
            task="cifar10", log_dir="/tmp/log", work_dir="/tmp/work",
            max_steps=10, max_time=3600, device=0, python="python",
            interactive=False, resume=None, resume_step=0,
            agent_type="ResearchAgent", llm_name="", fast_llm_name="",
            heavy_llm_name="llama-3.1-8b-instruct",
            edit_script_llm_name="", edit_script_llm_max_tokens=1024,
            agent_max_steps=10, actions_remove_from_prompt=[],
            actions_add_to_prompt=[], retrieval=False, valid_format_entires=None,
            max_steps_in_context=3, max_observation_steps_in_context=3,
            max_retries=5, langchain_agent="zero-shot-react-description",
            use_codecarbon=False, num_workers=3,
        )

        from MLAgentBench.runner import run
        run(None, args)

        MockOrch.assert_called_once()
        mock_asyncio.run.assert_called_once()
