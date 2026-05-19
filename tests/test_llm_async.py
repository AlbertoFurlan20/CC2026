import pytest
from unittest.mock import MagicMock, patch


def test_complete_text_openai_uses_new_client(monkeypatch):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "hello"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("MLAgentBench.LLM._sync_client", mock_client):
        from MLAgentBench.LLM import complete_text_openai
        result = complete_text_openai("test prompt", model="qwen2.5-7b-instruct")

    mock_client.chat.completions.create.assert_called_once()
    assert result == "hello"


import asyncio


@pytest.mark.asyncio
async def test_async_complete_text_openai_uses_async_client(monkeypatch):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "async hello"

    mock_async_client = MagicMock()

    async def fake_create(**kwargs):
        return mock_response

    mock_async_client.chat.completions.create = fake_create

    with patch("MLAgentBench.LLM._async_client", mock_async_client):
        from MLAgentBench.LLM import async_complete_text_openai
        result = await async_complete_text_openai("test", model="qwen2.5-7b-instruct")

    assert result == "async hello"
