import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import patch, MagicMock
import MLAgentBench.LLM as LLM


def _make_response(content="hello"):
    msg = MagicMock()
    msg.__getitem__ = lambda self, k: {"content": content}[k]
    choice = MagicMock()
    choice.__getitem__ = lambda self, k: {"message": msg}[k]
    resp = MagicMock()
    resp.__getitem__ = lambda self, k: {"choices": [choice]}[k]
    return resp


def test_top_p_injected_when_set():
    LLM.SAMPLING_TOP_P = 0.7
    LLM.SAMPLING_BEST_OF = None
    LLM.SAMPLING_N = None

    captured = {}

    def fake_create(messages, **kwargs):
        captured.update(kwargs)
        return _make_response()

    with patch("openai.ChatCompletion.create", side_effect=fake_create):
        LLM.complete_text_openai("test prompt", model="llama-3.1-8b-instruct")

    try:
        assert captured.get("top_p") == 0.7
    finally:
        LLM.SAMPLING_TOP_P = None


def test_best_of_injected_when_set():
    LLM.SAMPLING_TOP_P = None
    LLM.SAMPLING_BEST_OF = 3
    LLM.SAMPLING_N = None

    captured = {}

    def fake_create(messages, **kwargs):
        captured.update(kwargs)
        return _make_response()

    with patch("openai.ChatCompletion.create", side_effect=fake_create):
        LLM.complete_text_openai("test prompt", model="llama-3.1-8b-instruct")

    try:
        assert captured.get("best_of") == 3
    finally:
        LLM.SAMPLING_BEST_OF = None


def test_n_injected_when_set():
    LLM.SAMPLING_TOP_P = None
    LLM.SAMPLING_BEST_OF = None
    LLM.SAMPLING_N = 2

    captured = {}

    def fake_create(messages, **kwargs):
        captured.update(kwargs)
        return _make_response()

    with patch("openai.ChatCompletion.create", side_effect=fake_create):
        LLM.complete_text_openai("test prompt", model="llama-3.1-8b-instruct")

    try:
        assert captured.get("n") == 2
    finally:
        LLM.SAMPLING_N = None


def test_no_injection_when_all_none():
    LLM.SAMPLING_TOP_P = None
    LLM.SAMPLING_BEST_OF = None
    LLM.SAMPLING_N = None

    captured = {}

    def fake_create(messages, **kwargs):
        captured.update(kwargs)
        return _make_response()

    with patch("openai.ChatCompletion.create", side_effect=fake_create):
        LLM.complete_text_openai("test prompt", model="llama-3.1-8b-instruct")

    try:
        assert "top_p" not in captured
        assert "best_of" not in captured
        assert "n" not in captured
    finally:
        LLM.SAMPLING_TOP_P = None
        LLM.SAMPLING_BEST_OF = None
        LLM.SAMPLING_N = None
