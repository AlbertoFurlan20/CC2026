import pytest
from MLAgentBench.multi_agent.scoring import extract_eval_loss, select_best_worker
from MLAgentBench.multi_agent.whiteboard import Whiteboard
from MLAgentBench.schema import WhiteboardEntry, WorkerState
import asyncio, time


def test_extract_eval_loss_from_standard_output():
    output = "Epoch 5/10 - train_loss: 1.23 - val_loss: 0.87\nBest model saved."
    assert extract_eval_loss(output) == pytest.approx(0.87)


def test_extract_eval_loss_from_loss_equals_format():
    output = "loss=0.4231 at step 100"
    assert extract_eval_loss(output) == pytest.approx(0.4231)


def test_extract_eval_loss_returns_none_when_absent():
    assert extract_eval_loss("no loss info here") is None


@pytest.mark.asyncio
async def test_select_best_worker_picks_lowest_loss():
    wb = Whiteboard()
    for wid, loss in [("w0", 0.9), ("w1", 0.3), ("w2", 0.6)]:
        state = WorkerState(worker_id=wid, model="qwen2.5-7b-instruct", best_eval_loss=loss)
        await wb.update_worker_state(wid, state)
    best = select_best_worker(wb)
    assert best == "w1"


@pytest.mark.asyncio
async def test_select_best_worker_ignores_none_loss():
    wb = Whiteboard()
    for wid, loss in [("w0", None), ("w1", 0.5)]:
        state = WorkerState(worker_id=wid, model="qwen2.5-7b-instruct", best_eval_loss=loss)
        await wb.update_worker_state(wid, state)
    best = select_best_worker(wb)
    assert best == "w1"
