import asyncio
import pytest
from MLAgentBench.multi_agent.whiteboard import Whiteboard
from MLAgentBench.schema import WorkerStatus, WorkerState, WhiteboardEntry
import time


@pytest.mark.asyncio
async def test_write_and_read_all():
    wb = Whiteboard()
    state = WorkerState(worker_id="w1", model="qwen2.5-7b-instruct")
    await wb.update_worker_state("w1", state)
    all_states = wb.read_all()
    assert "w1" in all_states
    assert all_states["w1"].worker_id == "w1"


@pytest.mark.asyncio
async def test_append_entry_and_snapshot():
    wb = Whiteboard()
    entry = WhiteboardEntry(
        worker_id="w1", step=1, action="Execute Script",
        observation="loss=0.5", eval_loss=0.5, timestamp=time.time()
    )
    await wb.append_entry("w1", entry)
    snap = wb.snapshot("w1")
    assert len(snap) == 1
    assert snap[0].eval_loss == 0.5


@pytest.mark.asyncio
async def test_concurrent_writes_are_safe():
    wb = Whiteboard()

    async def write_worker(wid):
        state = WorkerState(worker_id=wid, model="qwen2.5-7b-instruct")
        for _ in range(10):
            await wb.update_worker_state(wid, state)

    await asyncio.gather(*(write_worker(f"w{i}") for i in range(5)))
    assert len(wb.read_all()) == 5
