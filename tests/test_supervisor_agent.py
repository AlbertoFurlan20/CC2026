import asyncio, time
import pytest
from MLAgentBench.multi_agent.whiteboard import Whiteboard
from MLAgentBench.schema import WorkerState, WorkerStatus, WhiteboardEntry
from MLAgentBench.multi_agent.supervisor import SupervisorAgent


def _make_entry(wid, step, action, loss):
    return WhiteboardEntry(
        worker_id=wid, step=step, action=action,
        observation="", eval_loss=loss, timestamp=time.time()
    )


@pytest.mark.asyncio
async def test_supervisor_detects_stagnation():
    wb = Whiteboard()
    state = WorkerState(
        worker_id="w0", model="qwen2.5-7b-instruct",
        last_actions=["Execute Script"] * 6
    )
    await wb.update_worker_state("w0", state)

    upgrade_event = asyncio.Event()
    sup = SupervisorAgent(whiteboard=wb, upgrade_event=upgrade_event,
                          stagnation_window=5, poll_interval=0.05)

    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert upgrade_event.is_set()


@pytest.mark.asyncio
async def test_supervisor_does_not_fire_without_stagnation():
    wb = Whiteboard()
    state = WorkerState(
        worker_id="w0", model="qwen2.5-7b-instruct",
        last_actions=["Execute Script", "Read File", "Edit Script (AI)"]
    )
    await wb.update_worker_state("w0", state)

    upgrade_event = asyncio.Event()
    sup = SupervisorAgent(whiteboard=wb, upgrade_event=upgrade_event,
                          stagnation_window=5, poll_interval=0.05)

    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.15)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert not upgrade_event.is_set()
