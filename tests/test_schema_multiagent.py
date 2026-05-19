from MLAgentBench.schema import WorkerStatus, WorkerState, WhiteboardEntry
import time


def test_worker_status_values():
    assert WorkerStatus.RUNNING.value == "running"
    assert WorkerStatus.STAGNANT.value == "stagnant"
    assert WorkerStatus.DONE.value == "done"
    assert WorkerStatus.CANCELLED.value == "cancelled"


def test_worker_state_defaults():
    ws = WorkerState(worker_id="w1", model="qwen2.5-7b-instruct")
    assert ws.status == WorkerStatus.RUNNING
    assert ws.best_eval_loss is None
    assert ws.history == []


def test_whiteboard_entry():
    entry = WhiteboardEntry(
        worker_id="w1",
        step=3,
        action="Execute Script",
        observation="loss: 0.42",
        eval_loss=0.42,
        timestamp=time.time(),
    )
    assert entry.eval_loss == 0.42
